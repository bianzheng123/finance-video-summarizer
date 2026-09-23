"""词级三阶段编排器（步骤 2）。

结构：
    batch 间**并行**；每个 batch 含多个话题；batch 内**串行**且链长可变：

        2_1 粗筛 → [2_2 词表判定 → 2_3 联网确认]
                   └── 可跳过 ──┘

    2_1 恒执行；2_2/2_3 在"该话题确实没问题"时跳过（判定在代码侧，见
    `_should_run_2_2`）。

本步收尾做两件确定性的事（零 LLM），产出下游一切阶段的字幕底稿：

1. **纠错**：确证错字原子写进 `text`（纯内存，不回写 ASR 原文件）；
2. **指代物化**：已确认（困惑=false）的指代内联进 `text_llm`
   （`原词（指代）`）。旧实现是渲染期每次重算、只在步骤 4 起生效；物化后
   **第 3 步也能看到已消解的指代**——3_2 话题重置判断"这段到底在讲什么"正需要它。
   `text` 保持纯纠错文本，作为 `text_llm` 物化的基础。

字幕只落盘一处：`llm_analysis/2_clean_subtitle.txt` 取 `text_llm`（纠错 + 指代
内联）——下游真正看到的那份文本，只供人工核对预处理质量，管线不读、禁止进上传
链路。`recognize_subtitle/subtitle_tencent.txt` 是 ASR 原始字幕，**只读**，本步
不再回写。

**收尾还落盘增量黑话库**：`incremental_database/jargon_incremental.json`（位于
`output_dir` 下、与 `llm_analysis/` 平级）。已确证黑话（`困惑=false`）按类型进
`个股黑话`/`板块黑话` 桶，尚未查清楚的黑话（`困惑=true`，即「疑似黑话未收录」）
进 `未解析` 清单——正式名不可采信、只登记类型。无论有无内容，该目录与文件**恒
创建**（下游按目录存在与否判断增量库是否已生成），另有确证错字时再落
`error_words.json` 易错词。

原第四阶段「2_4 关键词与剪枝」已迁为**第 3 步的 3_1**（`step3_refine/`），最终注释
组装随之移到第 3 步——类型终判与关键词都是 3_1 的产出。故本步的剪枝只有 2_1 的
**初判**（作用是给 2_2/2_3 划定工作范围，省掉约 12% 无效的词级判定），终判在 3_1。

检查点（均为 `{"输出": {...}, "结果": ...}` 整合形态）：
    2_1_coarse_screen.json（结果 = 剪枝行号初判）/
    2_2_vocab_decision.json（结果 = **各批 2_2 判定的并集**）/
    2_3_web_confirm.json（结果 = **各批 2_3 判定的并集**）/
    2_corrected_subtitle.json（**整步断点**：纠错 + 指代物化后的字幕与结论集，
        终态判定与三方合并指代都在这里）/
    2_3_vocab_suggestions.json（入库建议，供人工审核）

**阶段检查点只装自己的产出**：2_2 的「结果」里 `待2_3确认` 原样保留，不被 2_3
的结论覆盖——否则同一文件的「输出」与「结果」会自相矛盾，也就没法用它审计
"2_2 当时到底判了什么"。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from ...llm_infra.batch import batch_units
from ...llm_infra.io import read_json, write_integrated, write_json_atomic
from ...llm_infra.config import get_batch_size
from ...llm_infra.resource import fanout
from ...llm_infra.shared_state import SharedCorrectionHints, SharedFindings
from ...llm_infra.clients import WSASearchClient
from ...log_config import step_logger
from ..utils.llm_text_utils import (
    ANNOTATION_KEY,
    _paragraph_referents_by_row,
    _referent_names,
    build_topic_units,
    materialize_llm_text,
    materialize_terminals,
    render_annotation_dict,
    render_windowed_subtitle,
    save_subtitle_tsv,
)
from ..utils.rowid_scope import scope_id_set
from ..schemas import SubtitleRow, TopicSegment, TopicUnit
from .common import (
    HotwordResources,
    _referents_by_row,
    apply_mods_to_rows,
    filter_invalid_mods,
)
from .pinyin_matcher import row_reading
from .vocab_hints import scan_jargon_in_text
from .word_level_stages import (
    WordLevelStages,
    _text_by_row as _text_by_row_from_rendered,
)

logger = logging.getLogger(__name__)

# WSA 每次搜索保留前 N 条（与旧实现一致）
WSA_TOPK = 5
# 2_3 对话轮数上限：两轮搜索 + 一轮收尾（第二轮条件触发，第一轮解决即收尾；轮数用尽未解决的疑似点按现有逻辑保守保留）
STAGE_2_3_MAX_ROUNDS = 3
# row_pinyin 单次调用最多返回的行数（防模型一次要整场读音把上下文顶爆）
MAX_PINYIN_ROWS_PER_CALL = 30


class WordLevelStep:
    """步骤 2 编排器：预清洗 + 词级三阶段 + 纠错回写 + 指代物化。"""

    def __init__(self) -> None:
        self._res = HotwordResources()
        self._findings = SharedFindings()
        self._correction_hints = SharedCorrectionHints()
        self._stages = WordLevelStages(self._res, findings=self._findings)
        self._wsa: WSASearchClient | None = None
        self._subtitle_knowledge: list[dict] = []
        self._publish_date: str = ""

    # ==================== 主入口 ====================

    def process(
        self,
        subtitle_l: list[SubtitleRow],
        segments: list[TopicSegment],
        base_dir: Path,
        output_dir: Path,
        subtitle_knowledge: list[dict] | None = None,
    ) -> tuple[list[SubtitleRow], dict]:
        """词级三阶段主入口，返回 `(纠错并把指代/黑话结构化写进 annotation 的字幕, 结论集)`。

        结论集供第 3 步消费：`wrong_words`（三层错词累积）/ `referents` /
        `pruned`（2_1 初判）/ `mods`（全部纠错记录）/ `confirmed_jargon`（已确证黑话）。
        类型终判与关键词是 3_1 的产出，注释组装在第 3 步；`text_llm` 不在本步物化
        （懒加载），渲染期由 `llm_text_of` 读 annotation 动态内联。

        Args:
            subtitle_l: 完整字幕（组剪枝之后）
            segments: 1_1 + 1_2 合并后的话题段
            base_dir: 分析目录（llm_analysis/），用于断点与落盘
            output_dir: 易错词落盘目录
            subtitle_knowledge: 1_1/1_2 抽出的「字幕内自解释知识」，喂给 2_2 [DATABASE]
                并落盘 incremental_database/subtitle_knowledge.json
        """
        self._subtitle_knowledge = list(subtitle_knowledge or [])
        self._publish_date = self._read_publish_date(output_dir)
        with step_logger("第2步-词级三阶段"):
            cache_file = base_dir / "2_corrected_subtitle.json" if base_dir else None
            if cache_file is not None and cache_file.exists():
                cached = read_json(cache_file)
                if (
                    isinstance(cached, dict)
                    and isinstance(cached.get("结果"), list)
                    and cached["结果"]
                    and isinstance(cached.get("结论集"), dict)
                ):
                    logger.info(
                        "检测到 2_corrected_subtitle.json，跳过整个词级阶段（%d 行）",
                        len(cached["结果"]),
                    )
                    return (
                        [SubtitleRow.model_validate(sub) for sub in cached["结果"]],
                        _revive_conclusions(cached["结论集"]),
                    )

            units = build_topic_units(subtitle_l, segments)
            if not units:
                logger.warning("没有可处理的话题单元，词级阶段跳过")
                return subtitle_l, _empty_conclusions()
            logger.info("词级三阶段开始：%d 个话题单元", len(units))

            # 回放确定性：先载入上次落盘的发现集与同字词纠错提示作基准
            if base_dir is not None:
                sug_path = base_dir / "2_3_vocab_suggestions.json"
                if sug_path.exists():  # 首轮不存在属正常态，不打日志
                    snap = read_json(sug_path)
                    if isinstance(snap, dict):
                        self._findings.load_frozen(snap.get("发现集"))
                        self._correction_hints.load_frozen(snap.get("纠错提示"))

            batch_size = get_batch_size("2_1_粗筛")
            topic_batches = batch_units(units, batch_size, sort_key=lambda u: u.idx)
            logger.info(
                "分批：%d 话题 → %d 批（每批 ≤ %d 个），batch 间并行、batch 内串行",
                len(units), len(topic_batches), batch_size,
            )
            # rowID → 批次映射：回写阶段的丢弃 warning 需标注发生批次（各批已合并，
            # 这里按话题单元行号反查）。非工作单元行（无关/噪音）不在任何 batch。
            row_to_batch: dict[int, int] = {}
            for batch_idx, batch in enumerate(topic_batches):
                for unit in batch:
                    for rid in unit.row_ids:
                        row_to_batch[rid] = batch_idx

            futures = fanout("2_1-2_3", [
                (self._run_chain, (batch, batch_idx, subtitle_l, base_dir), {})
                for batch_idx, batch in enumerate(topic_batches)
            ])
            chain_results: list[dict] = []
            for future, batch in zip(futures, topic_batches):
                try:
                    chain_results.append(future.result())
                except Exception as e:
                    logger.warning(
                        "批（话题 %s）2_1-2_3 链失败：%s——本批按无产出继续",
                        [u.idx for u in batch], e,
                    )
                    chain_results.append({
                        "outputs": {}, "wrong_words": [], "referents": [],
                        "pruned": {"无关": set(), "噪音": set()}, "suggestions": [],
                        "uncertain": [],
                    })

            merged = self._merge_chain_results(chain_results)
            self._persist_chain(base_dir, merged)

            # ===== 收尾一：三层错词合并替换 text（纯 Python）=====
            corrected_subtitle_l, applied_mods = self._apply_corrections(
                subtitle_l, merged["wrong_words"], row_to_batch,
            )
            all_mods = applied_mods

            # ===== 收尾二：指代/黑话/段落指代结构化写进 annotation（懒加载）=====
            # 不物化 text_llm——文本还在变（第 3 步末才叠加关键词类型），渲染期由
            # `llm_text_of` 动态内联（读 annotation["指代"]/["段落指代"]）。第 3 步
            # 末拿到类型后一次性物化。`text` 不动，B 站 SRT 仍无内联注释。
            confirmed_jargon = _confirmed_jargon(merged["referents"])
            processed = _structurize_annotation(
                corrected_subtitle_l,
                merged["referents"],
                merged.get("段落指代") or [],
            )

            # ===== 收尾三：2_3 后仍未解决的疑似点写入 annotation（第三列注释）=====
            # 文本保留原词不动，把"不确定"说明写进 annotation["不确定"]，供第三列渲染。
            processed = _attach_uncertain(processed, merged["uncertain"])

            conclusions = {
                "wrong_words": merged["wrong_words"],
                "referents": merged["referents"],
                "pruned": merged["pruned"],
                "mods": all_mods,
                # 已确证黑话：3_1 的 [DATABASE] 要它。取全场一份（3_1 按批过滤
                # scope），断点复用时也就无需重放 SharedFindings
                "confirmed_jargon": confirmed_jargon,
            }
            if base_dir is not None:
                write_integrated(
                    base_dir / "2_corrected_subtitle.json",
                    {},
                    [sub.model_dump() for sub in processed],
                    extra={"结论集": _dump_conclusions(conclusions)},
                )
            # ===== 收尾三：字幕落盘（唯一一处）=====
            # 落盘 2_clean_subtitle.txt：取动态内联文本（`llm_text_of`——text_llm
            # 尚未物化，渲染期读 annotation 动态内联指代/黑话），供人工核对预处理
            # 质量。**只读产物**：管线自身不读它，且禁止进上传链路。
            # recognize_subtitle/subtitle_tencent.txt 是 ASR 原始字幕，本步不再回写。
            if base_dir is not None:
                base_dir.mkdir(parents=True, exist_ok=True)
                save_subtitle_tsv(
                    processed, base_dir / "2_clean_subtitle.txt", text_key="text_llm",
                )

            self._save_incremental_database(
                merged["wrong_words"], conclusions["confirmed_jargon"], output_dir,
                subtitle_knowledge=self._subtitle_knowledge,
            )
            return processed, conclusions

    @staticmethod
    def _read_publish_date(output_dir: Path | None) -> str:
        """从 `output_dir/metadata.json` 读直播日期（publish_date），缺失返回空串。

        2_3 联网确认用它提示模型「搜索请携带该日期」——金融事实（如「某板块崩了」）
        是时间敏感的，不带日期会搜到别的交易日。元数据缺失只降级为空，不中断纠错。
        """
        if output_dir is None:
            return ""
        meta = read_json(output_dir / "metadata.json")
        if not isinstance(meta, dict):
            return ""
        return (meta.get("publish_date") or "").strip()

    # ==================== batch 内串行链 ====================

    def _run_chain(
        self,
        batch: list[TopicUnit],
        batch_idx: int,
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None,
    ) -> dict:
        """一个 batch 的 2_1 → [2_2 → 2_3] 串行链（漏斗式三字段）。

        疑似点逐层传递（筛剩 + 新增），错词/指代列表逐层累积。每层的错词 + 指代
        局部应用到下一层的 [DATA]（预览性直接替换，立场门禁已在 _validate_* 过、
        锚定校验在收尾统一过），不污染原始 subtitle_l。各层错词发布到全局
        「同字词纠错」提示表，供后续 batch 跨片段互证。
        """
        topic_idxs = [u.idx for u in batch]
        outputs: dict[str, dict] = {}
        batch_row_ids = [rid for u in batch for rid in u.row_ids]
        batch_row_ids_set = set(batch_row_ids)
        subtitle_text = render_windowed_subtitle(subtitle_l, batch_row_ids)
        # 原始字幕 rowID→text（未修改）：同字词纠错 publish 时对原词做全文搜索用。
        text_by_id = {
            int(sub.rowID): sub.text
            for sub in subtitle_l
            if sub.rowID is not None
        }

        # ---------- 2_1 粗筛（恒执行）----------
        correction_hints = self._correction_hints_for(batch, subtitle_text)
        out_2_1 = self._stages.run_2_1(
            batch, subtitle_text, base_dir, batch_idx,
            correction_hints_text=correction_hints,
        )
        outputs[f"2_1_粗筛_b{batch_idx}"] = out_2_1
        self._correction_hints.publish(
            out_2_1["错词列表"], text_by_id, "2_1", batch_row_ids_set,
        )
        logger.info(
            "「2_1 粗筛 b%d」完成: %d 话题, 剪枝 无关%d/噪音%d 行, 疑似点 %d 个, "
            "指代 %d 条, 错词 %d 条",
            batch_idx, len(batch), len(out_2_1["无关行号"]), len(out_2_1["噪音行号"]),
            len(out_2_1["疑似点"]), len(out_2_1["指代列表"]), len(out_2_1["错词列表"]),
        )

        # 三层逐层累积：错词、指代、段落指代；疑似点逐层传递
        all_wrong: list[dict] = list(out_2_1["错词列表"])
        all_referents: list[dict] = list(out_2_1["指代列表"])
        all_paragraph_referents: list[dict] = list(out_2_1.get("段落指代") or [])
        suspects: list[dict] = list(out_2_1["疑似点"])
        suggestions: list[dict] = []
        trace: list[dict] = []
        annotation_text = self._annotation_for(subtitle_l, out_2_1)

        # 2_1 终结出口：嘴瓢（第三列注释）+ 事件描述（事件通道），不进下游 suspects
        all_slips: list[dict] = list(out_2_1.get("嘴瓢") or [])
        all_events: list[dict] = list(out_2_1.get("事件描述") or [])
        if all_events:
            logger.info(
                "「2_1 粗筛 b%d」事件描述 %d 条：跳过词级消歧，走事件主题通道",
                batch_idx, len(all_events),
            )

        # 2_1 确定性修改局部应用 → 2_2 的 [DATA]（错词改字 + 指代/嘴瓢/事件标签内联）
        subtitle_after = self._apply_deterministic_mods(
            subtitle_l, out_2_1["错词列表"], out_2_1["指代列表"],
            slips=all_slips, events=all_events,
        )
        subtitle_text_after = render_windowed_subtitle(subtitle_after, batch_row_ids)

        # ---------- 2_2 词表判定（可跳过）----------
        out_2_2: dict = {"疑似点": suspects, "指代列表": [], "错词列表": [], "段落指代": []}
        run_2_2, skip_reason = self._should_run_2_2(batch, suspects, subtitle_l)
        if not run_2_2:
            logger.info("「2_2 词表判定」话题 %s 跳过：%s", topic_idxs, skip_reason)
        else:
            correction_hints = self._correction_hints_for(
                batch, subtitle_text_after,
            )
            out_2_2 = self._stages.run_2_2(
                suspects, subtitle_text_after, annotation_text,
                base_dir, batch_idx,
                correction_hints_text=correction_hints,
                subtitle_knowledge=self._subtitle_knowledge,
                topics=batch,
                full_text_by_id=text_by_id,
            )
            outputs[f"2_2_词表判定_b{batch_idx}"] = out_2_2
            self._correction_hints.publish(
                out_2_2["错词列表"], text_by_id, "2_2", batch_row_ids_set,
            )
            logger.info(
                "「2_2 词表判定 b%d」完成: %d 疑似点 → %d 剩余, 指代 %d 条, 错词 %d 条",
                batch_idx, len(suspects), len(out_2_2["疑似点"]),
                len(out_2_2["指代列表"]), len(out_2_2["错词列表"]),
            )
            self._publish_findings(out_2_2)

        # 累积 2_2 的错词/指代，疑似点继续下传
        all_wrong.extend(out_2_2["错词列表"])
        all_referents = self._merge_referents(all_referents, out_2_2["指代列表"])
        all_paragraph_referents.extend(out_2_2.get("段落指代") or [])
        suspects = list(out_2_2["疑似点"])

        # 2_2 兜底改判：嘴瓢/事件描述累加（不进 2_3 suspects）
        slips_2_2 = list(out_2_2.get("嘴瓢") or [])
        events_2_2 = list(out_2_2.get("事件描述") or [])
        if slips_2_2:
            logger.info("「2_2 词表判定 b%d」兜底改判嘴瓢 %d 条", batch_idx, len(slips_2_2))
        if events_2_2:
            logger.info("「2_2 词表判定 b%d」兜底改判事件描述 %d 条", batch_idx, len(events_2_2))
        all_slips.extend(slips_2_2)
        all_events.extend(events_2_2)

        # 2_2 确定性修改局部应用 → 2_3 的 [DATA]
        subtitle_after = self._apply_deterministic_mods(
            subtitle_after, out_2_2["错词列表"], out_2_2["指代列表"],
            slips=slips_2_2, events=events_2_2,
        )
        subtitle_text_after = render_windowed_subtitle(subtitle_after, batch_row_ids)

        # ---------- 2_3 联网确认（可跳过）----------
        out_2_3: dict = {"疑似点": suspects, "指代列表": [], "错词列表": [], "段落指代": []}
        if not suspects:
            logger.info("「2_3 联网确认」话题 %s 跳过：2_2 无疑似点", topic_idxs)
        else:
            correction_hints = self._correction_hints_for(
                batch, subtitle_text_after,
            )
            out_2_3, trace = self._stages.run_2_3(
                suspects, subtitle_text_after, annotation_text,
                self._make_tool_executor(subtitle_text_after, batch_idx), base_dir,
                batch_idx, max_rounds=STAGE_2_3_MAX_ROUNDS,
                correction_hints_text=correction_hints,
                topics=batch,
                full_text_by_id=text_by_id,
                publish_date=self._publish_date,
            )
            outputs[f"2_3_联网确认_b{batch_idx}"] = out_2_3
            self._correction_hints.publish(
                out_2_3["错词列表"], text_by_id, "2_3", batch_row_ids_set,
            )
            suggestions = _build_vocab_suggestions(
                out_2_3["指代列表"], trace, self._res.jargon_map,
            )
            candidate_count = sum(
                1 for s in out_2_3["疑似点"] if (s.get("候选") or "").strip()
            )
            logger.info(
                "「2_3 联网确认 b%d」完成: %d 疑似点 → %d 剩余；"
                "入库建议 %d 条，搜索 %d 次，疑似候选 %d 条",
                batch_idx, len(suspects), len(out_2_3["疑似点"]),
                len(suggestions), _count_searches(trace), candidate_count,
            )
            self._publish_findings(out_2_3)

        # 累积 2_3 的错词/指代 + 兜底嘴瓢/事件描述
        all_wrong.extend(out_2_3["错词列表"])
        all_referents = self._merge_referents(all_referents, out_2_3["指代列表"])
        all_paragraph_referents.extend(out_2_3.get("段落指代") or [])
        all_slips.extend(out_2_3.get("嘴瓢") or [])
        all_events.extend(out_2_3.get("事件描述") or [])

        # 关键字轨迹日志：逐个 2_1 疑似点列其在 2_2/2_3 的完整归宿（修改前后 + 依据/违和原因）
        traj_lines: list[str] = []
        for s in out_2_1.get("疑似点") or []:
            rid = int(s["rowID"])
            frag = str(s.get("原先的片段") or "").strip()
            typ = s.get("初判类型") or "疑似"
            r22 = "跳过" if not run_2_2 else _stage_resolution(out_2_2, rid, frag)
            r23 = _stage_resolution(out_2_3, rid, frag)
            traj_lines.append(f"  {rid}「{frag}」[{typ}] 2_2:{r22} → 2_3:{r23}")
        if traj_lines:
            logger.info(
                "「词级轨迹 b%d」%d 个疑似点（搜索 %d 次）：\n%s",
                batch_idx, len(traj_lines), _count_searches(trace),
                "\n".join(traj_lines),
            )

        # 2_3 后仍未解决的疑似点 = 最终不确定（写第三列注释）；带候选的保留候选
        uncertain: list[dict] = [
            {
                "类型": "疑似点",
                "rowID": s["rowID"],
                "词": s["原先的片段"],
                "候选": (s.get("候选") or "").strip() or None,
            }
            for s in out_2_3["疑似点"]
        ]
        # 嘴瓢/口误：语义已明、不动字，只写第三列注释提示下游「可忽略」
        uncertain.extend(
            {
                "类型": "嘴瓢",
                "rowID": s["rowID"],
                "词": s["原先的片段"],
            }
            for s in all_slips
        )
        # 被立场词黑名单拦下的错词，降级为「不确定+候选」（不改字面，保留纠错线索供人工核对）
        blocked_wrong: list[dict] = []
        for out in (out_2_1, out_2_2, out_2_3):
            blocked_wrong.extend(out.get("立场门禁拦截", []))
        seen_blocked: set[tuple[int, str]] = set()
        for b in blocked_wrong:
            key = (b["rowID"], b["原先的词"])
            if key in seen_blocked:
                continue
            seen_blocked.add(key)
            uncertain.append({
                "类型": "错词",
                "rowID": b["rowID"],
                "词": b["原先的词"],
                "候选": b["候选"],
            })

        return {
            "outputs": outputs,
            "wrong_words": all_wrong,
            "referents": all_referents,
            "段落指代": all_paragraph_referents,
            "pruned": {
                "无关": set(out_2_1["无关行号"]),
                "噪音": set(out_2_1["噪音行号"]),
            },
            "suggestions": suggestions,
            "uncertain": uncertain,
        }

    def _should_run_2_2(
        self,
        batch: list[TopicUnit],
        suspects: list[dict],
        subtitle_l: list[SubtitleRow],
    ) -> tuple[bool, str]:
        """2_2 跳过判定（代码侧，两条件全满足才跳过）。

        第 2 条是**兜底**且不可省：2_1 没有词表，它可能自信地把「电梯」当普通名词而
        根本不标疑似点。只要话题文本命中任何词表黑话别名（确定性字符串匹配，零 LLM），
        就强制进 2_2 让模型带词表判一次——否则这类词会直接漏过。
        """
        if suspects:
            return True, ""
        topic_text = "\n".join(
            sub.text
            for u in batch
            for _rid, sub in u.rows
        )
        hits = scan_jargon_in_text(self._res.jargon_map, topic_text)
        if hits:
            return True, ""
        return False, "2_1 无疑似点、词表扫描未命中"

    # ==================== 确定性后处理 ====================

    @staticmethod
    def _apply_deterministic_mods(
        subtitle_l: list[SubtitleRow],
        wrong_words: list[dict],
        referents: list[dict],
        slips: list[dict] | None = None,
        events: list[dict] | None = None,
    ) -> list[SubtitleRow]:
        """把某阶段的确定性修改局部应用到字幕副本（预览性，不污染原始 subtitle_l）。

        错词 → `apply_mods_to_rows` 改 `text`；指代 → `materialize_llm_text` 物化
        `text_llm`。返回全场字幕浅拷贝副本，供渲染下游阶段的 [DATA]（读 text_llm）。
        立场门禁已在 `_validate_*` 阶段过；锚定校验在收尾统一过。
        """
        rows = [
            (int(sub.rowID), sub)
            for sub in subtitle_l
            if sub.rowID is not None
        ]
        mods = [
            {
                "rowID": w["rowID"],
                "修正前词组": w["原先的词"],
                "修正后词组": w["修改的词"],
            }
            for w in wrong_words
        ]
        corrected_by_id = apply_mods_to_rows(rows, mods)
        subtitle_l_corrected = [
            corrected_by_id.get(int(sub.rowID), sub)
            if sub.rowID is not None else sub
            for sub in subtitle_l
        ]
        out = materialize_llm_text(subtitle_l_corrected, referents)
        if slips:
            out = materialize_terminals(out, slips, "slip")
        if events:
            out = materialize_terminals(out, events, "event")
        return out

    def _apply_corrections(
        self,
        subtitle_l: list[SubtitleRow],
        wrong_words: list[dict],
        row_to_batch: dict[int, int] | None = None,
    ) -> tuple[list[SubtitleRow], list[dict]]:
        """把三层合并的错词列表应用到字幕（锚定校验，零 LLM）。

        `wrong_words` 是三层（2_1/2_2/2_3）错词列表的累积，条目 `{rowID, 原先的词,
        修改的词}`。立场门禁已在 `_validate_wrong_words` 过，这里再过锚定校验。
        """
        rows = [
            (int(sub.rowID), sub)
            for sub in subtitle_l
            if sub.rowID is not None
        ]
        mods: list[dict] = [
            {
                "rowID": w["rowID"],
                "修正前词组": w["原先的词"],
                "修正后词组": w["修改的词"],
            }
            for w in wrong_words
        ]
        kept = filter_invalid_mods(rows, mods, row_to_batch)
        dropped = len(mods) - len(kept)
        corrected_by_id = apply_mods_to_rows(rows, kept)
        out = [
            corrected_by_id.get(int(sub.rowID), sub)
            if sub.rowID is not None else sub
            for sub in subtitle_l
        ]
        logger.info(
            "字幕回写：应用 %d 处修改（%d 处被校验丢弃）", len(kept), dropped,
        )
        return out, kept

    # ==================== 合并与落盘 ====================

    @staticmethod
    def _merge_chain_results(results: list[dict]) -> dict:
        outputs: dict[str, dict] = {}
        wrong_words: list[dict] = []
        referents: list[dict] = []
        irrelevant: set[int] = set()
        noise: set[int] = set()
        suggestions: list[dict] = []
        uncertain: list[dict] = []
        paragraph_referents: list[dict] = []
        for r in results:
            outputs.update(r["outputs"])
            wrong_words.extend(r["wrong_words"])
            referents.extend(r["referents"])
            paragraph_referents.extend(r.get("段落指代") or [])
            irrelevant |= r["pruned"]["无关"]
            noise |= r["pruned"]["噪音"]
            suggestions.extend(r["suggestions"])
            uncertain.extend(r.get("uncertain") or [])
        return {
            "outputs": outputs,
            "wrong_words": wrong_words,
            "referents": referents,
            "段落指代": _dedupe_paragraph_referents(paragraph_referents),
            "pruned": {"无关": irrelevant, "噪音": noise},
            "suggestions": suggestions,
            "uncertain": uncertain,
        }

    def _persist_chain(self, base_dir: Path | None, merged: dict) -> None:
        """落盘 2_1-2_3 的检查点与入库建议（含共享发现集，供回放确定性）。

        **每个文件的「结果」只放该阶段自己的产出**（全批并集），不放下游覆盖后的
        终态：2_2 的「结果」里 `待2_3确认` 就该原样留着，它是 2_2 的真实结论。
        终态判定与三方合并后的指代在 `2_corrected_subtitle.json` 的「结论集」里，
        不在这三个文件的职责范围内。
        """
        if base_dir is None:
            return
        out_2_1 = {k: v for k, v in merged["outputs"].items() if k.startswith("2_1_")}
        out_2_2 = {k: v for k, v in merged["outputs"].items() if k.startswith("2_2_")}
        out_2_3 = {k: v for k, v in merged["outputs"].items() if k.startswith("2_3_")}
        write_integrated(base_dir / "2_1_coarse_screen.json", out_2_1, {
            "无关行号": sorted(merged["pruned"]["无关"]),
            "噪音行号": sorted(merged["pruned"]["噪音"]),
        })
        write_integrated(
            base_dir / "2_2_vocab_decision.json", out_2_2,
            _merge_stage_outputs(out_2_2),
        )
        write_integrated(
            base_dir / "2_3_web_confirm.json", out_2_3,
            _merge_stage_outputs(out_2_3),
        )
        correction_frozen = self._correction_hints.freeze()
        if merged["suggestions"] or self._findings.snapshot() or correction_frozen:
            write_json_atomic(base_dir / "2_3_vocab_suggestions.json", {
                "入库建议": merged["suggestions"],
                "发现集": self._findings.freeze(),
                "纠错提示": correction_frozen,
            })
            if merged["suggestions"]:
                logger.info(
                    "入库建议 %d 条已落盘（需人工审核后补入词表，不自动写库）",
                    len(merged["suggestions"]),
                )

    # ==================== 辅助 ====================

    def _publish_findings(self, stage_output: dict) -> None:
        """把本阶段已确证黑话（指代列表里的黑话/绰号映射）并入共享发现表，供后批互证。"""
        entries = _confirmed_jargon(stage_output.get("指代列表") or [])
        if entries:
            self._findings.add_many(entries)

    def _correction_hints_for(
        self, batch: list[TopicUnit], current_text: str = "",
    ) -> str:
        """本批话题范围内的「跨片段同字词纠错提示」文本（进 [TASK]）。

        只含「来自其他批次、且本批当前文本仍出现未修改」的错词提示。
        """
        scope = scope_id_set([rid for u in batch for rid in u.row_ids])
        current_by_id = (
            _text_by_row_from_rendered(current_text) if current_text else {}
        )
        return self._correction_hints.hints_for(scope, current_by_id)

    @staticmethod
    def _annotation_for(subtitle_l: list[SubtitleRow], out_2_1: dict) -> str:
        """2_1 剪枝初判 + 错词纠错痕迹渲染成 [ANNOTATION]（让下游知道哪些行已剪、哪些字已改）。"""
        ann: dict[int, dict] = {}
        for rid in out_2_1["无关行号"]:
            ann[rid] = {"类型": "无关"}
        for rid in out_2_1["噪音行号"]:
            ann[rid] = {"类型": "噪音"}
        return render_annotation_dict(ann, corrections=out_2_1.get("错词列表"))

    @staticmethod
    def _merge_referents(base: list[dict], override: list[dict]) -> list[dict]:
        """指代三层合并：同 (rowID, 原先的词, 出现序号) 以后来者为准。"""
        merged = {
            (r["rowID"], r["原先的词"], r.get("出现序号", 1)): r for r in base
        }
        for r in override:
            merged[(r["rowID"], r["原先的词"], r.get("出现序号", 1))] = r
        return list(merged.values())

    def _make_tool_executor(
        self, subtitle_text: str, batch_idx: int,
    ) -> Callable[[str, dict], str]:
        """构造 2_3 的工具执行器：`web_search`（联网）+ `row_pinyin`（本地读音）。

        `row_pinyin` 需要本批 [DATA] 的行文本，故做成闭包按批绑定——它查的必须是
        **模型眼前那份切片**里的行，用全场字幕会让越界行号也能查到读音，
        与"产出只能用 [DATA] 中出现过的 rowID"的协议冲突。`batch_idx` 仅用于
        工具日志标注批次。
        """
        text_by_id = _text_by_row_from_rendered(subtitle_text)

        def executor(name: str, args: dict) -> str:
            if name == "row_pinyin":
                return self._exec_row_pinyin(args, text_by_id, batch_idx)
            return self._exec_web_search(name, args)

        return executor

    @staticmethod
    def _exec_row_pinyin(
        args: dict, text_by_id: dict[int, str], batch_idx: int,
    ) -> str:
        """行读音查询（纯本地 pypinyin 计算：不走网络、不过 RESOURCE_MANAGER 闸）。"""
        raw = (args or {}).get("row_ids")
        if not isinstance(raw, list) or not raw:
            return json.dumps({"错误": "缺少 row_ids（应为整数数组）"}, ensure_ascii=False)
        readings: dict[str, str] = {}
        missing: list = []
        for rid in raw[:MAX_PINYIN_ROWS_PER_CALL]:
            try:
                key = int(rid)
            except (TypeError, ValueError):
                missing.append(rid)
                continue
            text = text_by_id.get(key)
            if not text:
                missing.append(rid)
                continue
            readings[str(key)] = row_reading(text)
        payload: dict = {"行读音": readings}
        if missing:
            # 回灌越界行号而非静默忽略：模型据此知道该换行号，而不是以为查过了
            payload["未找到的行号"] = missing
            payload["说明"] = "未找到的行号不在本次 [DATA] 切片内，请只查 [DATA] 里真实出现的行"
        if len(raw) > MAX_PINYIN_ROWS_PER_CALL:
            payload["说明"] = (
                f"一次最多查 {MAX_PINYIN_ROWS_PER_CALL} 行，已只返回前 "
                f"{MAX_PINYIN_ROWS_PER_CALL} 行"
            )
        logger.info(
            "「2_3 row_pinyin」批次 b%d 调用拼音 %d 行（%d 个未找到）",
            batch_idx, len(readings), len(missing),
        )
        return json.dumps(payload, ensure_ascii=False)

    def _exec_web_search(self, name: str, args: dict) -> str:
        """2_3 的 web_search 工具执行（WSA 检索 + URL 去重 + TopK 截断）。"""
        if name != "web_search":
            return json.dumps({"错误": f"未知工具 {name}"}, ensure_ascii=False)
        query = (args or {}).get("query") or ""
        if not query:
            return json.dumps({"错误": "缺少 query"}, ensure_ascii=False)
        if self._wsa is None:
            self._wsa = WSASearchClient()
        try:
            results = self._wsa.search(query) or []
        except Exception as e:  # noqa: BLE001 —— 网络请求，失败回灌错误让模型换词再搜
            logger.warning("WSA 搜索「%s」失败：%s", query, e)
            return json.dumps({"搜索词": query, "错误": str(e)}, ensure_ascii=False)
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in results:
            url = (r.get("url") or "").strip()
            if url and url in seen:
                continue
            if url:
                seen.add(url)
            deduped.append(r)
            if len(deduped) >= WSA_TOPK:
                break
        return json.dumps(
            {"搜索词": query, "搜索结果": deduped}, ensure_ascii=False,
        )

    def _save_incremental_database(
        self,
        wrong_words: list[dict],
        confirmed_jargon: list[dict],
        output_dir: Path | None,
        subtitle_knowledge: list[dict] | None = None,
    ) -> None:
        """收尾落盘 `incremental_database/`（增量黑话库 + 易错词 + 字幕知识），目录恒创建。

        1. `jargon_incremental.json`：已确证黑话（词表/联网确证）按 `类型` 进
           `个股黑话`/`板块黑话` 桶（新架构无「未解析」黑话，困惑恒 false）。
        2. `error_words.json`：确证错字的易错词映射（错词 → 正确词），供人工回看。
        3. `subtitle_knowledge.json`：1_1/1_2 抽出的「字幕内自解释知识」（黑话→
           字幕解释），供人工审核后补入正式词表。

        无论有无内容都创建目录并写 `jargon_incremental.json`（下游按目录存在与否
        判断增量库是否已生成）。
        """
        if output_dir is None:
            return
        target_dir = output_dir / "incremental_database"
        target_dir.mkdir(parents=True, exist_ok=True)

        stock: dict[str, dict] = {}
        sector: dict[str, dict] = {}
        for j in confirmed_jargon:
            word = j.get("词")
            formal = (j.get("正式名") or "").strip()
            jargon_type = j.get("类型")
            if not word or not formal:
                continue
            if jargon_type == "板块黑话":
                sector[word] = {"正式板块名": formal}
            else:
                stock[word] = {"正式名": formal}
        write_json_atomic(target_dir / "jargon_incremental.json", {
            "个股黑话": stock,
            "板块黑话": sector,
            "未解析": [],
        })
        logger.info(
            "增量黑话库落盘：已解析 %d（个股 %d / 板块 %d）",
            len(stock) + len(sector), len(stock), len(sector),
        )

        words: dict[str, list[str]] = {}
        for w in wrong_words:
            fixed = (w.get("修改的词") or "").strip()
            before = (w.get("原先的词") or "").strip()
            if not fixed or not before:
                continue
            words.setdefault(fixed, [])
            if before not in words[fixed]:
                words[fixed].append(before)
        if words:
            write_json_atomic(target_dir / "error_words.json", words)
            logger.info("易错词 %d 组已落盘", len(words))

        if subtitle_knowledge:
            write_json_atomic(
                target_dir / "subtitle_knowledge.json", subtitle_knowledge,
            )
            logger.info(
                "字幕知识落盘：subtitle_knowledge.json 共 %d 条",
                len(subtitle_knowledge),
            )


# ==================== 模块级纯函数 ====================


def _dedupe_paragraph_referents(items: list[dict]) -> list[dict]:
    """段落指代按 rowID 数组去重（同组行保留首条）。"""
    out: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for pr in items:
        rows = pr.get("rowID") or []
        if not rows:
            continue
        key = tuple(sorted(rows))
        if key in seen:
            continue
        seen.add(key)
        out.append(pr)
    return out


def _format_basis(item: dict) -> str:
    """有「依据」字段时返回「[依据]」，否则返回空串。"""
    basis = (item.get("依据") or "").strip()
    return f"[{basis}]" if basis else ""


def _stage_resolution(stage: dict, rid: int, frag: str) -> str:
    """聚合疑似点 (rid, frag) 在某阶段（out_2_2/out_2_3）的全部归宿，返回可读串。

    依次收集错词 / 指代 / 嘴瓢 / 事件描述 / 立场门禁拦截的命中项，多个命中用
    「 + 」连接（黑话错别字拆两列表时错词 + 指代同时打印，不互相遮蔽）。
    匹配条件：rowID 相同且词相等或互为子串；错词/指代取 `原先的词`，嘴瓢/事件
    描述取 `原先的片段`。

    均未命中时：仍在疑似点 → 「保留」/「保留(候选:X)」；否则 → 「未匹配(归宿不明)」。
    """
    parts: list[str] = []

    for w in stage.get("错词列表") or []:
        word = (w.get("原先的词") or "").strip()
        if w.get("rowID") == rid and word and (
            word == frag or word in frag or frag in word
        ):
            fixed = (w.get("修改的词") or "").strip()
            parts.append(f"错词「{word}」→「{fixed}」{_format_basis(w)}")

    for r in stage.get("指代列表") or []:
        word = (r.get("原先的词") or "").strip()
        if r.get("rowID") == rid and word and (
            word == frag or word in frag or frag in word
        ):
            names = "、".join(_referent_names(r.get("指代")))
            parts.append(f"指代「{word}」→「{names}」{_format_basis(r)}")

    for s in stage.get("嘴瓢") or []:
        word = (s.get("原先的片段") or "").strip()
        if s.get("rowID") == rid and word and (
            word == frag or word in frag or frag in word
        ):
            reason = (s.get("违和原因") or "").strip()
            parts.append(f"嘴瓢「{word}」(不改字){f'[{reason}]' if reason else ''}")

    for e in stage.get("事件描述") or []:
        word = (e.get("原先的片段") or "").strip()
        if e.get("rowID") == rid and word and (
            word == frag or word in frag or frag in word
        ):
            reason = (e.get("违和原因") or "").strip()
            parts.append(
                f"事件描述「{word}」(不改字){f'[{reason}]' if reason else ''}"
            )

    for b in stage.get("立场门禁拦截") or []:
        word = (b.get("原先的词") or "").strip()
        if b.get("rowID") == rid and word and (
            word == frag or word in frag or frag in word
        ):
            cand = (b.get("候选") or "").strip()
            parts.append(f"立场拦截「{word}」候选「{cand}」(不改字)")

    if parts:
        return " + ".join(parts)

    for s in stage.get("疑似点") or []:
        if s.get("rowID") == rid and (s.get("原先的片段") or "").strip() == frag:
            cand = (s.get("候选") or "").strip()
            return f"保留(候选:{cand})" if cand else "保留"

    return "未匹配(归宿不明)"


def _confirmed_jargon(referents: list[dict]) -> list[dict]:
    """从三层合并的指代列表提取「已确证黑话」（黑话/绰号 → 正式名），供 3_1 关键词注入。

    只收「原先的词 ≠ 指代 且 依据 ∈ {数据库, 联网}」的条目——有词表/联网证据的
    语义映射（黑话/绰号），排除代词消解（依据=上下文）与 2_1 的上下文推断。
    形态对齐 3_1 关键词条目（困惑恒 false，新架构无未解析黑话）。
    """
    out: list[dict] = []
    for r in referents:
        word = (r.get("原先的词") or "").strip()
        formals = _referent_names(r.get("指代"))
        basis = r.get("依据") or ""
        if not word or not formals:
            continue
        if basis not in {"数据库", "联网"}:
            continue
        # 一对多：每个正式名拆成一条已确证黑话，逐条过实体名门禁（单名无顿号，通过）
        for formal in formals:
            if word == formal:
                continue
            if not _looks_like_entity_name(formal):
                logger.warning(
                    "row%s 指代「%s」的正式名不是实体名形态，不收录为已确证黑话：%s",
                    r.get("rowID"), word, formal[:40],
                )
                continue
            out.append({
                "rowID": r["rowID"],
                "词": word,
                "正式名": formal,
                "类型": _guess_jargon_type(formal),
                "依据": basis,
                "困惑": False,
                "困惑原因": "",
            })
    return out


# 正式名的实体名上限（与 `2_3_联网确认.json` 的 maxLength 同口径）。词表实测正式名
# 最长 6 字，20 留足余量给「英伟达供应链」这类合法长名
MAX_FORMAL_NAME_LEN = 20
# 解释性描述的判据：实体名里不会出现句读标点
_DESCRIPTION_PUNCT = frozenset("，。；：、（）()「」【】\"'…")
# "多指…" 这类释义引导词：出现即说明这是词典式解释，不是实体名
_GLOSS_MARKERS = ("多指", "泛指", "意指", "常指", "指的是", "的说法", "的叫法")


def _looks_like_entity_name(formal: str) -> bool:
    """正式名是否像一个能独立当主题名用的实体名（公司 / 板块 / 概念 / 资产）。

    只做**形状**判断，不判语义——目的是拦住「事件描述 / 修辞释义」被当成黑话正式名。
    这类正式名会成为第 4 步的主题名，一旦放行，主题清单里就会出现整句话。

    判据三条（任一不满足即否）：长度 ≤ `MAX_FORMAL_NAME_LEN`、无句读标点、无释义
    引导词。`/` 刻意不算标点——「航天/商业航天板块」是合法的并列实体名。
    """
    name = formal.strip()
    if not name or len(name) > MAX_FORMAL_NAME_LEN:
        return False
    if _DESCRIPTION_PUNCT & set(name):
        return False
    return not any(m in name for m in _GLOSS_MARKERS)


# 板块/概念类正式名的词尾特征。`疑似黑话未收录` 的正式名是模型新造的，**必然不在
# 任何词表里**，纯查表永远判不出板块——只能看词形
_SECTOR_NAME_SUFFIXES = ("板块", "概念", "行业", "产业链", "供应链", "指数")


def _guess_jargon_type(formal: str) -> str:
    """由正式名反推关键词类型（板块黑话 / 个股黑话）。

    三级判据：板块表命中 → 个股表命中 → **词尾特征**。第三级不可省：
    `疑似黑话未收录` 的正式名是模型新造的名字，两个词表都不可能收录它，
    只查表会把「韩国半导体板块」这类明显的板块名一律判成 `个股黑话`。

    都判不出时按 `个股黑话` 兜底（「绝不丢弃」优先，且个股是更常见的情形）。
    """
    from ...knowledge_base.jargon_sector_manager import get_sector_manager
    from ...knowledge_base.jargon_stock_manager import get_stock_manager

    try:
        if formal in get_sector_manager().get_all_formal_names():
            return "板块黑话"
        if formal in get_stock_manager().get_all_formal_names():
            return "个股黑话"
    except Exception as e:  # noqa: BLE001 —— 词表不可用时退到词形判据
        logger.warning("词表不可用，正式名「%s」的类型改按词形判：%s", formal, e)
    if formal.endswith(_SECTOR_NAME_SUFFIXES):
        return "板块黑话"
    return "个股黑话"


def _structurize_annotation(
    subtitle_l: list[SubtitleRow],
    referents: list[dict],
    paragraph_referents: list[dict],
) -> list[SubtitleRow]:
    """第 2 步末：指代/黑话/段落指代结构化写进 annotation（不物化 text_llm，懒加载）。

    供 3_1/3_2 渲染 [DATA] 时 `llm_text_of` 动态内联。类型终判与关键词是 3_1 的
    产出，第 3 步末 `_assemble` 会重建完整 annotation。已有 annotation（如第 1 步
    组剪枝合成的「类型」）保留不动。
    """
    refs_by_row = _referents_by_row(referents)
    pr_by_row = _paragraph_referents_by_row(paragraph_referents)
    out: list[SubtitleRow] = []
    for sub in subtitle_l:
        rid = sub.rowID
        refs = refs_by_row.get(int(rid)) if rid is not None else None
        prs = pr_by_row.get(int(rid)) if rid is not None else None
        if not refs and not prs:
            out.append(sub)
            continue
        ann = sub.annotation if isinstance(sub.annotation, dict) else {}
        if refs:
            ann["指代"] = refs
        if prs:
            ann["段落指代"] = prs
        out.append(sub.model_copy(update={ANNOTATION_KEY: ann}))
    return out


def _attach_uncertain(
    subtitle_l: list[SubtitleRow], uncertain: list[dict],
) -> list[SubtitleRow]:
    """把 2_3 后仍不确定的错词/指代写入 annotation["不确定"]（第三列注释）。

    文本保留原词不动，只把「不确定」说明挂到该行注释，供步骤 4 起第三列渲染随行携带。
    """
    if not uncertain:
        return subtitle_l
    by_row: dict[int, list[dict]] = {}
    for u in uncertain:
        by_row.setdefault(int(u["rowID"]), []).append(u)
    out: list[SubtitleRow] = []
    for sub in subtitle_l:
        rid = sub.rowID
        entries = by_row.get(int(rid)) if rid is not None else None
        if not entries:
            out.append(sub)
            continue
        ann = sub.annotation
        if not isinstance(ann, dict):
            ann = {}
        merged = list(ann.get("不确定") or [])
        for e in entries:
            item: dict = {"类型": e["类型"], "词": e["词"]}
            if e.get("候选"):
                item["候选"] = e["候选"]
            merged.append(item)
        out.append(sub.model_copy(update={ANNOTATION_KEY: {**ann, "不确定": merged}}))
    return out


def _empty_conclusions() -> dict:
    """无话题单元时的空结论集（形状与正常产出一致，下游无需判空）。"""
    return {
        "wrong_words": [], "referents": [],
        "pruned": {"无关": set(), "噪音": set()},
        "mods": [], "confirmed_jargon": [],
    }


def _dump_conclusions(conclusions: dict) -> dict:
    """结论集转可 JSON 序列化形态（`pruned` 的 set → 升序 list）。"""
    return {
        **conclusions,
        "pruned": {
            "无关": sorted(conclusions["pruned"]["无关"]),
            "噪音": sorted(conclusions["pruned"]["噪音"]),
        },
    }


def _revive_conclusions(dumped: dict) -> dict:
    """从检查点还原结论集（`pruned` 的 list → set）。

    断点复用时必须还原成 set：第 3 步用它做集合运算（`pruned["无关"] & scope`），
    拿 list 会静默抛 TypeError 到批级 except 里，变成"3_1 整批失败"。
    """
    pruned = dumped.get("pruned") or {}
    return {
        "wrong_words": dumped.get("wrong_words") or [],
        "referents": dumped.get("referents") or [],
        "pruned": {
            "无关": scope_id_set(pruned.get("无关")),
            "噪音": scope_id_set(pruned.get("噪音")),
        },
        "mods": dumped.get("mods") or [],
        "confirmed_jargon": dumped.get("confirmed_jargon") or [],
    }


def _count_searches(trace: list[dict]) -> int:
    """统计 trace 里模型实际发起的 web_search 次数。"""
    n = 0
    for round_rec in trace or []:
        n += len(round_rec.get("工具调用") or [])
    return n


def _build_vocab_suggestions(
    referents: list[dict],
    trace: list[dict],
    jargon_map: dict[str, list[str]],
) -> list[dict]:
    """从 2_3 指代列表里「正式名不在词表」的条目生成入库建议（新黑话，代码生成）。

    `证据搜索词` 取自 trace 里模型**实际用过**的 query——比模型自报的更可信。

    trace 的形状由 `call_conversation` 的返回值定义：**每个工具调用一项**，
    英文键 `{"round", "name", "args", "result_chars"}`。注意它与落盘日志里
    按轮聚合的中文键（`{"工具调用": [{"名称", "参数"}]}`）不是一回事。
    """
    queries: list[str] = []
    for tc in trace or []:
        if not isinstance(tc, dict):
            continue
        q = str((tc.get("args") or {}).get("query") or "").strip()
        if q and q not in queries:
            queries.append(q)
    # 按 (别名, 正式名) 去重：一对多黑话拆成多条（每个正式名一条），同一黑话整段
    # 反复出现时只保留一条、行号合并。人工审核要的是「有哪些词待入库」。
    out: list[dict] = []
    by_key: dict[tuple[str, str], dict] = {}
    for r in referents:
        word = (r.get("原先的词") or "").strip()
        formals = _referent_names(r.get("指代"))
        if not word or not formals:
            continue
        rid = r.get("rowID")
        for formal in formals:
            if word == formal:
                continue
            if not _looks_like_entity_name(formal):
                continue
            key = (word, formal)
            existing = by_key.get(key)
            if existing is not None:
                if rid is not None and rid not in existing["出现行号"]:
                    existing["出现行号"].append(rid)
                continue
            entry = {
                "类型": _guess_jargon_type(formal),
                "别名": word,
                "正式名": formal,
                "出现行号": [rid] if rid is not None else [],
                "证据搜索词": queries,
                "判定理由": "",
                "已在词表": word in jargon_map,
            }
            by_key[key] = entry
            out.append(entry)
    return out


def _merge_stage_outputs(outputs: dict[str, dict]) -> dict:
    """合并某阶段各批输出的三字段（疑似点/指代列表/错词列表），供检查点「结果」落盘。"""
    suspects: list[dict] = []
    referents: list[dict] = []
    wrong_words: list[dict] = []
    for v in outputs.values():
        if not isinstance(v, dict):
            continue
        suspects.extend(v.get("疑似点") or [])
        referents.extend(v.get("指代列表") or [])
        wrong_words.extend(v.get("错词列表") or [])
    return {"疑似点": suspects, "指代列表": referents, "错词列表": wrong_words}
