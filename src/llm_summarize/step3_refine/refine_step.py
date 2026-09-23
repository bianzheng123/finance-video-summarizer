"""第 3 步编排器：3_1 剪枝与关键词终判 ∥ 3_2 话题重置（两者并行）。

```
        ┌── 3_1 关键词与剪枝（按话题批，恒执行）──┐
第 2 步 ┤                                        ├→ 注释组装 + 段列表更新
        └── 3_2 话题重置（按困惑段聚成的单元）───┘
```

## 为什么这两件事在同一步且并行

两者都是"在已纠错、指代已内联的字幕上重做一次判定"，但**互不依赖**：

- 3_1 修正**逐行类型**（重点/无关/噪音）与提取关键词，产出按 rowID 键控；
- 3_2 修正**话题段的分割方式**，产出按段区间键控。

3_1 的分批是按 **3_2 之前**的段做的——这是正确的、不是 bug：3_1 的工作范围是
全部可分析行的并集，段列表只决定"怎么分批"（哪些行凑一次调用），不决定"处理哪些
行"。批的窗口换了不影响任何一行的判定结果，故无需等 3_2。反过来 3_2 也不需要
3_1 的类型终判——它判的是段边界，逐行剪枝与它无关。

## 断点

- `3_1_keywords_pruning.json`：3_1 各批输出 + **最终注释组装 + 关键词类型内联
  后的字幕**（兼作整步的字幕断点）。`text` 在第 2 步收尾定案、本步不动；
  `text_llm` 在第 2 步收尾物化指代/黑话，本步末尾再叠加关键词类型内联，此后
  第 4~7 步逐字节不动；故本步不落任何字幕 txt，供人工查看的干净字幕是第 2 步
  的 `2_clean_subtitle.txt`；
- `3_2_topic_reset.json`：3_2 各单元输出 + **全场最终段列表**（下游步骤 4 的
  事件区间、5_1 的段索引都以它为准）。

两个检查点都存在即整步跳过。
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...llm_infra.batch import batch_units
from ...llm_infra.io import read_json, write_integrated, write_json_atomic
from ...llm_infra.config import get_batch_size
from ...llm_infra.resource import fanout
from ...log_config import step_logger
from ..subtitle_cleaning.common import build_row_annotations
from ..schemas import SubtitleRow, TopicSegment, TopicUnit
from ..utils.llm_text_utils import (
    ANNOTATION_KEY,
    build_topic_units,
    materialize_once,
    render_annotation_dict,
    render_windowed_subtitle,
)
from ..utils.rowid_scope import scope_id_set
from .refine_stages import RefineStages
from .topic_aggregator import aggregate_topics_by_type
from .topic_reset import TopicResetter

logger = logging.getLogger(__name__)


class RefineStep:
    """步骤 3 编排器：3_1 关键词与剪枝终判 ∥ 3_2 话题重置。"""

    def __init__(self) -> None:
        self._stages = RefineStages()
        self._resetter = TopicResetter(self._stages)

    def process(
        self,
        subtitle_l: list[SubtitleRow],
        segments: list[TopicSegment],
        step2_result: dict,
        base_dir: Path,
    ) -> tuple[list[SubtitleRow], list[TopicSegment]]:
        """第 3 步主入口。

        Args:
            subtitle_l: 第 2 步产出的字幕（已纠错、指代/黑话结构化在 annotation，
                `text_llm` 未物化，渲染期懒加载）
            segments: 1_1 + 1_2 合并后的话题段（含 `困惑` / `困惑原因`）
            step2_result: 第 2 步的结论集（`verdicts` / `referents` / `pruned` /
                `mods` / `confirmed_jargon`）
            base_dir: 分析目录（llm_analysis/）

        Returns:
            `(带最终注释的字幕, 3_2 修正后的话题段)`
        """
        with step_logger("第3步-剪枝终判与话题重置"):
            cached = self._load_checkpoints(base_dir, segments)
            if cached is not None:
                return cached

            units = build_topic_units(subtitle_l, segments)
            if not units:
                logger.warning("没有可处理的话题单元，3_1 跳过（3_2 仍照常判定）")

            batch_size = get_batch_size("3_1_关键词与剪枝")
            topic_batches = batch_units(units, batch_size, sort_key=lambda u: u.idx)
            logger.info(
                "第 3 步开始：3_1 分 %d 批（%d 话题，每批 ≤ %d）、"
                "3_2 按困惑段聚类——两者并行提交同一线程池",
                len(topic_batches), len(units), batch_size,
            )

            # ===== 3_1 与 3_2 并行提交 =====
            # 各自独立 fanout，先提交后收；两组任务在共享池里真正交错执行
            futures_3_1 = self._submit_3_1(
                topic_batches, subtitle_l, step2_result, base_dir,
            )
            reset_result: dict = {}

            def _run_3_2() -> None:
                segs, outputs = self._resetter.reset(
                    segments, subtitle_l,
                    step2_result.get("confirmed_jargon") or [],
                    base_dir,
                )
                reset_result["segments"] = segs
                reset_result["outputs"] = outputs

            futures_3_2 = fanout("3_2", [(_run_3_2, (), {})])

            merged_3_1 = self._collect_3_1(futures_3_1, topic_batches, step2_result)
            for f in futures_3_2:
                f.result()  # 3_2 内部已按单元容错，此处异常属编排级，直接上抛

            final_segments = reset_result.get("segments") or list(segments)

            # ===== 最终注释组装（类型终判 + 纠错 + 指代 + 关键词）=====
            processed = self._assemble(subtitle_l, step2_result, merged_3_1)

            # ===== 一次性物化 text_llm（纯 Python，零 LLM）=====
            # 文本至此稳定：指代/黑话在第 2 步末结构化写进 annotation，关键词类型
            # 由 3_1 产出，这里从纯 text 一次性物化「指代 + 黑话 + 类型 + 段落指代」。
            # 此后第 4~7 步渲染的 [DATA] 直接读 text_llm，逐字节稳定。
            processed = materialize_once(processed)

            # 落盘完整关键词列表（3_1 各批汇总，按 rowID 升序）：只读产物，
            # 供人工核对与程序化回读，管线自身不读它。与 `2_clean_subtitle.txt`
            # 同口径——只在第 3 步实际运行时写，命中检查点跳过时不重写。
            keywords = sorted(
                merged_3_1["关键词列表"],
                key=lambda kw: kw.get("rowID") or 0,
            )
            write_json_atomic(base_dir / "3_keywords.json", keywords)
            logger.info("关键词列表已落盘：3_keywords.json 共 %d 条", len(keywords))

            write_integrated(
                base_dir / "3_1_keywords_pruning.json",
                merged_3_1["outputs"],
                [sub.model_dump() for sub in processed],
            )
            write_integrated(
                base_dir / "3_2_topic_reset.json",
                reset_result.get("outputs") or {},
                [seg.model_dump() for seg in final_segments],
            )
            # 关键词原始输入：按类型分组聚合（段落事件由段标注注入、行号留空），
            # 落盘供第 4 步消费（第 4 步从它构建主题层级结构初始态）。
            grouped = aggregate_topics_by_type(processed, final_segments)
            write_json_atomic(base_dir / "3_aggregated_topics.json", grouped)
            # 本步不落字幕 txt：`text` 在第 2 步收尾定案、本步不动；`text_llm`
            # 在本步末尾一次性物化（指代 + 黑话 + 类型）。供人工查看的干净字幕
            # 由第 2 步落成 `2_clean_subtitle.txt`（动态内联文本）。
            return processed, final_segments

    # ==================== 断点 ====================

    def _load_checkpoints(
        self, base_dir: Path, segments: list[TopicSegment],
    ) -> tuple[list[SubtitleRow], list[TopicSegment]] | None:
        """三个检查点都有效时返回缓存结果，否则 None（整步重跑）。

        只有一个有效也整步重跑：3_1 的字幕、3_2 的段列表与聚合产物要一起进下游，
        半套缓存会让"注释按新字幕、段按旧分割"这类错配静默发生。
        """
        sub_path = base_dir / "3_1_keywords_pruning.json"
        seg_path = base_dir / "3_2_topic_reset.json"
        agg_path = base_dir / "3_aggregated_topics.json"
        if not (sub_path.exists() and seg_path.exists() and agg_path.exists()):
            return None
        sub_data = read_json(sub_path)
        seg_data = read_json(seg_path)
        if not (
            isinstance(sub_data, dict) and isinstance(sub_data.get("结果"), list)
            and sub_data["结果"]
            and isinstance(seg_data, dict) and isinstance(seg_data.get("结果"), list)
            and seg_data["结果"]
        ):
            logger.info("第 3 步检查点无效，重跑整步")
            return None
        logger.info(
            "检测到 3_1_keywords_pruning.json、3_2_topic_reset.json 与 "
            "3_aggregated_topics.json，跳过整个第 3 步（%d 行字幕、%d 个话题段）",
            len(sub_data["结果"]), len(seg_data["结果"]),
        )
        return (
            [SubtitleRow.model_validate(sub) for sub in sub_data["结果"]],
            [TopicSegment.model_validate(seg) for seg in seg_data["结果"]],
        )

    # ==================== 3_1 ====================

    def _submit_3_1(
        self,
        topic_batches: list[list[TopicUnit]],
        subtitle_l: list[SubtitleRow],
        step2_result: dict,
        base_dir: Path | None,
    ) -> list:
        """提交 3_1 各批（不等待）。

        `[DATA]` 按本批话题行 ± N_REDUNDANT_ROWS 切片，基于第 2 步产出的字幕
        重新开窗——关键词要在**纠错后且指代已内联**的字面上提（「赵毅」→「兆易」
        才可能映射到兆易创新）。
        """
        pruned = step2_result.get("pruned") or {"无关": set(), "噪音": set()}
        # 从字幕行提取「疑似候选」（2_3 后仍未解决、但带候选的疑似点），随
        # [ANNOTATION] 传给 3_1 参考——候选是疑似、未确证，只作提示不物化。
        uncertain_by_row: dict[int, dict] = {}
        for sub in subtitle_l:
            ann = sub.annotation
            if not isinstance(ann, dict):
                continue
            uncertain = ann.get("不确定")
            if not uncertain:
                continue
            cands = [
                u for u in uncertain
                if isinstance(u, dict) and (u.get("候选") or "").strip()
            ]
            rid = sub.rowID
            if cands and rid is not None:
                uncertain_by_row[int(rid)] = {"不确定": cands}
        annotation_text = render_annotation_dict({
            **{rid: {"类型": "无关"} for rid in sorted(pruned["无关"])},
            **{rid: {"类型": "噪音"} for rid in sorted(pruned["噪音"])},
            **uncertain_by_row,
        })
        specs: list[tuple] = []
        for i, batch in enumerate(topic_batches):
            specs.append((
                self._stages.run_3_1,
                (
                    batch,
                    render_windowed_subtitle(
                        subtitle_l, [rid for u in batch for rid in u.row_ids],
                    ),
                    annotation_text,
                    pruned,
                    base_dir,
                    i,
                ),
                {},
            ))
        return fanout("3_1", specs)

    def _collect_3_1(
        self,
        futures: list,
        topic_batches: list[list[TopicUnit]],
        step2_result: dict,
    ) -> dict:
        """收 3_1 各批结果；失败批沿用第 2 步剪枝初判、本批无关键词。"""
        pruned_2 = step2_result.get("pruned") or {"无关": set(), "噪音": set()}
        outputs: dict[str, dict] = {}
        keywords: list[dict] = []
        irrelevant: set[int] = set()
        noise: set[int] = set()
        reclass: list[dict] = []
        for i, (future, batch) in enumerate(zip(futures, topic_batches)):
            try:
                out = future.result()
            except Exception as e:  # noqa: BLE001 —— 单批失败不拖垮整步
                logger.warning(
                    "「3_1 关键词与剪枝 b%d」失败：%s——沿用第 2 步剪枝初判、本批无关键词",
                    i, e,
                )
                scope = scope_id_set([rid for u in batch for rid in u.row_ids])
                irrelevant |= pruned_2["无关"] & scope
                noise |= pruned_2["噪音"] & scope
                continue
            outputs[f"3_1_关键词与剪枝_b{i}"] = out
            keywords.extend(out["关键词列表"])
            irrelevant |= set(out["无关行号"])
            noise |= set(out["噪音行号"])
            reclass.extend(out["改判说明"])
            logger.info(
                "「3_1 关键词与剪枝 b%d」完成: %d 话题, 关键词 %d 条, "
                "终判 无关%d/噪音%d 行, 改判 %d 处",
                i, len(batch), len(out["关键词列表"]),
                len(out["无关行号"]), len(out["噪音行号"]), len(out["改判说明"]),
            )
        for r in reclass:
            logger.info(
                "「3_1 改判」row%s：%s → %s（%s）",
                r.get("rowID"), r.get("原判"), r.get("终判"), r.get("原因", "")[:60],
            )
        return {
            "outputs": outputs, "关键词列表": keywords,
            "无关": irrelevant, "噪音": noise, "改判说明": reclass,
        }

    # ==================== 注释组装 ====================

    def _assemble(
        self,
        subtitle_l: list[SubtitleRow],
        step2_result: dict,
        merged_3_1: dict,
    ) -> list[SubtitleRow]:
        """组装最终注释：3_1 终判的类型 + 第 2 步纠错/指代 + 3_1 关键词。

        本函数原在第 2 步末尾（`word_level_step._assemble`），随 2_4 → 3_1 迁来：
        类型终判与关键词都是 3_1 的产出，注释只能在第 3 步才组装完整。

        **组剪枝必须先播种**（`_group_pruned_types`）：1_1 判「不重要」的整段由
        `mark_unimportant_segments` 打成 `无关` 合成注释，而这些行**不建工作单元**
        （`build_topic_units` 跳过），故 3_1 根本不会为它们输出行号。若只用 3_1 的
        判定当 exceptions，这些行会在重建注释时回落成 `重点`——第 1 步的组剪枝
        就此静默失效，与 `mark_unimportant_segments` 承诺的"把剪枝传给所有下游阶段"
        相矛盾。加了 `[DATA]` 第三列后这个矛盾还会被放大：那些行会被明确标成
        `重点` 送进模型，等于主动邀请它去分析卖课带货段。
        """
        exceptions = _group_pruned_types(subtitle_l)
        for rid in merged_3_1["无关"]:
            exceptions[rid] = "无关"
        for rid in merged_3_1["噪音"]:
            exceptions[rid] = "噪音"

        rows_by_id = {
            int(sub.rowID): sub
            for sub in subtitle_l
            if sub.rowID is not None
        }
        # 关键词完全采用 3_1 的产出：指代与黑话已由第 2 步物化进字幕文本，
        # 3_1 线性扫描提取即可，无需任何代码侧兜底补入。
        keywords = merged_3_1["关键词列表"]
        annotations = build_row_annotations(
            rows_by_id,
            step2_result.get("mods") or [],
            exceptions,
            step2_result.get("referents") or [],
            keywords,
        )
        out: list[SubtitleRow] = []
        for sub in subtitle_l:
            rid = sub.rowID
            if rid is None:
                out.append(sub)
                continue
            new_ann = annotations.get(int(rid), {"类型": "重点"})
            # 保留第 2 步写进、build_row_annotations 重建时不带的键：
            # 「不确定」候选（含 2_3 疑似候选）与「段落指代」——前者供步骤 4 起
            # 第三列渲染，后者供最终物化时动态内联，都必须显式合并回。
            old_ann = sub.annotation
            if isinstance(old_ann, dict):
                if old_ann.get("不确定"):
                    new_ann = {**new_ann, "不确定": old_ann["不确定"]}
                if old_ann.get("段落指代"):
                    new_ann = {**new_ann, "段落指代": old_ann["段落指代"]}
            out.append(sub.model_copy(update={ANNOTATION_KEY: new_ann}))
        kept_rows = sum(1 for a in annotations.values() if a["类型"] == "重点")
        logger.info(
            "注释组装完成：重点 %d 行 / 无关 %d / 噪音 %d；关键词 %d 条，指代 %d 条",
            kept_rows, len(merged_3_1["无关"]), len(merged_3_1["噪音"]),
            len(keywords), len(step2_result.get("referents") or []),
        )
        return out


def _group_pruned_types(subtitle_l: list[SubtitleRow]) -> dict[int, str]:
    """收集入参字幕上已有的 无关/噪音 类型（第 1 步组剪枝的合成注释）。

    这些行不进 3_1 的工作单元，故必须从入参继承——见 `_assemble` 的说明。
    """
    out: dict[int, str] = {}
    for sub in subtitle_l:
        rid = sub.rowID
        ann = sub.annotation
        if rid is None or not isinstance(ann, dict):
            continue
        typ = ann.get("类型")
        if typ in ("无关", "噪音"):
            out[int(rid)] = typ
    if out:
        logger.info(
            "继承第 1 步组剪枝 %d 行（不重要话题段，不参与 3_1 判定）", len(out),
        )
    return out
