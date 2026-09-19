"""词级三阶段（2_1-2_3）的单阶段执行器：渲染 → 调用 → 语义校验。

每个阶段一次调用处理**一批话题**，输出扁平（rowID 全局唯一，话题归属由代码用
行号范围反查）。本模块只负责"发一次调用并把结果校验干净"，批量分组/回退/跨阶段
串联由 `word_level_step.py` 编排。

原第四阶段「2_4 关键词与剪枝」已迁为**第 3 步的 3_1**（`step3_refine/`）：它与
「话题重置 3_2」并行，都属于"在已纠错字幕上重做判定"，与本步的词级纠错不同性质。

设计要点（CLAUDE.md 设计哲学）：
- 三阶段统一「漏斗式三字段」输出：2_1 五字段（含剪枝），2_2/2_3 三字段
  （疑似点 + 指代列表 + 错词列表，无剪枝）。疑似点逐层传递（筛剩+新增），
  错词/指代列表逐层累积（新确定）。
- 语义校验层层收口：片段必须是原文子串、错词不得涉及立场词翻转（多空/买卖/涨跌）、
  行号必须落在本批范围内。不合规的条目单独丢弃并 warning，不废掉整批。
- [DATA] 三阶段共用同一份按批切片的字幕，前缀缓存靠它命中。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ...llm_infra.shared_state import SharedFindings
from ...llm_infra.tools import ROW_PINYIN_TOOL, WEB_SEARCH_TOOL
from ..utils.llm_text_utils import load_prompt
from ..utils.rowid_scope import format_scope, scope_id_set
from ..utils.llm_text_utils import render_annotation_dict
from ..schemas import TopicUnit
from .common import HotwordResources
from .pinyin_matcher import (
    _find_suspect_candidates,
    _find_truncation_candidates,
    row_reading,
    scan_row_candidates,
)
from .vocab_hints import build_jargon_hints

logger = logging.getLogger(__name__)

_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent)),
    keep_trailing_newline=True,
)
# 本步骤的三个阶段（原 2_4 关键词与剪枝已迁为第 3 步的 3_1，见 step3_refine/）
_STAGES = ("2_1_粗筛", "2_2_词表判定", "2_3_联网确认")


# 立场/方向敏感词黑名单：承载主讲人看多/看空、买卖、涨跌判断。修正前后命中的
# 立场词集合不一致即视为高危翻转，错词列表一律丢弃该条（安全优先于"特别确定"）。
# 只收「双字及以上明确立场词」，不收单字——避免「很多」「空仓」等被单字误伤。
_OPINION_FLIP_WORDS: tuple[str, ...] = (
    # 多空
    "看多", "看空", "做多", "做空", "多头", "空头", "翻多", "翻空",
    # 买卖 / 仓位
    "买入", "卖出", "加仓", "减仓", "增持", "减持", "建仓", "清仓",
    "追高", "杀跌", "抄底", "逃顶", "止盈", "止损",
    # 涨跌
    "看涨", "看跌", "上涨", "下跌", "涨停", "跌停",
    # 加息降息 / 利好利空 / 突破跌破
    "加息", "降息", "利好", "利空", "利好出尽", "利空出尽",
    "突破", "跌破",
)


def _hits_opinion_blacklist(anchor: str, fixed: str) -> bool:
    """修正前后在立场词上的差异：涉及多空/买卖/涨跌等立场词变化一律降级。

    立场词承载主讲人的看多/看空判断，改错一个字会翻转报告结论（看多→看空），
    是不可逆的高危改写。与其用读音判据（易错杀真实误识），不如直接黑名单：
    修正前后命中的立场词集合不一致就丢弃该错词（安全优先于"特别确定"）。

    集合不一致覆盖三类高危修改：
    - 立场翻转：看多→看空、买入→卖出；
    - 引入立场词：看度→看多（纠错却引入了方向判断）；
    - 删除立场词：看多→看到（把主讲人的方向判断改没了）。
    普通纠错（闪离→闪迪、国司→国瓷）两集合均为空，直接放行。
    """
    before = {w for w in _OPINION_FLIP_WORDS if w in anchor}
    after = {w for w in _OPINION_FLIP_WORDS if w in fixed}
    return before != after


class WordLevelStages:
    """2_1-2_3 三个阶段的调用执行器（无状态，资源在构造时装配）。"""

    def __init__(
        self,
        resources: HotwordResources | None = None,
        findings: SharedFindings | None = None,
    ) -> None:
        self._res = resources or HotwordResources()
        self._findings = findings
        self._llm = LLMClient()
        self._prompts = {
            name: load_prompt(f"{name}.txt", __file__) for name in _STAGES
        }
        self._schemas = {name: _load_schema(name, __file__) for name in _STAGES}
        self._templates = {
            name: _TEMPLATE_ENV.get_template(f"{name}_user.jinja2") for name in _STAGES
        }

    # ==================== 阶段 2_1：粗筛 ====================

    def run_2_1(
        self,
        topics: list[TopicUnit],
        subtitle_text: str,
        base_dir: Path | None,
        batch_idx: int,
        correction_hints_text: str = "",
        log_suffix: str = "",
    ) -> dict:
        """2_1 粗筛：剪枝初判 + 疑似点 + 自推指代 + 错词。无 [DATABASE]（定位就是不给词表）。

        `[ANNOTATION]` 带确定性拼音候选（**不含正式名**——那是词表知识，给了会废掉
        2_1/2_2 的分工），让模型重点留意这些位置；反锚定规则见 prompt 与
        shared_system 的 `拼音候选` 节。

        Args:
            topics: `[{"idx", "label", "row_ids"}]`，本批话题
        Returns:
            经语义校验的 `{"无关行号", "噪音行号", "疑似点", "指代列表", "错词列表"}`
        """
        all_row_ids = [rid for t in topics for rid in t.row_ids]
        text_by_id = _text_by_row(subtitle_text)
        candidates = scan_row_candidates(
            text_by_id, self._res.pinyin_index, self._res.truncation_index,
            include_formal=False,
            jargon_aliases=frozenset(self._res.jargon_map),
        )
        annotation_text = render_annotation_dict(
            {}, pinyin_candidates=candidates,
        )
        user_prompt = self._templates["2_1_粗筛"].render(
            task_rules=self._prompts["2_1_粗筛"],
            correction_hints_text=correction_hints_text,
            topics=[
                {"idx": t.idx, "label": t.label or "", "事件": t.事件 or "",
                 "scope": format_scope(t.row_ids)}
                for t in topics
            ],
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
            rowid_scope=format_scope(all_row_ids),
        )
        log_dir = get_step_output_dir(base_dir, 2) if base_dir is not None else None
        result = self._llm.call(
            user_prompt=user_prompt,
            temperature=0.0,
            schema=self._schemas["2_1_粗筛"],
            log_dir=log_dir,
            log_name=f"2_1_粗筛_b{batch_idx}{log_suffix}",
            step_name="2_1_粗筛",
        )
        return self._validate_2_1(result, all_row_ids, subtitle_text)

    def _validate_2_1(self, result: dict, row_ids: list[int], subtitle_text: str) -> dict:
        """2_1 语义校验：行号范围、片段子串、线索词在 [DATA] 出现、依据不得为数据库。

        错词列表是 2_1 的「特别确定」确定性修改，这里校验并过立场词黑名单——
        涉及多空/买卖/涨跌翻转的错词一律丢弃（安全优先于"特别确定"）。
        """
        scope = scope_id_set(row_ids)
        text_by_id = _text_by_row(subtitle_text)

        irrelevant = _filter_row_ids(result.get("无关行号"), scope, "2_1 无关行号")
        noise = _filter_row_ids(result.get("噪音行号"), scope, "2_1 噪音行号")
        # 同一行同时出现在两数组 → 判为无关（更保守：无关不参与分析但也不算胡言乱语）
        both = irrelevant & noise
        if both:
            logger.warning("2_1 有 %d 行同时标为无关与噪音，判为无关：%s", len(both), sorted(both)[:5])
            noise -= both
        pruned = irrelevant | noise

        suspects: list[dict] = []
        for item in _normalize_suspects(result.get("疑似点") or [], text_by_id, "2_1"):
            rid = item.get("rowID")
            if rid is None or int(rid) not in scope:
                logger.warning("2_1 疑似点行号 %s 越界，丢弃", rid)
                continue
            frag = (item.get("原先的片段") or "").strip()
            row_text = text_by_id.get(int(rid), "")
            if not frag or frag not in row_text:
                logger.warning("2_1 疑似点片段「%s」不是 row%s 原文子串，丢弃", frag, rid)
                continue
            # 线索词必须在 [DATA] 逐字出现（供 2_2 做 hotspot 门控，禁止脑补）
            clues = [c for c in (item.get("上下文线索") or []) if c and c in subtitle_text]
            dropped = len(item.get("上下文线索") or []) - len(clues)
            if dropped:
                logger.warning("2_1 row%s 有 %d 个线索词未在字幕出现，已剔除", rid, dropped)
            entry = {
                "rowID": int(rid),
                "原先的片段": frag,
                "出现序号": _occurrence_index(item.get("出现序号")),
                "初判类型": item.get("初判类型") or "疑似错别字",
                "违和原因": item.get("违和原因") or "",
                "上下文线索": clues,
            }
            cand = (item.get("候选") or "").strip()
            if cand:
                entry["候选"] = cand
            suspects.append(entry)

        # 先校验错词、后校验指代：与 2_2/2_3 的 _validate_filter 同序。
        # 顺序不能反——「黑话错别字拆两列表」依赖 _normalize_referents 拿到本阶段
        # 错词表，把指代锚点从错字映射成改后词（如「老鹰」→「老英」），再用改后
        # 文本锚定；否则指代锚点仍是已消失的原字，物化时锚定失败报序号超界。
        wrong_words, blocked_wrong = _validate_wrong_words(
            _normalize_wrong_words(result.get("错词列表") or [], text_by_id, "2_1"),
            scope, text_by_id, pruned,
        )
        corrected_by_id = _apply_wrong_words(text_by_id, wrong_words)
        referents = _validate_referents(
            _normalize_referents(result.get("指代列表") or [], text_by_id, wrong_words, "2_1"),
            scope, corrected_by_id, pruned,
            allowed_bases={"上下文", ""}, stage="2_1",
        )
        paragraph_referents = _validate_paragraph_referents(
            result.get("段落指代"), scope,
        )
        slips = _normalize_terminals(
            result.get("嘴瓢") or [], text_by_id, "2_1", "slip",
        )
        events = _normalize_terminals(
            result.get("事件描述") or [], text_by_id, "2_1", "event",
        )
        return {
            "无关行号": sorted(irrelevant),
            "噪音行号": sorted(noise),
            "疑似点": suspects,
            "指代列表": referents,
            "错词列表": wrong_words,
            "段落指代": paragraph_referents,
            "嘴瓢": slips,
            "事件描述": events,
            "立场门禁拦截": blocked_wrong,
        }

    # ==================== 阶段 2_2：词表判定 ====================

    def run_2_2(
        self,
        suspects: list[dict],
        subtitle_text: str,
        annotation_text: str,
        base_dir: Path | None,
        batch_idx: int,
        correction_hints_text: str = "",
        log_suffix: str = "",
        subtitle_knowledge: list[dict] | None = None,
        topics: list[TopicUnit] | None = None,
        full_text_by_id: dict[int, str] | None = None,
    ) -> dict:
        """2_2 词表判定：带全套词表消歧 2_1 的疑似点，输出三字段（疑似点/指代列表/错词列表）。

        Args:
            suspects: 2_1 的疑似点（含 rowID / 原先的片段 / 初判类型 / 上下文线索）
            subtitle_knowledge: 1_1/1_2 抽出的「字幕内自解释知识」，按上下文过滤后
                渲染进 [DATABASE] 供消歧优先参考（词必须出现在本批 [DATA] 里）。
        """
        text_by_id = _text_by_row(subtitle_text)
        context_text = self._suspects_context(suspects, text_by_id, subtitle_text)
        jargon_hints = build_jargon_hints(self._res.jargon_map, context_text)
        confirmed_hints = self._confirmed_jargon_hints(text_by_id, context_text)
        # 字幕知识按上下文过滤：只保留词出现在本批 [DATA] 里的条目，避免无关知识撑大 prompt
        scoped_knowledge = [
            k for k in (subtitle_knowledge or [])
            if isinstance(k, dict)
            and (word := (k.get("词") or "").strip())
            and word in subtitle_text
        ]
        enriched = self._enrich_with_candidates(suspects, text_by_id, full_text_by_id)
        row_ids = sorted(text_by_id.keys())
        annotation_text = self._merge_pinyin_annotation(
            annotation_text, suspects, [], text_by_id,
        )

        user_prompt = self._templates["2_2_词表判定"].render(
            task_rules=self._prompts["2_2_词表判定"],
            correction_hints_text=correction_hints_text,
            topics=[
                {"idx": t.idx, "label": t.label or "", "事件": t.事件 or ""}
                for t in (topics or [])
            ],
            suspects=enriched,
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
            jargon_hints=jargon_hints,
            confirmed_hints=confirmed_hints,
            subtitle_knowledge=scoped_knowledge,
            plain_names=[],
            news_items=self._res.get_market_news(),
            rowid_scope=format_scope(row_ids),
        )
        log_dir = get_step_output_dir(base_dir, 2) if base_dir is not None else None
        result = self._llm.call(
            user_prompt=user_prompt,
            temperature=0.0,
            schema=self._schemas["2_2_词表判定"],
            log_dir=log_dir,
            log_name=f"2_2_词表判定_b{batch_idx}{log_suffix}",
            step_name="2_2_词表判定",
        )
        return self._validate_filter(
            result, suspects, subtitle_text,
            wrong_bases={"数据库", "拼音候选", "上下文"},
            referent_bases={"数据库", "上下文"},
            stage="2_2",
        )

    # ==================== 阶段 2_3：联网确认 ====================

    def run_2_3(
        self,
        pending: list[dict],
        subtitle_text: str,
        annotation_text: str,
        tool_executor: Any,
        base_dir: Path | None,
        batch_idx: int,
        max_rounds: int = 3,
        correction_hints_text: str = "",
        log_suffix: str = "",
        topics: list[TopicUnit] | None = None,
        full_text_by_id: dict[int, str] | None = None,
        publish_date: str = "",
    ) -> tuple[dict, list[dict]]:
        """2_3 联网确认：两轮搜索 + 一轮收尾（第二轮条件触发）消歧 2_2 筛剩的疑似点，输出三字段。

        Returns: `(校验后结果, trace)` —— trace 供代码提取模型实际用过的搜索词
        生成入库建议（比模型自报更可信）。
        """
        text_by_id = _text_by_row(subtitle_text)
        context_text = self._suspects_context(pending, text_by_id, subtitle_text)
        jargon_hints = build_jargon_hints(self._res.jargon_map, context_text)
        confirmed_hints = self._confirmed_jargon_hints(text_by_id, context_text)
        enriched = self._enrich_with_candidates(pending, text_by_id, full_text_by_id)
        row_ids = sorted(text_by_id.keys())

        user_prompt = self._templates["2_3_联网确认"].render(
            task_rules=self._prompts["2_3_联网确认"],
            correction_hints_text=correction_hints_text,
            topics=[
                {"idx": t.idx, "label": t.label or "", "事件": t.事件 or ""}
                for t in (topics or [])
            ],
            pending=enriched,
            max_rounds=max_rounds,
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
            jargon_hints=jargon_hints,
            confirmed_hints=confirmed_hints,
            publish_date=publish_date,
            rowid_scope=format_scope(row_ids),
        )
        log_dir = get_step_output_dir(base_dir, 2) if base_dir is not None else None
        result, trace = self._llm.call_conversation(
            user_prompt=user_prompt,
            tools=[WEB_SEARCH_TOOL, ROW_PINYIN_TOOL],
            tool_executor=tool_executor,
            schema=self._schemas["2_3_联网确认"],
            max_rounds=max_rounds,
            dedup_directions=True,
            max_tool_calls=len(pending) * 2,
            tool_call_limit_names={"web_search"},
            temperature=0.0,
            log_dir=log_dir,
            log_name=f"2_3_联网确认_b{batch_idx}{log_suffix}",
            step_name="2_3_联网确认",
        )
        validated = self._validate_filter(
            result or {}, pending, subtitle_text,
            wrong_bases={"联网", "上下文"},
            referent_bases={"联网", "上下文"},
            stage="2_3",
        )
        return validated, trace or []

    def _confirmed_jargon_hints(
        self, text_by_id: dict[int, str], context_text: str,
    ) -> list[dict]:
        """本场已确证黑话（其他 batch 已确认的别名→正式名）→ [DATABASE] 别名清单。

        复用 `build_jargon_hints` 的「别名出现 + 线索门控」口径，与词表黑话一致；
        `row_scope` 取本批 [DATA] 窗口行（含 ±15 上下文）。
        """
        if self._findings is None:
            return []
        confirmed_map = self._findings.confirmed_map(set(text_by_id.keys()))
        if not confirmed_map:
            return []
        return build_jargon_hints(confirmed_map, context_text)

    # ==================== 共享校验 ====================

    def _validate_filter(
        self,
        result: dict,
        expected_suspects: list[dict],
        subtitle_text: str,
        wrong_bases: set[str],
        referent_bases: set[str],
        stage: str,
    ) -> dict:
        """2_2/2_3 三字段校验：疑似点（筛剩+新增+漏项补位）+ 指代列表 + 错词列表。

        校验顺序：先校验错词列表（原文子串 + 立场门禁），把错词应用到字幕副本，
        再校验指代列表（改后字幕子串锚定，支持黑话错别字的「改后词」）。最后校验
        疑似点：输出项分「筛剩」（在输入清单内）与「新增」（在 [DATA] 内）两类都
        保留；输入清单里既没被解决、也没出现在输出疑似点的，保守补位回疑似点。
        """
        text_by_id = _text_by_row(subtitle_text)
        # 工作范围 = [DATA] 的所有行（新增疑似点/错词/指代可落在 [DATA] 内任意行，
        # 不局限于上一层疑似点的行号）
        scope = scope_id_set(text_by_id.keys())

        # 1. 错词列表：原文子串 + 立场门禁（先由「标注」还原「原先的词」）
        wrong_words, blocked_wrong = _validate_wrong_words(
            _normalize_wrong_words(result.get("错词列表") or [], text_by_id, stage),
            scope, text_by_id, set(),
            allowed_bases=wrong_bases, stage=stage,
        )
        # 2. 应用错词到字幕副本，供指代锚定「改后词」（黑话错别字）
        corrected_by_id = _apply_wrong_words(text_by_id, wrong_words)
        # 3. 指代列表：改后字幕子串锚定（span 由原文标注解析，黑话错别字映射改后词）
        referents = _validate_referents(
            _normalize_referents(
                result.get("指代列表") or [], text_by_id, wrong_words, stage,
            ),
            scope, corrected_by_id, set(),
            allowed_bases=referent_bases, stage=stage,
        )
        # 4. 疑似点：筛剩 + 新增 + 漏项补位（先由「标注」还原「原先的片段」）
        suspects = self._validate_filtered_suspects(
            _normalize_suspects(result.get("疑似点") or [], text_by_id, stage),
            expected_suspects, scope,
            text_by_id, subtitle_text, wrong_words, referents, stage,
        )
        paragraph_referents = _validate_paragraph_referents(
            result.get("段落指代"), scope,
        )
        slips = _normalize_terminals(
            result.get("嘴瓢") or [], text_by_id, stage, "slip",
        )
        events = _normalize_terminals(
            result.get("事件描述") or [], text_by_id, stage, "event",
        )
        return {
            "疑似点": suspects,
            "指代列表": referents,
            "错词列表": wrong_words,
            "段落指代": paragraph_referents,
            "嘴瓢": slips,
            "事件描述": events,
            "立场门禁拦截": blocked_wrong,
        }

    def _validate_filtered_suspects(
        self,
        output_suspects: list[dict],
        expected_suspects: list[dict],
        scope: set[int],
        text_by_id: dict[int, str],
        subtitle_text: str,
        wrong_words: list[dict],
        referents: list[dict],
        stage: str,
    ) -> list[dict]:
        """2_2/2_3 疑似点校验：筛剩 + 新增 + 漏项补位。

        - 输出疑似点逐条校验（行号范围、片段子串、线索词在 [DATA]），「筛剩」与
          「新增」两类都保留（新增 = 不在输入清单、但行号/片段在 [DATA] 内）。
        - 输入清单里，既没出现在输出疑似点、也没被错词/指代解决的，保守补位回疑似点。
        """
        by_key = {(s["rowID"], s["原先的片段"]): s for s in expected_suspects}
        kept: list[dict] = []
        seen: set[tuple[int, str]] = set()

        for item in output_suspects or []:
            rid = item.get("rowID")
            if rid is None or int(rid) not in scope:
                continue
            rid = int(rid)
            frag = (item.get("原先的片段") or "").strip()
            if not frag or frag not in text_by_id.get(rid, ""):
                logger.warning("%s 疑似点片段「%s」不是 row%d 原文子串，丢弃", stage, frag, rid)
                continue
            if (rid, frag) in seen:
                continue
            seen.add((rid, frag))
            clues = [c for c in (item.get("上下文线索") or []) if c and c in subtitle_text]
            entry = {
                "rowID": rid,
                "原先的片段": frag,
                "出现序号": _occurrence_index(item.get("出现序号")),
                "初判类型": item.get("初判类型") or "疑似错别字",
                "违和原因": item.get("违和原因") or "",
                "上下文线索": clues,
            }
            cand = (item.get("候选") or "").strip()
            if cand:
                entry["候选"] = cand
            kept.append(entry)

        # 被解决的 key：错词（原文锚定）+ 指代（原文锚定，即「原先的词」仍在原文里）
        solved = {(w["rowID"], w["原先的词"]) for w in wrong_words}
        for r in referents:
            if r["原先的词"] in text_by_id.get(r["rowID"], ""):
                solved.add((r["rowID"], r["原先的词"]))

        # 漏项补位：输入清单里既没输出、也没被解决的，保守保留（不静默丢弃）
        for key, s in by_key.items():
            if key in seen or key in solved:
                continue
            logger.warning("%s 疑似点 (row%d,「%s」) 未获归宿，保守保留", stage, key[0], key[1])
            entry = {
                "rowID": key[0],
                "原先的片段": key[1],
                "出现序号": _occurrence_index(s.get("出现序号")),
                "初判类型": s.get("初判类型") or "疑似错别字",
                "违和原因": s.get("违和原因") or "",
                "上下文线索": [c for c in (s.get("上下文线索") or []) if c and c in subtitle_text],
            }
            cand = (s.get("候选") or "").strip()
            if cand:
                entry["候选"] = cand
            kept.append(entry)
            seen.add(key)

        return kept

    def _suspects_context(
        self, suspects: list[dict], text_by_id: dict[int, str], subtitle_text: str,
    ) -> str:
        """疑似点所在行 + 其上下文线索拼成的文本，供词表门控判定用。

        门控要求"线索词在字幕里逐字出现"，这里给的是**疑似点周边**而非全场字幕——
        用全场会让任何别名都能找到线索词，门控就失效了。
        """
        parts: list[str] = []
        for s in suspects:
            parts.append(text_by_id.get(s["rowID"], ""))
            parts.extend(s.get("上下文线索") or [])
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _merge_pinyin_annotation(
        annotation_text: str,
        suspects: list[dict],
        referents: list[dict],
        text_by_id: dict[int, str],
    ) -> str:
        """把 `行读音` 并入已渲染的 `[ANNOTATION]` JSON（保留原有剪枝标记）。

        入参 `annotation_text` 是上游已渲染好的紧凑 JSON 字符串（2_1 的剪枝初判），
        不能直接丢弃重渲——那会让 2_2 看不到哪些行已被剪枝。

        只给**疑似点与待校正指代所在行**的整行读音，不给全场：全场读音体量约与
        [DATA] 同阶，而 2_2 只判这些行。给整行（而非仅片段）是为了让模型能重新
        切分词边界——2_1 圈出的片段边界本身可能就是错的。
        """
        rows = sorted(
            {s["rowID"] for s in suspects} | {r["rowID"] for r in referents},
        )
        readings = {
            rid: row_reading(text_by_id[rid])
            for rid in rows
            if text_by_id.get(rid)
        }
        if not readings:
            return annotation_text
        payload: dict = {}
        if annotation_text:
            try:
                loaded = json.loads(annotation_text)
                if isinstance(loaded, dict):
                    payload = loaded
            except json.JSONDecodeError:
                logger.warning("2_2 上游 [ANNOTATION] 非合法 JSON，改为只带行读音")
        payload["行读音"] = {str(rid): r for rid, r in readings.items()}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _enrich_with_candidates(
        self,
        suspects: list[dict],
        text_by_id: dict[int, str],
        full_text_by_id: dict[int, str] | None = None,
    ) -> list[dict]:
        """给每个疑似点附上确定性拼音候选（同音 / 吞音）与所在行原文及前后文。

        `前文`/`后文` 各取 rowID 前后 2 行文本。rowID 有缺号，须按排序后的相邻
        位置取（`rid±1` 会少给行）。让模型不离开 [TASK] 就能看到并列/指代/起因
        线索——如「海力士又崩了 → 老美光也崩了」这类并列结构直接呈现在清单里。

        `full_text_by_id` 非 None 时，额外给每个疑似点附「频次」：其片段自身及
        邻接切分候选词在全场字幕的出现次数（如「美光」2 vs「老美」29），供模型
        用「高频 vs 低频」判断粘连片段断句。候选词先收集去重、一次性统计，全场
        文本只遍历一轮。
        """
        ordered = sorted(text_by_id.keys())
        idx_by_id = {rid: i for i, rid in enumerate(ordered)}
        # 候选词频次：先收集去重再一次性统计，全场文本只遍历一轮
        freq_cache: dict[str, int] = {}
        if full_text_by_id:
            all_words: set[str] = set()
            for s in suspects:
                frag = (s.get("原先的片段") or "").strip()
                row_text = text_by_id.get(s.get("rowID"), "")
                all_words.update(_adjacent_words(frag, row_text))
            for word in all_words:
                freq_cache[word] = sum(
                    text.count(word) for text in full_text_by_id.values()
                )
        out: list[dict] = []
        for s in suspects:
            row_text = text_by_id.get(s["rowID"], "")
            i = idx_by_id.get(s["rowID"])
            before: list[str] = []
            after: list[str] = []
            if i is not None:
                before = [text_by_id[ordered[j]] for j in range(max(0, i - 2), i)]
                after = [text_by_id[ordered[j]] for j in range(i + 1, min(len(ordered), i + 3))]
            entry: dict = {
                **s,
                "原文": row_text,
                "前文": before,
                "后文": after,
                "homophone_candidates": _find_suspect_candidates(
                    row_text, self._res.pinyin_index, max_per_line=5, max_score=0,
                ),
                "truncation_candidates": _find_truncation_candidates(
                    row_text, self._res.truncation_index,
                ),
            }
            if full_text_by_id:
                frag = (s.get("原先的片段") or "").strip()
                words = _adjacent_words(frag, row_text)
                entry["频次"] = {w: freq_cache.get(w, 0) for w in words}
            out.append(entry)
        return out


# ==================== 模块级辅助 ====================


def _adjacent_words(frag: str, row_text: str) -> list[str]:
    """生成疑似点片段的邻接候选切分词（片段自身 + 左邻组合 + 右邻组合）。

    粘连片段「老美光」里 frag「美光」的切分候选：
    - 片段自身「美光」；
    - 左切分「老美」（前邻字 + 片段首字，对应「老美 | 光」）；
    - 右切分「光也」（片段尾字 + 后邻字）。
    用于全场频次统计，让模型用「高频 vs 低频」判断该片段是否应独立成词。
    """
    if not frag or frag not in row_text:
        return [frag] if frag else []
    start = row_text.index(frag)
    end = start + len(frag)
    words = [frag]
    if start > 0:
        words.append(row_text[start - 1] + frag[0])
    if end < len(row_text):
        words.append(frag[-1] + row_text[end])
    return words


def _text_by_row(subtitle_text: str) -> dict[int, str]:
    """从渲染后的 [DATA] 文本反解 {rowID: 行文本}（制表符分隔，首列为 rowID）。

    取**第二列**而非 `parts[-1]`：`[DATA]` 从步骤 4 起是三列
    （`rowID\\t文本\\t行号标识`），`parts[-1]` 在那种形态下会取到 `重点`/`无关`，
    使所有"片段是否为原文子串"的校验全数失败。第 2 步自身的 [DATA] 仍是两列，
    但按第二列解析对两种形态都正确，不留这个坑给后来者。
    """
    out: dict[int, str] = {}
    for line in subtitle_text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            out[int(parts[0])] = parts[1]
        except ValueError:
            continue
    return out


def _filter_row_ids(raw: Any, scope: set[int], label: str) -> set[int]:
    """行号数组过滤：非整数/越界丢弃并 warning。"""
    kept: set[int] = set()
    dropped = 0
    for rid in raw or []:
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if rid_int in scope:
            kept.add(rid_int)
        else:
            dropped += 1
    if dropped:
        logger.warning("%s 有 %d 个行号越界或非法，已丢弃", label, dropped)
    return kept


# ==================== XML 标注解析（span 原地标记 → 内部形态） ====================
#
# 第 2 步三个 span 型列表（错词/指代/疑似点）的 LLM 输出改为「标注」字段：模型在
# 原文行里用 <err>/<ref>/<suspect> 标签原地圈出目标 span，代码解析标签提取 span、
# 由 offset 反推出现序号，还原成既有内部形态（原先的词/原先的片段/出现序号），
# 复用下游校验与物化。标签只圈 span，消歧结果（修改的词/指代/依据等）仍留 JSON 字段。
_TAG_RE = re.compile(r"<(err|ref|suspect|slip|event)(?:\s[^>]*)?>(.*?)</\1>", re.DOTALL)


def _occurrence_at(text: str, word: str, pos: int) -> int | None:
    """由 offset 反推 word 在 text 中的非重叠出现序号（从 1 起）；对不上返回 None。

    与 `count_occurrences` / `build_inlined_text` 同口径：自左向右、每次匹配后
    跳过该词。`pos` 必须恰好落在某次出现的起始位置。
    """
    occ = 1
    start = 0
    while True:
        idx = text.find(word, start)
        if idx < 0:
            return None
        if idx == pos:
            return occ
        occ += 1
        start = idx + len(word)


def _parse_tagged_span(
    tagged: str,
    expected_tag: str,
    row_text: str,
    stage: str,
) -> tuple[str, int, int] | None:
    """解析单个「标注」字段，返回 `(span, offset, occurrence)`；非法返回 None。

    校验顺序：恰好一个预期标签 → span 非空 → 去标签后与 row_text 逐字节相等
    （防模型改写原文）→ offset 能对上原文中第 N 次非重叠出现（`_occurrence_at`
    反推 occurrence）。任一不满足即丢弃并 warning（沿用既有「不废整批」风格）。
    """
    if not isinstance(tagged, str) or not tagged:
        return None
    matches = list(_TAG_RE.finditer(tagged))
    if len(matches) != 1:
        logger.warning(
            "%s 标注「%s」标签数应为 1，实为 %d，丢弃", stage, tagged[:60], len(matches),
        )
        return None
    m = matches[0]
    if m.group(1) != expected_tag:
        logger.warning(
            "%s 标注「%s」标签应为 <%s>，实为 <%s>，丢弃",
            stage, tagged[:60], expected_tag, m.group(1),
        )
        return None
    span = m.group(2)
    if not span:
        logger.warning("%s 标注「%s」标签内 span 为空，丢弃", stage, tagged[:60])
        return None
    # 去标签重建原文：标签整体替换为 span 内容，必须逐字节等于该行原文
    reconstructed = tagged[:m.start()] + span + tagged[m.end():]
    if reconstructed != row_text:
        logger.warning(
            "%s 标注去标签后与原文不一致（原文「%s」≠重建「%s」），丢弃",
            stage, row_text[:40], reconstructed[:40],
        )
        return None
    offset = m.start()
    occurrence = _occurrence_at(row_text, span, offset)
    if occurrence is None:
        logger.warning(
            "%s 标注 span「%s」在原文 offset %d 处对不上非重叠出现，丢弃",
            stage, span, offset,
        )
        return None
    return span, offset, occurrence


def _normalize_suspects(
    items: list[dict],
    text_by_id: dict[int, str],
    stage: str,
) -> list[dict]:
    """把疑似点的「标注」还原为内部形态（`原先的片段` + `出现序号`）。

    解析失败 / 行号不在本批的条目原样透传，由后续 `_validate_*` 统一丢弃，这里
    不做范围裁决。
    """
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        rid = item.get("rowID")
        tagged = item.get("标注")
        if rid is None or not isinstance(tagged, str):
            out.append(item)
            continue
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            out.append(item)
            continue
        row_text = text_by_id.get(rid_int)
        if row_text is None:
            out.append(item)
            continue
        parsed = _parse_tagged_span(tagged, "suspect", row_text, stage)
        if parsed is None:
            out.append(item)
            continue
        span, _offset, occ = parsed
        out.append({**item, "原先的片段": span, "出现序号": occ})
    return out


def _normalize_terminals(
    items: list[dict],
    text_by_id: dict[int, str],
    stage: str,
    expected_tag: str,
) -> list[dict]:
    """把嘴瓢（<slip>）/事件描述（<event>）列表的「标注」还原为内部形态。

    与疑似点同机制：解析标签圈出的 span，反推出现序号；`违和原因` 仍取自 JSON
    字段。非法条目丢弃并 warning（沿用「不废整批」风格）。
    """
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rid = item.get("rowID")
        tagged = item.get("标注")
        if rid is None or not isinstance(tagged, str):
            continue
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            continue
        row_text = text_by_id.get(rid_int)
        if row_text is None:
            continue
        parsed = _parse_tagged_span(tagged, expected_tag, row_text, stage)
        if parsed is None:
            continue
        span, _offset, occ = parsed
        out.append({
            "rowID": rid_int,
            "原先的片段": span,
            "出现序号": occ,
            "违和原因": (item.get("违和原因") or "").strip(),
        })
    return out


def _normalize_wrong_words(
    items: list[dict],
    text_by_id: dict[int, str],
    stage: str,
) -> list[dict]:
    """把错词列表的「标注」还原为内部形态（`原先的词` + `出现序号`）。

    标签内容即错字本身（原文子串），`修改的词`/`依据` 仍取自 JSON 字段。
    """
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        rid = item.get("rowID")
        tagged = item.get("标注")
        if rid is None or not isinstance(tagged, str):
            out.append(item)
            continue
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            out.append(item)
            continue
        row_text = text_by_id.get(rid_int)
        if row_text is None:
            out.append(item)
            continue
        parsed = _parse_tagged_span(tagged, "err", row_text, stage)
        if parsed is None:
            out.append(item)
            continue
        span, _offset, occ = parsed
        out.append({**item, "原先的词": span, "出现序号": occ})
    return out


def _normalize_referents(
    items: list[dict],
    text_by_id: dict[int, str],
    wrong_words: list[dict] | None,
    stage: str,
) -> list[dict]:
    """把指代列表的「标注」还原为内部形态（`原先的词` + `出现序号`）。

    `wrong_words` 非 None 时，按 `(rowID, span, 出现序号)` 查已校验通过的错词表：
    命中（黑话错别字「拆两列表」）则把 `原先的词` 替换为该错词的 `修改的词`，
    与「先纠错后内联」语义一致——`[DATA]` 里只有原文错字，模型只能圈错字，正确
    写法要代码侧回填。
    """
    lookup: dict[tuple[int, str, int], str] = {}
    if wrong_words:
        for w in wrong_words:
            lookup[(w["rowID"], w["原先的词"], w.get("出现序号", 1))] = w["修改的词"]
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        rid = item.get("rowID")
        tagged = item.get("标注")
        if rid is None or not isinstance(tagged, str):
            out.append(item)
            continue
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            out.append(item)
            continue
        row_text = text_by_id.get(rid_int)
        if row_text is None:
            out.append(item)
            continue
        parsed = _parse_tagged_span(tagged, "ref", row_text, stage)
        if parsed is None:
            out.append(item)
            continue
        span, _offset, occ = parsed
        mapped = lookup.get((rid_int, span, occ), span)
        out.append({**item, "原先的词": mapped, "出现序号": occ})
    return out


def _occurrence_index(raw: Any) -> int:
    """出现序号安全归一：非法/缺失/小于 1 一律归 1（自左向右从 1 起）。"""
    try:
        occ = int(raw)
    except (TypeError, ValueError):
        return 1
    return occ if occ >= 1 else 1


def _normalize_referent_targets(raw: Any) -> list[str]:
    """把指代列表的「指代」字段归一化成非空实体名字列表（一对多）。

    接受 LLM 输出的字符串数组，或历史/误判的单字符串（拆成单元素）。
    每个元素 strip 后非空才保留，去重保序。
    """
    if isinstance(raw, str):
        candidates: list = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if not isinstance(c, str):
            continue
        s = c.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _validate_paragraph_referents(items: Any, scope: set[int]) -> list[dict]:
    """段落指代校验：rowID 数组非空且行号落在 scope、真实含义非空、理由非空。

    按排序后的 rowID 数组去重（同组行只保留一条）。
    """
    kept: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        rows_raw = item.get("rowID")
        if not isinstance(rows_raw, list) or not rows_raw:
            continue
        rows: list[int] = []
        valid = True
        for r in rows_raw:
            try:
                rid = int(r)
            except (TypeError, ValueError):
                valid = False
                break
            if rid not in scope:
                valid = False
                break
            if rid not in rows:
                rows.append(rid)
        if not valid or not rows:
            continue
        meaning = (item.get("真实含义") or "").strip()
        reason = (item.get("理由") or "").strip()
        if not meaning or not reason:
            continue
        key = tuple(sorted(rows))
        if key in seen:
            continue
        seen.add(key)
        kept.append({"rowID": rows, "真实含义": meaning, "理由": reason})
    return kept


def _validate_wrong_words(
    items: list[dict],
    scope: set[int],
    text_by_id: dict[int, str],
    pruned: set[int],
    allowed_bases: set[str] | None = None,
    stage: str = "2_1",
) -> tuple[list[dict], list[dict]]:
    """错词列表校验：行号范围 → 非剪枝行 → 词为该行子串 → 立场词黑名单 → 去重。

    返回 `(kept, blocked)`：`kept` 是正常通过的错词（改字面）；`blocked` 是被
    `_hits_opinion_blacklist` 立场门禁拦下的错词（含 `候选`=修改的词），**不改字面**，
    由调用方降级为「不确定+候选」写进 annotation——让「看涨期→看长期」这类正确纠错
    线索不因安全优先而静默丢失。`allowed_bases` 按层参数化（2_1 上下文/空，
    2_2 数据库/拼音候选/上下文，2_3 联网/上下文）。
    """
    if allowed_bases is None:
        allowed_bases = {"上下文", ""}
    kept: list[dict] = []
    blocked: list[dict] = []
    seen: set[tuple[int, str, int]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        rid = item.get("rowID")
        if rid is None or int(rid) not in scope:
            continue
        rid = int(rid)
        if rid in pruned:
            continue  # 剪枝行不参与语义标注
        word = (item.get("原先的词") or "").strip()
        if not word or word not in text_by_id.get(rid, ""):
            logger.warning("%s 错词「%s」不是 row%d 原文子串，丢弃", stage, word, rid)
            continue
        fixed = (item.get("修改的词") or "").strip()
        if not fixed or fixed == word:
            logger.warning("%s 错词「%s」→「%s」修改无效，丢弃", stage, word, fixed)
            continue
        # 立场词黑名单：涉及多空/买卖/涨跌翻转的错词修改不写回字面，降级为
        # 「不确定+候选」——安全优先于"特别确定"，但不静默丢弃正确纠错线索。
        if _hits_opinion_blacklist(word, fixed):
            logger.warning(
                "%s row%d 错词「%s」→「%s」涉及立场词变化，降级为不确定+候选", stage, rid, word, fixed,
            )
            blocked.append({"rowID": rid, "原先的词": word, "候选": fixed})
            continue
        occ = _occurrence_index(item.get("出现序号"))
        key = (rid, word, occ)
        if key in seen:
            continue
        seen.add(key)
        entry: dict = {
            "rowID": rid, "原先的词": word, "修改的词": fixed, "出现序号": occ,
        }
        basis = item.get("依据")
        if basis in allowed_bases and basis:
            entry["依据"] = basis
        kept.append(entry)
    return kept, blocked


def _validate_referents(
    items: list[dict],
    scope: set[int],
    text_by_id: dict[int, str],
    pruned: set[int],
    allowed_bases: set[str],
    stage: str,
) -> list[dict]:
    """指代校验：行号范围 → 非剪枝行 → 词为该行子串 → 指代必填 → 去重。"""
    kept: list[dict] = []
    seen: set[tuple[int, str, int]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        rid = item.get("rowID")
        if rid is None or int(rid) not in scope:
            continue
        rid = int(rid)
        if rid in pruned:
            continue  # 剪枝行不参与语义标注
        word = (item.get("原先的词") or "").strip()
        if not word or word not in text_by_id.get(rid, ""):
            logger.warning("%s 指代词「%s」不是 row%d 行子串，丢弃", stage, word, rid)
            continue
        target = _normalize_referent_targets(item.get("指代"))
        if not target:
            continue  # 指代必填（三层指代列表都不设困惑）
        occ = _occurrence_index(item.get("出现序号"))
        key = (rid, word, occ)
        if key in seen:
            continue
        seen.add(key)
        entry: dict = {
            "rowID": rid, "原先的词": word, "指代": target, "出现序号": occ,
        }
        basis = item.get("依据")
        if basis in allowed_bases and basis:
            entry["依据"] = basis
        kept.append(entry)
    return kept


def _apply_wrong_words(
    text_by_id: dict[int, str], wrong_words: list[dict],
) -> dict[int, str]:
    """把错词列表应用到 {rowID: 文本} 副本，返回改后文本（供指代锚定「改后词」）。"""
    corrected = dict(text_by_id)
    by_row: dict[int, list[dict]] = {}
    for w in wrong_words:
        by_row.setdefault(w["rowID"], []).append(w)
    for rid, mods in by_row.items():
        text = corrected.get(rid, "")
        for m in mods:
            text = text.replace(m["原先的词"], m["修改的词"])
        corrected[rid] = text
    return corrected
