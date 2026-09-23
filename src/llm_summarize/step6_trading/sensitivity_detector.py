"""敏感信号检测模块（步骤 6 的 6_2）— 正则召回 + LLM 语义判定 双层方案

第一层（正则）：广撒网召回所有可能的敏感信号候选。模式覆盖 ASR 同音字变体
（私域 → 私欲/思域/死域 等；会员课 → 会员可/惠员课 等）。允许误识别，因为有第二层兜底。

第二层（LLM）：把候选清单（`[DATABASE]`）连同完整字幕（`[DATA]`）交给 deepseek-v4-flash
做语义判定，模型按锚定行号自己去字幕里读上下文。只有 `是否真信号=true` 且
`置信度 ∈ {high, medium}` 的候选才会被锚定到主题。
LLM 还可以修正信号类型（例如把误标为"下次再讲"的反讽语判为"非真信号"）。

调用时机：步骤 6 内部（6_2），交易技巧（6_1）之后、聚合与 6_3 校验之前。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template

from ...log_config import step_logger
from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ..schemas import SubtitleRow, TopicData
from ..utils.llm_text_utils import (
    collect_non_analysis_row_ids,
    load_prompt,
    render_annotations,
    render_windowed_subtitle,
    window_row_ids,
)
from ..utils.rowid_scope import format_scope

logger = logging.getLogger(__name__)

# 锚定窗口：信号 rowID 向前/后扩展 N 行字幕，用来匹配相关主题
_ANCHOR_LOOKBACK = 10
_ANCHOR_LOOKAHEAD = 5
# 原话拼接窗口：信号 rowID 前后取 N 行字幕作为展示用的"原话"
_QUOTE_LOOKBACK = 8
_QUOTE_LOOKAHEAD = 3
_QUOTE_MAX_CHARS = 240
# 信号聚类间距：相邻 rowID 差 ≤ 该值的多个命中视为同一段表述，只保留首个
_CLUSTER_GAP = 15
# 接受的 LLM 置信度：low 一律丢弃
_ACCEPT_CONFIDENCES = {"high", "medium"}


# ============== ASR 同音字变体 ==============
# 私域 → ASR 误识别变体
_SI_YU = r"(私域|私欲|思域|死域|丝域|四域|斯域)"
_SI_YU_LI = rf"{_SI_YU}(里|里边|里面)?"
# 公域 → ASR 误识别变体
_GONG_YU = r"(公域|攻域|共域|工域|公裕)"
# 会员课 → ASR 误识别变体（会员是高频词，会员课/会员群/会员里组合时才作信号）
_HUI_YUAN_KE = r"(会员课|会员可|惠员课|慧员课|会员裤|会员群|会员里(?:边|面)?)"


# ============== 信号模式表（粗筛，宽松召回）==============
# 排序原则：强信号 → 弱信号；命中即停止。LLM 会在第二层做语义复核。
_SENSITIVITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            rf"(下次|下回|下节|下一次|下一节|下一回|下一期|下期).{{0,8}}{_HUI_YUAN_KE}"
        ),
        "下次会员课预告",
    ),
    (
        re.compile(
            r"(下次|下回|下节|下一次|下一节|下一回|下一期|下期).{0,8}"
            r"(再讲|再聊|再说|讲一讲|讲一下|讲讲|聊一聊|聊一下|聊聊|"
            r"展开|拆解|分享|更新|补充|讲|聊|分析)"
        ),
        "下次再讲",
    ),
    (
        re.compile(r"留到下(次|回|节|期|一次|一节)"),
        "下次再讲",
    ),
    (
        re.compile(rf"{_SI_YU_LI}.{{0,6}}(讲|说|聊|分享|讨论|更新|展开)"),
        "私域可讲",
    ),
    (
        re.compile(rf"{_HUI_YUAN_KE}.{{0,6}}(讲|说|聊|分享|讨论|更新|展开)"),
        "私域可讲",
    ),
    (
        re.compile(
            rf"{_GONG_YU}(里|里边|这|这儿|这里|这边|这块|里面)?(讲|说|聊)"
            r"(不了|不行|不能|不方便|不便)"
        ),
        "公域讲不了",
    ),
    (
        re.compile(
            r"(具体|这个|那个).{0,12}(公司|股票|标的|名字).{0,12}"
            r"(讲不了|说不了|不能讲|不能说|不便讲|不便说|不方便讲|不方便说)"
        ),
        "具体标的不便明说",
    ),
    (
        re.compile(
            r"(在这儿|在这里|这儿|这里)"
            r"(讲不了|说不了|不能讲|不能说|不方便讲|不方便说)"
        ),
        "公域讲不了",
    ),
    (
        re.compile(r"咱们(讲不了|说不了|不能讲|不能说)"),
        "公域讲不了",
    ),
    (
        re.compile(r"这(能说|能讲|敢说|敢讲)吗"),
        "暗示敏感",
    ),
    (
        re.compile(r"懂的都懂|你懂的|大家都懂"),
        "暗示敏感",
    ),
    (
        re.compile(r"不便(明说|多说|细说)"),
        "暗示敏感",
    ),
]


# ============== LLM 判定单例（懒加载）==============

class _Verifier:
    """LLM 判定 client/prompt/schema/模板 单例。无信号时不会触发任何 IO。"""

    _client: LLMClient | None = None
    _system: str | None = None
    _schema: dict | None = None
    _template = None

    @classmethod
    def get(cls) -> tuple[LLMClient, str, dict | None]:
        if cls._client is None:
            cls._client = LLMClient()
            cls._system = load_prompt("6_2_敏感信号判定.txt", __file__)
            cls._schema = _load_schema("6_2_敏感信号判定", __file__)
        return cls._client, cls._system, cls._schema

    @classmethod
    def get_template(cls) -> Template:
        if cls._template is None:
            _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
            _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
            cls._template = _jinja_env.get_template("6_2_敏感信号判定_user.jinja2")
        return cls._template


# ============== 主入口 ==============

def detect_and_annotate(
    keyword_data: dict[str, TopicData],
    subtitle_l: list[SubtitleRow],
    base_dir: Path | None = None,
    result_sink: dict[str, dict] | None = None,
) -> dict[str, TopicData]:
    """扫描字幕识别敏感信号，把命中的信号挂到相关主题的 `敏感提示` 字段。

    流程：
        1. 正则召回所有候选（宽松，含 ASR 同音字变体）。
        2. 同段聚类去重（间距 ≤ _CLUSTER_GAP 的合成一条）。
        3. LLM 单次批量语义判定，丢弃伪信号 / low 置信度。
        4. 把真信号锚定到 keyword_data 里的主题，挂上 `敏感提示`。

    原地修改 keyword_data 并返回。`result_sink` 收集 6_2 判定调用的
    schema 校验后 result dict（正则无召回 / 调用失败时不写键）。
    """
    with step_logger("第6_2步-敏感信号判定"):
        if not keyword_data or not subtitle_l:
            return keyword_data

        # 无关/噪音行不作候选锚点（正则召回只在可分析行上跑）
        blocked = collect_non_analysis_row_ids(subtitle_l)

        rowid_to_text: dict[int, str] = {}
        for line in subtitle_l:
            if not isinstance(line, SubtitleRow):
                continue
            try:
                rid = int(line.rowID)
            except (TypeError, ValueError):
                continue
            if rid in blocked:
                continue
            rowid_to_text[rid] = (line.text or "").strip()

        if not rowid_to_text:
            return keyword_data

        sorted_rowids = sorted(rowid_to_text)

        # ===== 第一层：正则召回 =====
        signals: list[dict] = []
        for rid in sorted_rowids:
            text = rowid_to_text[rid]
            if not text:
                continue
            for pattern, label in _SENSITIVITY_PATTERNS:
                if pattern.search(text):
                    signals.append({
                        "rowID": rid,
                        "信号类型": label,
                        "原句": text,
                    })
                    break

        if not signals:
            logger.info("正则未召回候选")
            return keyword_data

        signals = _cluster_signals(signals)
        logger.info(
            "正则召回 %d 条候选（聚类后）：%s",
            len(signals),
            [(s["rowID"], s["信号类型"]) for s in signals],
        )

        # ===== 第二层：LLM 语义判定 =====
        # [DATA] 按候选锚定行 ± N_REDUNDANT_ROWS 切片：判定「这句是不是真信号」
        # 只需要锚定句周边语境；[ANNOTATION] 切到同一行集
        anchor_ids = [s["rowID"] for s in signals]
        signals = _verify_signals_with_llm(
            signals,
            render_windowed_subtitle(subtitle_l, anchor_ids, with_type=True),
            render_annotations(
                subtitle_l, row_scope=window_row_ids(subtitle_l, anchor_ids),
            ),
            base_dir, result_sink,
        )
        if not signals:
            logger.info("LLM 判定后无真信号")
            return keyword_data

        logger.info(
            "LLM 判定通过 %d 条：%s",
            len(signals),
            [(s["rowID"], s["信号类型"]) for s in signals],
        )

        # ===== 锚定到主题 =====
        annotated_topics: set[str] = set()
        for signal in signals:
            rid = signal["rowID"]
            anchored_topics = _find_anchored_topics(keyword_data, rid)
            if not anchored_topics:
                logger.info("rowID=%d 未锚定到任何主题，丢弃此信号", rid)
                continue

            quote = _build_quote(rowid_to_text, sorted_rowids, rid)
            hit_rowids = _collect_hit_rowids(sorted_rowids, rid)

            for topic in anchored_topics:
                td = keyword_data.get(topic)
                if not isinstance(td, TopicData):
                    continue
                hint_list = list(getattr(td, "敏感提示", []) or [])
                hint_list.append({
                    "原话": quote,
                    "原话总结": signal.get("原话总结", ""),
                    "信号类型": signal["信号类型"],
                    "命中行号": hit_rowids,
                    "锚定行号": rid,
                    "LLM理由": signal.get("LLM理由", ""),
                })
                td.敏感提示 = hint_list
                annotated_topics.add(topic)

        if annotated_topics:
            logger.info("已标记主题：%s", sorted(annotated_topics))
        return keyword_data


# ============== 第二层：LLM 判定 ==============

def _verify_signals_with_llm(
    signals: list[dict],
    full_subtitle_text: str,
    annotation_text: str,
    base_dir: Path | None,
    result_sink: dict[str, dict] | None = None,
) -> list[dict]:
    """单次批量 LLM 调用判定所有候选；返回真信号子集，并把 LLM 修正后的信号类型回写。"""
    if not signals:
        return signals

    # 上下文不再逐条拼进 payload——`[DATA]` 已按锚定行开窗（含 ±冗余上下文），
    # 模型自己按锚定行号在其中定位。
    payload: list[dict] = [
        {
            "候选id": idx,
            "信号类型_正则": sig["信号类型"],
            "锚定行号": sig["rowID"],
            "锚定句": sig["原句"],
        }
        for idx, sig in enumerate(signals)
    ]

    client, system, schema = _Verifier.get()
    user_prompt = _Verifier.get_template().render(
        task_rules=system,
        payload=payload,
        subtitle_text=full_subtitle_text,
        annotation_text=annotation_text,
        rowid_scope=format_scope([s["rowID"] for s in signals]),
    )

    log_dir = None
    if base_dir is not None:
        log_dir = get_step_output_dir(base_dir, 6)

    try:
        result = client.call(
            user_prompt=user_prompt,
            temperature=0.0,
            schema=schema,
            log_dir=log_dir,
            log_name="6_2_敏感信号判定",
            step_name="6_2_敏感信号判定",
        )
    except Exception as e:
        logger.warning("LLM 判定调用失败，保留所有正则候选作 fallback：%s", e)
        return signals

    if result_sink is not None:
        result_sink["6_2_敏感信号判定"] = result

    verdicts = result.get("判定", []) if isinstance(result, dict) else []
    keep: list[dict] = []
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        idx = v.get("候选id")
        if not isinstance(idx, int) or idx < 0 or idx >= len(signals):
            continue
        if not v.get("是否真信号"):
            continue
        if v.get("置信度", "low") not in _ACCEPT_CONFIDENCES:
            continue
        new_type = (v.get("信号类型") or "").strip()
        if new_type == "非真信号" or not new_type:
            continue
        sig = dict(signals[idx])
        sig["信号类型"] = new_type  # LLM 修正后的类型
        sig["LLM理由"] = (v.get("理由") or "").strip()
        sig["原话总结"] = (v.get("原话总结") or "").strip()
        keep.append(sig)
    return keep


# ============== 锚定 + 原话工具 ==============

def _find_anchored_topics(
    keyword_data: dict[str, TopicData], signal_rowid: int,
) -> list[str]:
    """找出 引用 数组覆盖到 [signal_rowid - LOOKBACK, signal_rowid + LOOKAHEAD] 的主题。"""
    win_lo = signal_rowid - _ANCHOR_LOOKBACK
    win_hi = signal_rowid + _ANCHOR_LOOKAHEAD

    matched: list[tuple[str, int]] = []
    for topic, td in keyword_data.items():
        if not isinstance(td, TopicData):
            continue
        refs = td.引用 or []
        if not isinstance(refs, list):
            continue
        best_distance: int | None = None
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            start = _safe_int(ref.get("开始行号"))
            end = _safe_int(ref.get("结束行号"))
            if start is None or end is None:
                continue
            if start > end:
                start, end = end, start
            if end < win_lo or start > win_hi:
                continue
            distance = min(abs(signal_rowid - start), abs(signal_rowid - end))
            if best_distance is None or distance < best_distance:
                best_distance = distance
        if best_distance is not None:
            matched.append((topic, best_distance))

    matched.sort(key=lambda x: x[1])
    return [t for t, _ in matched]


def _build_quote(
    rowid_to_text: dict[int, str],
    sorted_rowids: list[int],
    signal_rowid: int,
) -> str:
    """信号 rowID 前后窗口内字幕拼成的展示用原话。"""
    lo = signal_rowid - _QUOTE_LOOKBACK
    hi = signal_rowid + _QUOTE_LOOKAHEAD
    parts: list[str] = []
    for rid in sorted_rowids:
        if rid < lo:
            continue
        if rid > hi:
            break
        text = rowid_to_text.get(rid, "").strip()
        if text:
            parts.append(text)
    joined = "".join(parts)
    if len(joined) > _QUOTE_MAX_CHARS:
        joined = joined[:_QUOTE_MAX_CHARS] + "…"
    return joined


def _collect_hit_rowids(sorted_rowids: list[int], signal_rowid: int) -> list[int]:
    """信号 rowID 周围实际存在于字幕里的 rowID 列表。"""
    lo = signal_rowid - _QUOTE_LOOKBACK
    hi = signal_rowid + _QUOTE_LOOKAHEAD
    return [rid for rid in sorted_rowids if lo <= rid <= hi]


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cluster_signals(signals: list[dict]) -> list[dict]:
    """rowID 间距 ≤ _CLUSTER_GAP 的连续命中合成一个簇，保留簇中首条。"""
    if not signals:
        return signals
    signals_sorted = sorted(signals, key=lambda s: s["rowID"])
    clusters: list[dict] = [signals_sorted[0]]
    for sig in signals_sorted[1:]:
        if sig["rowID"] - clusters[-1]["rowID"] <= _CLUSTER_GAP:
            continue
        clusters.append(sig)
    return clusters
