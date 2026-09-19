"""事件主题的共享规则（步骤 4 内部）。

事件分两类（第 3 步聚合阶段定案），以两个独立 `类型` 值区分：

- **段落事件**：`1_1`/`1_2` 在话题段上标注了完整事件弧的事件，
  报告的一等主题，**只能作父、不能作子**（恒为顶层）。
- **关键词事件**：3_1 关键词提取标成事件、但无段级事件弧
  的事件提及，**只能作子**（随板块/段落事件收编）。

本模块集中这两个类型的判据，供候选生成（4_2/4_4）、候选剪枝（4_8）共用：

- `is_segment_event` / `is_keyword_event`：类型判定的唯一口径。
- `collect_event_ranges`：从话题段还原每个段落事件名的行号区间（可跨多段，相邻段合并）。
- `anchor_ratio_in_ranges`：child 锚点落在某区间内的占比——4_8 候选剪枝「共现率
  最高者当选父」的打分基础。事件与板块候选父一视同仁按共现率竞争，倒挂场景下
  大板块与事件的共现率天然很低，不会当选，故不再需要「事件父优先 + 容器保护」。
"""

import logging
from collections.abc import Iterable
from typing import Any

from ..schemas import TopicData, TopicSegment

logger = logging.getLogger(__name__)


def is_segment_event(topic_data: Any) -> bool:
    """是否为「段落事件」：1_1/1_2 标注的完整事件弧，一等主题，只能作父。"""
    return (
        isinstance(topic_data, TopicData)
        and topic_data.类型 == "段落事件"
    )


def is_keyword_event(topic_data: Any) -> bool:
    """是否为「关键词事件」：无段级事件弧的关键词级事件提及，只能作子。"""
    return (
        isinstance(topic_data, TopicData)
        and topic_data.类型 == "关键词事件"
    )


# 未锁定黑话的类型口径：`个股黑话` / `板块黑话`。第 3 步起关键词不再带 `困惑`
# 字段，解析不出的黑话直接用这两个类型标记（`正式名` 即词本身，不可采信）。
_UNRESOLVED_JARGON_TYPES = frozenset({"个股黑话", "板块黑话"})


def is_unresolved_jargon(topic_data: Any) -> bool:
    """主题是否为「未锁定黑话」：类型为个股/板块黑话。

    这类主题是消解的半成品——模型怀疑它是黑话、却锁定不了背后的实体（`正式名`
    即词本身、不可采信）。它不该像正常叶子主题那样参与候选父挂靠：否则会按
    「同段共现」被误挂到某个板块（实测「啤酒」被 4_4 挂进「半导体」），污染真板块。
    """
    return (
        isinstance(topic_data, TopicData)
        and topic_data.类型 in _UNRESOLVED_JARGON_TYPES
    )


def event_topic_names(keyword_data: dict[str, TopicData]) -> set[str]:
    """keyword_data 里全部段落事件主题名（一等事件主题）。"""
    return {t for t, td in keyword_data.items() if is_segment_event(td)}


def collect_event_ranges(
    segments: list[TopicSegment] | None,
) -> dict[str, list[tuple[int, int]]]:
    """从话题段还原 `{事件名: [(开始行号, 结束行号), ...]}`（相邻/重叠区间合并）。

    只收**重要**段：不重要段（感谢/闲聊/卖课）已被剪枝，不构成事件区间。
    同一事件名的多个段（事件弧的后续角度）各自成区间，相邻的连成一片。
    """
    raw: dict[str, list[tuple[int, int]]] = {}
    for seg in segments or []:
        if seg.是否重要 is False:
            continue
        name = str(seg.事件 or "").strip()
        if not name:
            continue
        start = int(seg.开始行号)
        end = int(seg.结束行号)
        if start > end:
            start, end = end, start
        raw.setdefault(name, []).append((start, end))

    return {name: _merge_ranges(ranges) for name, ranges in raw.items()}


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠与相邻（end + 1 == next_start）区间。"""
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged: list[list[int]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def resolve_event_ranges(
    keyword_data: dict[str, TopicData],
    segments: list[TopicSegment] | None,
) -> dict[str, list[tuple[int, int]]]:
    """按 keyword_data 里的事件主题解析区间，`{事件主题名: [(开始, 结束), ...]}`。

    两条路径（并集）：

    1. **按名字**：段的 `事件` 字段 == 主题名（注入路径的常态）。
    2. **按锚点兜底**：4_1 可能把两个事件合并、正式名取的是另一个写法，此时段里
       找不到该名字。于是回溯：**带事件名的段**里，凡开始行号落在该事件主题锚点集合
       中的，其区间也算这个事件的。限定「带事件名的段」是刻意的——事件主题经去重
       可能并入过普通概念的锚点，若不限定，那些锚点会把无关段拉进事件区间。
    """
    by_name = collect_event_ranges(segments)
    resolved: dict[str, list[tuple[int, int]]] = {}
    unresolved: list[str] = []
    for topic, td in keyword_data.items():
        if not is_segment_event(td):
            continue
        ranges = list(by_name.get(topic) or [])
        if not ranges:
            anchors = set(coerce_anchors(td.引用 or []))
            for seg in segments or []:
                if seg.是否重要 is False:
                    continue
                if not str(seg.事件 or "").strip():
                    continue
                start = int(seg.开始行号)
                end = int(seg.结束行号)
                if start in anchors:
                    ranges.append((start, end) if start <= end else (end, start))
            if ranges:
                logger.info(
                    "事件主题「%s」按名字取不到区间，按锚点回溯到 %d 段", topic, len(ranges),
                )
        if ranges:
            resolved[topic] = _merge_ranges(ranges)
        else:
            unresolved.append(topic)
    if unresolved:
        logger.info(
            "共 %d 个事件主题无段级事件弧（关键词级事件提及），不参与事件父共现率竞争：%s",
            len(unresolved), unresolved,
        )
    return resolved


def row_in_ranges(row_id: int, ranges: list[tuple[int, int]]) -> bool:
    """行号是否落在任一区间内。"""
    return any(start <= row_id <= end for start, end in ranges)


def coerce_anchors(anchors: Iterable[Any]) -> list[int]:
    """锚点列表转 int（跳过非法项）。本步骤内锚点恒为 rowID 整数。"""
    out: list[int] = []
    for a in anchors or []:
        try:
            out.append(int(a))
        except (TypeError, ValueError):
            continue
    return out


def anchor_ratio_in_ranges(
    anchors: Iterable[Any],
    ranges: list[tuple[int, int]],
) -> tuple[int, int]:
    """返回 (落在区间内的锚点数, 锚点总数)。总数为 0 时返回 (0, 0)。"""
    rows = coerce_anchors(anchors)
    if not rows:
        return 0, 0
    inside = sum(1 for r in rows if row_in_ranges(r, ranges))
    return inside, len(rows)
