"""第 3 步末尾的主题聚合器：把逐行注释关键词聚合为按类型分组的数组。

分组数组是关键词的「原始输入」——第 4 步从它构建主题层级结构（`{名称:{类型,引用}}`），
后续 4_1~4_10 再产出「子项」父关系树。本模块同时提供从分组数组还原层级结构初始态的
`build_keyword_data_from_grouped`，供第 4 步（含无落盘兜底路径）复用。

聚合是纯 Python，无 LLM 调用、无词表依赖，等价于原「`AnnotationAggregator` + 事件
一等化」两段逻辑，但产出从 `{正式名:{类型,引用}}` 改为按类型分组数组。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from ..utils.keyword_utils import select_topic_type
from ..utils.llm_text_utils import ANNOTATION_KEY, row_is_analyzable
from ..schemas import SubtitleRow, TopicSegment, TopicData

logger = logging.getLogger(__name__)

# 分组 key 顺序：段落事件恒在首位（一等主题），其余按关键词类型枚举顺序。
TOPIC_GROUP_TYPES: tuple[str, ...] = (
    "段落事件", "关键词事件",
    "股票", "板块", "指数", "风格因子", "债券", "大宗商品", "外汇", "加密货币",
    "个股黑话", "板块黑话",
)


def aggregate_topics_by_type(
    subtitle_l: list[SubtitleRow],
    segments: list[TopicSegment],
) -> dict[str, list[dict]]:
    """把逐行注释关键词按类型聚合为分组数组（关键词原始输入）。

    输出 `{类型: [{"名称", "行号", "话题段"}]}`，`TOPIC_GROUP_TYPES` 里所有 key 均
    出现（空类型为空数组）。普通关键词「行号」= 该正式名出现的 rowID 升序，「话题段」
    = 这些行所在段编号（1-based）去重升序。段落事件由话题段的「事件」字段注入：
    「行号」留空、「话题段」= 事件名出现的段编号集合；段事件名与某关键词正式名同名时
    升格（该关键词从普通分组移入段落事件分组）。
    """
    # 1) rowID -> 段号（1-based）
    row_to_segment: dict[int, int] = {}
    for idx, seg in enumerate(segments or [], start=1):
        start = int(seg.开始行号)
        end = int(seg.结束行号)
        for rid in range(start, end + 1):
            row_to_segment[rid] = idx

    # 2) 逐行注释关键词按正式名聚合（类型仲裁 + 行号收集）
    buckets: dict[str, dict] = {}
    for sub in subtitle_l:
        if not row_is_analyzable(sub):
            continue
        ann = sub.annotation
        if not isinstance(ann, dict):
            continue
        rid = sub.rowID
        if rid is None:
            continue
        try:
            rid = int(rid)
        except (TypeError, ValueError):
            continue
        for entry in ann.get("关键词") or []:
            if not isinstance(entry, dict):
                continue
            formal = (entry.get("正式名") or "").strip()
            if not formal:
                continue
            bucket = buckets.setdefault(formal, {
                "_行号_set": set(),
                "_类型_set": set(),
            })
            typ = entry.get("类型")
            if isinstance(typ, str) and typ:
                bucket["_类型_set"].add(typ)
            bucket["_行号_set"].add(rid)

    # 3) 事件一等化：段标注事件名 -> 段号集合（只收「重要」段）
    event_segments: dict[str, set[int]] = {}
    for idx, seg in enumerate(segments or [], start=1):
        if seg.是否重要 is False:
            continue
        event_name = str(seg.事件 or "").strip()
        if not event_name:
            continue
        event_segments.setdefault(event_name, set()).add(idx)

    # 段事件名与关键词正式名同名 -> 升格：从普通分组移出（行号清空、只留话题段）
    for event_name in event_segments:
        buckets.pop(event_name, None)

    # 4) 组装分组数组
    grouped: dict[str, list[dict]] = {t: [] for t in TOPIC_GROUP_TYPES}
    for name in sorted(event_segments):
        grouped["段落事件"].append({
            "名称": name,
            "行号": [],
            "话题段": sorted(event_segments[name]),
        })
    for name in sorted(buckets):
        bucket = buckets[name]
        typ = select_topic_type(bucket["_类型_set"])
        rows = sorted(bucket["_行号_set"])
        segs = sorted({row_to_segment[r] for r in rows if r in row_to_segment})
        grouped.setdefault(typ, []).append({
            "名称": name,
            "行号": rows,
            "话题段": segs,
        })

    _log_grouped(grouped)
    return grouped


def build_keyword_data_from_grouped(
    grouped: dict[str, list[dict]],
    segments: list[TopicSegment],
) -> dict[str, TopicData]:
    """从分组数组构建主题层级结构初始态 `{名称: TopicData}`。

    普通主题「引用」= 行号数组；段落事件「引用」= 由其「话题段」各段取开始行号
    （作为锚点，保证通过第 4 步「过滤空引用主题」且 4_10 可扩展）。
    """
    seg_start: dict[int, int] = {}
    for idx, seg in enumerate(segments or [], start=1):
        seg_start[idx] = int(seg.开始行号)

    keyword_data: dict[str, TopicData] = {}
    for typ, items in grouped.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            name = (item.get("名称") or "").strip()
            if not name:
                continue
            if typ == "段落事件":
                refs = sorted({
                    seg_start[n] for n in _coerce_ints(item.get("话题段"))
                    if n in seg_start
                })
            else:
                refs = _coerce_ints(item.get("行号"))
            keyword_data[name] = TopicData(类型=typ, 引用=refs)
    return keyword_data


def _coerce_ints(values: Iterable[Any]) -> list[int]:
    """把可迭代项转 int 列表（跳过非法项），去重升序。"""
    out: set[int] = set()
    for v in values or []:
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _log_grouped(grouped: dict[str, list[dict]]) -> None:
    """聚合完成状态日志：各类型数量。"""
    counts = {t: len(grouped.get(t) or []) for t in TOPIC_GROUP_TYPES}
    non_empty = [f"{t} {c}" for t, c in counts.items() if c]
    total = sum(counts.values())
    logger.info(
        "注释聚合完成：共 %d 个主题%s",
        total, f"（{'，'.join(non_empty)}）" if non_empty else "",
    )
