"""引用扩展模块（流水线第 4_10 小步：锚点 rowID → 行号区间）。

非销毁式树上的**递归段并集**扩展：
1. 每个主题把自身锚点扩展成段区间——锚点所在段即整段区间，不再做「标签匹配
   主导性检测」退化单点：伪主题（一笔带过的提及）的识别交由下游主题分析承担。
2. 根主题（不在任何主题「子项」列表中的节点）递归收集自身 + 所有子孙的锚点
   （事件父只收区间内锚点），作为其完整引用——用于控制第 7 步 DATA 范围；
   子主题只保留自身段区间（作为下游「个股清单 / 引用定位」提示）。

输入：keyword_data（非销毁式扁平树）+ segments。
输出：每个主题 `{类型, 引用: [{开始行号, 结束行号}], 子项}`。
"""

import logging
from collections.abc import Iterable
from typing import Any

from ...log_config import step_logger
from ..schemas import SubtitleRow, TopicData, TopicSegment
from .event_topics import resolve_event_ranges

logger = logging.getLogger(__name__)


class CitationExpander:
    """把每个主题的锚点 rowID 扩展成行号区间（根主题递归段并集）。"""

    def expand(
        self,
        keyword_data: dict[str, TopicData],
        segments: list[TopicSegment],
        subtitle_l: list[SubtitleRow] | None = None,
    ) -> dict[str, TopicData]:
        with step_logger("第4步-引用扩展"):
            if not keyword_data:
                return keyword_data
            if not segments:
                logger.warning("segments 为空，无法把锚点定位到话题段，回退到单点区间")
                for td in keyword_data.values():
                    td.引用 = _anchors_to_singletons(td.引用 or [])
                return keyword_data

            # 递归段并集扩展：根主题引用 = 自身锚点段 ∪ 递归所有子孙锚点段。
            return self._expand_by_recursive_segment_union(keyword_data, segments, subtitle_l)

    @staticmethod
    def _expand_by_recursive_segment_union(
        keyword_data: dict[str, TopicData],
        segments: list[TopicSegment],
        subtitle_l: list[SubtitleRow] | None,
    ) -> dict[str, TopicData]:
        """递归段并集扩展：根主题引用 = 自身锚点段 ∪ 递归所有子孙锚点段。

        阶段 1：每个主题把自身锚点扩展成段区间（子主题也扩展，作为下游提示；
        伪主题识别交由下游主题分析，本阶段不再做标签匹配主导性检测）。
        阶段 2：根主题递归收集自身 + 所有子孙的段区间并集（事件父只收完全落在
        事件区间内的段——区间外的独立讨论不得撑开事件引用）。
        """
        roots = _root_topics(keyword_data)
        event_ranges = resolve_event_ranges(keyword_data, segments)

        # 阶段 1：每个主题自身锚点 → 段区间
        for topic, td in keyword_data.items():
            anchors = _coerce_int_anchors(td.引用 or [])
            td.引用 = _ranges_for_anchors(anchors, segments)

        # 阶段 2：根主题递归段并集（事件父只收区间内段）
        for topic in roots:
            td = keyword_data.get(topic)
            collected = _collect_subtree_ranges(topic, keyword_data)
            if topic in event_ranges:
                collected = [
                    r for r in collected
                    if _range_in_ranges(r, event_ranges[topic])
                ]
            td.引用 = _merge_overlapping_ranges(collected)
            logger.info(
                "根主题 '%s' 递归段并集后区间数: %d（子项 %d）",
                topic, len(td.引用), len(td.子项 or []),
            )

        return keyword_data


def _root_topics(keyword_data: dict[str, TopicData]) -> set[str]:
    """返回不在任何主题「子项」列表中的主题名（树的根）。"""
    children: set[str] = set()
    for td in keyword_data.values():
        for c in td.子项 or []:
            if isinstance(c, str) and c:
                children.add(c)
    return {t for t in keyword_data if t not in children}


def _collect_subtree_ranges(
    topic: str,
    keyword_data: dict[str, TopicData],
    visited: set[str] | None = None,
) -> list[dict]:
    """递归收集 topic 自身 + 所有子孙的段区间（扁平 list[dict]）。

    子树节点的「引用」应已是扩展后的 `[{开始行号, 结束行号}]`。
    `visited` 防环（树理论无环，但残留环的节点各自成根，这里兜底）。
    """
    if visited is None:
        visited = set()
    if topic in visited:
        return []
    visited = visited | {topic}
    td = keyword_data.get(topic)
    if td is None:
        return []
    collected: list[dict] = list(td.引用 or [])
    for child in td.子项 or []:
        if isinstance(child, str) and child != topic:
            collected.extend(_collect_subtree_ranges(child, keyword_data, visited))
    return collected


def _range_in_ranges(r: dict, ranges: list[tuple[int, int]]) -> bool:
    """判断段区间 r 是否完全落在某个事件区间内。"""
    try:
        s = int(r["开始行号"])
        e = int(r["结束行号"])
    except (KeyError, TypeError, ValueError):
        return False
    return any(es <= s and e <= ee for es, ee in ranges)


def _coerce_int_anchors(anchors: Iterable[Any]) -> list[int]:
    out: list[int] = []
    for a in anchors:
        try:
            out.append(int(a))
        except (TypeError, ValueError):
            continue
    return out


def _anchors_to_singletons(anchors: Iterable[Any]) -> list[dict]:
    out: list[dict] = []
    for a in _coerce_int_anchors(anchors):
        out.append({"开始行号": a, "结束行号": a})
    return out


def _ranges_for_anchors(
    anchors: list[int],
    segments: list[TopicSegment],
) -> list[dict]:
    """把每个锚点 rowID 落到所在话题段，返回去重后的段边界列表。

    伪主题（一笔带过的提及）的识别交由下游主题分析，本阶段不再做主导性检测：
    锚点命中段即扩张为整段区间。

    若某锚点不落在任何段里（理论上不应发生——3_2 已校验全覆盖），fallback 到该锚点的单点区间。
    """
    if not anchors:
        return []

    seg_idx_set: set[int] = set()
    orphan_anchors: list[int] = []
    for a in anchors:
        found = False
        for idx, seg in enumerate(segments):
            if seg.开始行号 <= a <= seg.结束行号:
                seg_idx_set.add(idx)
                found = True
                break
        if not found:
            orphan_anchors.append(a)

    if orphan_anchors:
        logger.warning(
            "%d 个锚点未落到任何话题段：%s（fallback 到单点区间）",
            len(orphan_anchors), orphan_anchors[:5],
        )

    ranges: list[dict] = []
    for idx in sorted(seg_idx_set):
        ranges.append({
            "开始行号": int(segments[idx].开始行号),
            "结束行号": int(segments[idx].结束行号),
        })
    for a in orphan_anchors:
        ranges.append({"开始行号": a, "结束行号": a})
    return _merge_overlapping_ranges(ranges)


def _merge_overlapping_ranges(
    ranges: list[dict],
    merge_adjacent: bool = False,
) -> list[dict]:
    """合并重叠区间；`merge_adjacent=True` 时相邻区间（s == last_end + 1）也合并。

    段归属路径（ownership）传 True：相邻段同属一个主题时连成一片，减少引用时间
    区间的个数（rowID 连续，相邻即时间连续）。标签匹配降级路径保持 False——
    其相邻段之间的边界代表 1_2 判定为"不合并"的话题切换，禁止粘连。
    """
    cleaned: list[tuple[int, int]] = []
    for r in ranges:
        if not isinstance(r, dict):
            continue
        try:
            s = int(r.get("开始行号"))
            e = int(r.get("结束行号"))
        except (TypeError, ValueError):
            continue
        if s > e:
            s, e = e, s
        cleaned.append((s, e))
    if not cleaned:
        return []
    cleaned.sort(key=lambda x: x[0])
    merged_pairs: list[list[int]] = [list(cleaned[0])]
    threshold = 1 if merge_adjacent else 0
    for s, e in cleaned[1:]:
        last = merged_pairs[-1]
        if s <= last[1] + threshold:
            last[1] = max(last[1], e)
        else:
            merged_pairs.append([s, e])
    return [{"开始行号": s, "结束行号": e} for s, e in merged_pairs]
