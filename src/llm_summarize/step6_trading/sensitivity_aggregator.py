"""主讲人暗示重要话题聚合 - 不调 LLM，纯数据聚合。

源：sensitivity_detector（6_2）挂在 keyword_data 各主题下的 `敏感提示`。
本聚合器按 `锚定行号` 去重，把同一信号的多个源主题汇总到 `板块` 字段，产出
`主讲人暗示重要话题` 顶层固定章节。不依赖主题 analysis（排序依据从 keyword_data
独立算引用时长），供步骤 6 内联调用。
"""

import logging

from ...utils.time_utils import hms_to_seconds
from ..schemas import SubtitleRow, TopicData
from ..utils.row_id_2_time_range import convert_subtitle_l2rawID2subtitle_m

logger = logging.getLogger(__name__)

_FIXED_SECTION_NAMES = {"宏观分析", "交易技巧", "精炼总结", "主讲人暗示重要话题"}


class SensitivityAggregator:
    """敏感提示聚合器。"""

    @staticmethod
    def aggregate(
        keyword_data: dict[str, TopicData], subtitle_l: list[SubtitleRow],
    ) -> dict:
        rowID2subtitle_m = (
            convert_subtitle_l2rawID2subtitle_m(subtitle_l) if subtitle_l else {}
        )

        topic_durations: dict[str, float] = {}
        for topic, td in keyword_data.items():
            if topic in _FIXED_SECTION_NAMES:
                continue
            if isinstance(td, TopicData):
                topic_durations[topic] = SensitivityAggregator._topic_duration(td)

        aggregated: dict[int, dict] = {}
        for topic, topic_data in keyword_data.items():
            if topic in _FIXED_SECTION_NAMES:
                continue
            if not isinstance(topic_data, TopicData):
                continue
            hints = getattr(topic_data, "敏感提示", None)
            if not hints or not isinstance(hints, list):
                continue
            for hint in hints:
                if not isinstance(hint, dict):
                    continue
                anchor = hint.get("锚定行号")
                try:
                    anchor_key = int(anchor)
                except (TypeError, ValueError):
                    anchor_key = id(hint)
                entry = aggregated.get(anchor_key)
                if entry is None:
                    citation = SensitivityAggregator._build_citation(
                        hint.get("命中行号") or [],
                        rowID2subtitle_m,
                        (hint.get("原话") or "").strip(),
                    )
                    entry = {
                        "信号类型": (hint.get("信号类型") or "敏感提示").strip(),
                        "板块": [],
                        "原话": (hint.get("原话") or "").strip(),
                        "原话总结": (hint.get("原话总结") or "").strip(),
                        "锚定行号": anchor_key if isinstance(anchor_key, int) else None,
                        "命中行号": hint.get("命中行号") or [],
                        "LLM理由": hint.get("LLM理由", ""),
                        "引用": [citation] if citation else [],
                    }
                    aggregated[anchor_key] = entry
                if topic not in entry["板块"]:
                    entry["板块"].append(topic)

        if not aggregated:
            return {}

        records: list[dict] = []
        for entry in aggregated.values():
            entry["板块"].sort(key=lambda t: topic_durations.get(t, 0), reverse=True)
            records.append(entry)

        def _record_sort_key(rec: dict) -> tuple:
            top_dur = max(
                (topic_durations.get(t, 0) for t in rec.get("板块", [])),
                default=0,
            )
            anchor = rec.get("锚定行号") or 0
            return (-top_dur, anchor)

        records.sort(key=_record_sort_key)
        return {"主讲人暗示重要话题": records}

    @staticmethod
    def _topic_duration(topic_data: TopicData) -> float:
        """从 keyword_data 主题的 引用 字段算引用总时长（合并重叠区间）。

        与 TopicAnalyzer.add_citation_duration 同口径：引用字段里的 开始时间/结束时间
        转秒后合并重叠区间累加。第 4 步聚类已把行号转成时间，故无需字幕。
        """
        refs = topic_data.引用 or []
        if not isinstance(refs, list):
            return 0.0
        intervals: list[tuple[float, float]] = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            start_time = ref.get("开始时间", "") or ""
            end_time = ref.get("结束时间", "") or ""
            if not (start_time and end_time):
                continue
            try:
                s = hms_to_seconds(start_time)
                e = hms_to_seconds(end_time)
            except Exception:
                continue
            if e > s:
                intervals.append((s, e))
        if not intervals:
            return 0.0
        intervals.sort(key=lambda x: x[0])
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return float(sum(e - s for s, e in merged))

    @staticmethod
    def _build_citation(
        hit_rowids: list, rowID2subtitle_m: dict, fallback_text: str,
    ) -> dict | None:
        """根据 hit_rowids 在 rowID 映射里找首尾时间，组装一条引用字典。"""
        if not hit_rowids or not rowID2subtitle_m:
            return None
        valid: list[int] = []
        for r in hit_rowids:
            try:
                rid = int(r)
            except (TypeError, ValueError):
                continue
            if rid in rowID2subtitle_m:
                valid.append(rid)
        if not valid:
            return None
        valid.sort()
        first = rowID2subtitle_m[valid[0]]
        last = rowID2subtitle_m[valid[-1]]
        start_time = first.start_time or getattr(first, "开始时间", "")
        end_time = last.end_time or getattr(last, "结束时间", "")
        original_parts: list[str] = []
        for rid in valid:
            sub = rowID2subtitle_m.get(rid)
            if sub is None:
                continue
            text = (sub.text or getattr(sub, "文本", "") or "").strip()
            if text:
                original_parts.append(text)
        original_text = "".join(original_parts) or fallback_text
        return {
            "开始时间": start_time,
            "结束时间": end_time,
            "原文": original_text,
        }
