"""时长计算工具——步骤 5（主题分析）与步骤 6（引用校验）共用同一口径。

模型分级：主题时长 < 视频总时长/10 用 deepseek-v4-flash，否则用 deepseek-v4-pro。
两处必须导入同一实现，保证引用校验的分级与主题分析一致。
"""

import logging

from ...utils.time_utils import hms_to_seconds
from ..schemas import SubtitleRow

logger = logging.getLogger(__name__)


def calculate_video_total_duration(subtitle_l: list[SubtitleRow]) -> float:
    """视频总时长 = 最后一条字幕结束时间 - 第一条开始时间（秒）。"""
    if not subtitle_l:
        return 0.0
    starts: list[float] = []
    ends: list[float] = []
    for s in subtitle_l:
        if not isinstance(s, SubtitleRow):
            continue
        st = s.start_time or getattr(s, "开始时间", "")
        et = s.end_time or getattr(s, "结束时间", "")
        if st:
            try:
                starts.append(hms_to_seconds(st))
            except Exception:
                pass
        if et:
            try:
                ends.append(hms_to_seconds(et))
            except Exception:
                pass
    if not starts or not ends:
        return 0.0
    return max(0.0, max(ends) - min(starts))


def calculate_topic_duration(topic_data: dict) -> float:
    """主题顶层引用区间并集长度（秒）。"""
    if not isinstance(topic_data, dict):
        return 0.0
    intervals: list[tuple[float, float]] = []
    for key in ("引用", "引用时间"):
        refs = topic_data.get(key) or []
        if not isinstance(refs, list):
            continue
        for item in refs:
            if not isinstance(item, dict):
                continue
            start = item.get("开始时间", "")
            end = item.get("结束时间", "")
            if not (start and end):
                continue
            try:
                s = hms_to_seconds(start)
                e = hms_to_seconds(end)
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
