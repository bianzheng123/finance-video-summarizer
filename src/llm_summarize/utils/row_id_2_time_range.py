from typing import Any

from ..schemas import SubtitleRow

# 合并相邻引用区间的最大缺号行数：两个区间之间缺的行数 ≤ 该值即并成一段。
# 例：106 / 108-111 / 113（中间只缺 107、112 两行语气词）→ 合并为 106-113 一段。
MAX_CITATION_MERGE_GAP = 2


def merge_row_ids_to_ranges(
    row_ids: list[Any], max_gap: int = MAX_CITATION_MERGE_GAP,
) -> list[dict]:
    """把离散的 rowID 整数列表合并为区间列表，缺号行数 ≤ max_gap 的相邻行并入同一区间。

    输入：[3, 4, 5, 8, 10, 11]
    输出：[{"开始行号": 3, "结束行号": 5}, {"开始行号": 8, "结束行号": 8}, {"开始行号": 10, "结束行号": 11}]
    输入：[106, 108, 109, 110, 111, 113]（中间缺 107、112）
    输出：[{"开始行号": 106, "结束行号": 113}]
    """
    if not row_ids:
        return []
    sorted_ids = sorted(int(r) for r in row_ids if r is not None)
    if not sorted_ids:
        return []
    ranges = []
    start = sorted_ids[0]
    end = sorted_ids[0]
    for r in sorted_ids[1:]:
        if r - end - 1 <= max_gap:
            end = r
        else:
            ranges.append({"开始行号": start, "结束行号": end})
            start = r
            end = r
    ranges.append({"开始行号": start, "结束行号": end})
    return ranges


def merge_adjacent_citation_ranges(
    ranges: list[dict], max_gap: int = MAX_CITATION_MERGE_GAP,
) -> list[dict]:
    """合并相邻的 {开始行号, 结束行号} 区间：间隙缺号行数 ≤ max_gap 的两段并成一段。

    用于引用校验的重写路径：LLM 输出的碎片区间（106-106 / 108-111 / 113-113，
    中间隔 107、112 两行语气词）合并为 106-113。非 dict / 缺字段条目跳过。
    """
    spans: list[tuple[int, int]] = []
    for item in ranges:
        if not isinstance(item, dict):
            continue
        try:
            s = int(item["开始行号"])
            e = int(item["结束行号"])
        except (KeyError, TypeError, ValueError):
            continue
        if s > e:
            s, e = e, s
        spans.append((s, e))
    if not spans:
        return []
    spans.sort()
    merged: list[dict] = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s - cur_e - 1 <= max_gap:
            cur_e = max(cur_e, e)
        else:
            merged.append({"开始行号": cur_s, "结束行号": cur_e})
            cur_s, cur_e = s, e
    merged.append({"开始行号": cur_s, "结束行号": cur_e})
    return merged


def convert_row_range_to_time(
    citation: dict, line_to_subtitle: dict[int, SubtitleRow],
    max_gap: int = MAX_CITATION_MERGE_GAP,
) -> list[dict] | None:
    """将包含开始行号/结束行号的单条引用转换为时间范围。

    处理逻辑：
    1. 若开始或结束行号不在 line_to_subtitle 中，双向收缩到最近存在的有效行号：
       - 开始行号收缩：找 >= 原开始行号的最小有效行号
       - 结束行号收缩：找 <= 原结束行号的最大有效行号
    2. 若收缩后起始 > 结束，返回 None
    3. 遍历收缩后的区间，按连续存在的 rowID 拆分为多个子区间；
       缺号连续 run ≤ max_gap 不拆分（保持整段引用），> max_gap 才断开
    4. 每个子区间生成一个引用字典副本，包含更新后的行号和对应的时间范围
    5. 若没有任何有效子区间，返回 None

    Args:
        citation: 引用字典，包含"开始行号"和"结束行号"
        line_to_subtitle: 行号(int)到字幕条目的映射

    Returns:
        转换后的引用字典列表，每个包含"开始时间"和"结束时间"。
        若完全无效，返回 None。
    """
    assert isinstance(citation, dict)
    assert "开始行号" in citation and "结束行号" in citation

    start_line = int(citation["开始行号"])
    end_line = int(citation["结束行号"])

    if start_line > end_line:
        return None

    if not line_to_subtitle:
        return None

    valid_lines = sorted(line_to_subtitle.keys())

    # 双向收缩：找最近存在的有效边界
    new_start = None
    for line in valid_lines:
        if line >= start_line:
            new_start = line
            break

    new_end = None
    for line in reversed(valid_lines):
        if line <= end_line:
            new_end = line
            break

    if new_start is None or new_end is None or new_start > new_end:
        return None

    # 从 new_start 到 new_end 遍历，缺号连续 run > max_gap 才拆分为子区间
    segments = []
    current_seg_start = None
    missing_run = 0

    for line in range(new_start, new_end + 1):
        if line in line_to_subtitle:
            if current_seg_start is None:
                current_seg_start = line
            missing_run = 0
        else:
            missing_run += 1
            if missing_run > max_gap and current_seg_start is not None:
                segments.append((current_seg_start, line - missing_run))
                current_seg_start = None

    if current_seg_start is not None:
        segments.append((current_seg_start, new_end))

    if not segments:
        return None

    result = []
    for seg_start, seg_end in segments:
        new_citation = dict(citation)
        new_citation["开始行号"] = seg_start
        new_citation["结束行号"] = seg_end
        start_sub = line_to_subtitle[seg_start]
        end_sub = line_to_subtitle[seg_end]
        new_citation["开始时间"] = start_sub.start_time
        new_citation["结束时间"] = end_sub.end_time
        result.append(new_citation)

    return result


def convert_subtitle_l2rawID2subtitle_m(
    subtitle_l: list[SubtitleRow],
) -> dict[int, SubtitleRow]:
    assert subtitle_l

    rowID2subtitle_m = {}
    for sub in subtitle_l:
        row_id = sub.rowID
        rowID2subtitle_m[int(row_id)] = sub
    return rowID2subtitle_m


def extract_text_by_row_range(
    subtitle_l: list[SubtitleRow], start_row: int, end_row: int
) -> str:
    """按 rowID 闭区间从字幕里逐字拼接连续文本（仅 text 字段，不含 rowID/Speaker）。

    用于引用校验的重写与引用区间字幕拼接、原文填充。直接拼接字符串、无分隔，
    确保跨行 substring 匹配也能命中。

    区间反转或全部行号越界 → 返回空串。
    """
    if not subtitle_l:
        return ""
    if start_row > end_row:
        return ""
    rowID2subtitle_m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
    parts = []
    for row in range(int(start_row), int(end_row) + 1):
        sub = rowID2subtitle_m.get(row)
        if sub is None:
            continue
        text = sub.text
        if text:
            parts.append(text)
    return "".join(parts)


def _has_citation_keys(data: Any) -> bool:
    """检查数据中是否有任何 '引用' 或 '引用时间' 字段。"""
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ("引用", "引用时间"):
                return True
            if isinstance(value, (dict, list)) and _has_citation_keys(value):
                return True
    elif isinstance(data, list):
        for item in data:
            if _has_citation_keys(item):
                return True
    return False


def _all_citations_empty(data: Any) -> bool:
    """检查数据中所有 '引用' 和 '引用时间' 字段是否都为空列表。"""
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ("引用", "引用时间"):
                if value:
                    return False
            elif isinstance(value, (dict, list)):
                if not _all_citations_empty(value):
                    return False
    elif isinstance(data, list):
        for item in data:
            if not _all_citations_empty(item):
                return False
    return True


def _normalize_citation_list(value: list[Any]) -> list[dict]:
    """将引用列表统一为 [{开始行号, 结束行号}] 格式。

    支持两种输入格式：
    - 新格式：整数数组 [3, 4, 5, 8]  → 合并为区间
    - 旧格式：对象数组 [{"开始行号": 3, "结束行号": 5}]  → 原样保留
    """
    if not value:
        return []
    # 判断是整数数组还是对象数组
    if all(isinstance(item, (int, float)) for item in value):
        return merge_row_ids_to_ranges(value)
    # 对象数组：过滤掉非 dict 项
    return [item for item in value if isinstance(item, dict)]


def convert_nested_row_ids(
    data: Any, line_to_subtitle: dict[int, SubtitleRow],
) -> Any:
    """递归遍历嵌套结构，将所有包含行号的引用转换为时间范围。

    识别 key 为 "引用" 或 "引用时间" 的列表，支持：
    - 新格式：整数数组（LLM 输出的离散 rowID），先合并为区间再转时间
    - 旧格式：{开始行号, 结束行号} 对象数组，直接转时间

    若某个小节（list 中的 dict）的所有引用字段转换后均为空列表，
    则删除该小节。

    Args:
        data: 可能包含嵌套 dict/list 的数据
        line_to_subtitle: 行号(int)到字幕条目的映射

    Returns:
        深拷贝后的新数据，不修改原对象。
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key in ("引用", "引用时间") and isinstance(value, list):
                normalized = _normalize_citation_list(value)
                converted = []
                for item in normalized:
                    if isinstance(item, dict):
                        res = convert_row_range_to_time(item, line_to_subtitle)
                        if res:
                            converted.extend(res)
                result[key] = converted
            elif isinstance(value, (dict, list)):
                result[key] = convert_nested_row_ids(value, line_to_subtitle)
            else:
                result[key] = value
        return result
    elif isinstance(data, list):
        processed = []
        for item in data:
            if isinstance(item, (dict, list)):
                processed_item = convert_nested_row_ids(item, line_to_subtitle)
                # 如果 item 是 dict 且包含引用字段，检查是否所有引用都为空
                if isinstance(processed_item, dict) and _has_citation_keys(processed_item):
                    if _all_citations_empty(processed_item):
                        continue  # 删除该小节
                processed.append(processed_item)
            else:
                processed.append(item)
        return processed
    else:
        return data
