"""跨 section 后处理（通用部分）。

LLM 跑完后顺序执行的、对所有 section 一视同仁的后处理：
- filter_empty_fields: 单 section 空字段清理
- convert_row_ids_to_time_ranges: 行号 → 时间范围
- populate_citation_texts: 时间范围 → 原文字段

section 专属的后处理（主题引用合并 / 引用总时长 / 黑话注入等）已下沉到对应
analyzer（TopicAnalyzer / MacroAnalyzer），由 TopicAnalysisStep 在并行池或
跨 section 阶段按需调用。
"""

import logging
from typing import Any

from ..schemas import SubtitleRow
from ..utils.row_id_2_time_range import (
    convert_subtitle_l2rawID2subtitle_m,
    convert_nested_row_ids,
)
from ..utils.llm_text_utils import llm_text_of, row_is_analyzable, strip_paragraph_referents
from ...utils.time_utils import hms_to_seconds

logger = logging.getLogger(__name__)


class PostProcessor:
    """通用后处理。所有方法 staticmethod。"""

    # ==================== 单章节空字段清理 ====================

    @staticmethod
    def filter_empty_fields(data: dict) -> dict:
        """过滤掉空数组和空字典的字段。"""
        if not isinstance(data, dict):
            return data

        filtered_dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                if "具体方法" in value:
                    references = value.get("引用", [])
                    if not references:
                        continue
                    filtered_dict[key] = value
                elif value:
                    filtered_dict[key] = value
            elif isinstance(value, list):
                filtered_list = [item for item in value if item not in (None, "", [], {})]
                if filtered_list:
                    filtered_dict[key] = filtered_list
            else:
                if value not in (None, "", [], {}):
                    filtered_dict[key] = value

        return filtered_dict

    # ==================== 行号 → 时间 ====================

    @staticmethod
    def convert_row_ids_to_time_ranges(analysis: dict, subtitle_l: list[SubtitleRow]) -> dict:
        """对整张 analysis 做行号→时间转换（递归）。"""
        rowID2subtitle_m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        return convert_nested_row_ids(analysis, rowID2subtitle_m)

    @staticmethod
    def convert_row_ids_in_section(section_data: dict, subtitle_l: list[SubtitleRow]) -> dict:
        """对单个 section 做行号→时间转换。供并行池中的 worker 复用。"""
        rowID2subtitle_m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        return convert_nested_row_ids(section_data, rowID2subtitle_m)

    # ==================== 时间范围 → 原文 ====================

    @staticmethod
    def _extract_subtitle_text(subtitle_l: list[SubtitleRow], start_time: str, end_time: str) -> str:
        start_sec = hms_to_seconds(start_time)
        end_sec = hms_to_seconds(end_time)

        extracted_texts = []
        if not subtitle_l:
            return ""

        for sub in subtitle_l:
            # 引用原文不收录无关/噪音行（保住可溯源）
            if not row_is_analyzable(sub):
                continue
            line_start = sub.start_time
            line_end = sub.end_time
            text = sub.text
            if not line_start or not line_end or not text:
                continue
            try:
                line_start_sec = hms_to_seconds(line_start)
                line_end_sec = hms_to_seconds(line_end)
                # 闭区间重叠判定：单秒行（start==end）恰落在引用边界时必须收录
                if line_start_sec <= end_sec and line_end_sec >= start_sec:
                    extracted_texts.append(strip_paragraph_referents(llm_text_of(sub)).strip())
            except Exception:
                continue
        return " ".join(extracted_texts) if extracted_texts else ""

    @staticmethod
    def populate_citation_texts(analysis: dict, subtitle_l: list[SubtitleRow]) -> dict:
        """递归找到所有 引用[]，按 开始时间/结束时间 写入 原文 字段。"""
        for section_data in analysis.values():
            PostProcessor.populate_citation_texts_in_section(section_data, subtitle_l)
        return analysis

    @staticmethod
    def populate_citation_texts_in_section(section_data: dict, subtitle_l: list[SubtitleRow]) -> dict:
        """对单个 section 递归填充 引用[].原文 字段。供并行池中的 worker 复用。"""
        def process_value(value: Any) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    if k == "引用" and isinstance(v, list):
                        for citation in v:
                            if isinstance(citation, dict) and "开始时间" in citation and "结束时间" in citation:
                                start_time = citation["开始时间"]
                                end_time = citation["结束时间"]
                                citation["原文"] = PostProcessor._extract_subtitle_text(
                                    subtitle_l, start_time, end_time,
                                )
                    elif isinstance(v, (dict, list)):
                        process_value(v)
            elif isinstance(value, list):
                for item in value:
                    process_value(item)

        process_value(section_data)
        return section_data
