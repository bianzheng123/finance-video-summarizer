"""llm_summarize 工具层单元测试（不触网、不调 LLM）。

覆盖对象是流水线里的纯逻辑：行号↔时间转换、schema 数字矫正、黑话内联、
修复条目整形、prompt 加载与渲染。这些函数决定了引用能否正确落到时间轴、
LLM 返回的字符串数字能否被 schema 接受——出错时症状隐蔽，因此优先覆盖。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_summarize.analysis_common._jargon_utils import (
    _format_with_jargon,
    _strip_jargon_suffix,
    collect_used_jargons,
    inline_array_field,
    inline_single_field,
)
from src.llm_summarize.analysis_common._repair_utils import build_repair_items
from src.llm_summarize.schemas import SubtitleRow
from src.llm_infra.schema import (
    _coerce_numeric_by_schema,
    _load_schema,
    get_step_output_dir,
)
from src.llm_summarize.utils.llm_text_utils import load_prompt, subtitle_dict_to_llm_text
from src.llm_summarize.utils.row_id_2_time_range import (
    convert_nested_row_ids,
    convert_row_range_to_time,
    convert_subtitle_l2rawID2subtitle_m,
    extract_text_by_row_range,
    merge_row_ids_to_ranges,
)


@pytest.fixture
def subtitle_l() -> list[SubtitleRow]:
    """5 条连续字幕，rowID 1..5。"""
    return [
        SubtitleRow(rowID=1, start_time="00:00:01", end_time="00:00:03", text="今天大盘涨了"),
        SubtitleRow(rowID=2, start_time="00:00:03", end_time="00:00:06", text="半导体最强"),
        SubtitleRow(rowID=3, start_time="00:00:06", end_time="00:00:09", text="中芯国际涨停"),
        SubtitleRow(rowID=4, start_time="00:00:09", end_time="00:00:12", text="我觉得可以考虑"),
        SubtitleRow(rowID=5, start_time="00:00:12", end_time="00:00:15", text="但要等回调"),
    ]


# ═══════════════════ merge_row_ids_to_ranges ═══════════════════

class TestMergeRowIdsToRanges:
    """LLM 只吐散点 rowID，靠这个函数合并成区间——合并错会导致引用区间张冠李戴。"""

    def test_consecutive_merged_into_one(self) -> None:
        assert merge_row_ids_to_ranges([3, 4, 5]) == [{"开始行号": 3, "结束行号": 5}]

    def test_gaps_split_into_separate_ranges(self) -> None:
        """缺号行数 ≤ max_gap(2) 会并入同一区间，超过才断开。"""
        assert merge_row_ids_to_ranges([3, 4, 5, 9]) == [
            {"开始行号": 3, "结束行号": 5},
            {"开始行号": 9, "结束行号": 9},
        ]

    def test_unsorted_input_is_sorted_first(self) -> None:
        assert merge_row_ids_to_ranges([9, 4, 3, 5]) == [
            {"开始行号": 3, "结束行号": 5},
            {"开始行号": 9, "结束行号": 9},
        ]

    def test_small_gap_merged_into_single_range(self) -> None:
        """中间只缺 6、7 两行语气词 → 缺号 2 ≤ max_gap，并成一段。"""
        assert merge_row_ids_to_ranges([3, 4, 5, 8]) == [
            {"开始行号": 3, "结束行号": 8},
        ]

    def test_single_id(self) -> None:
        assert merge_row_ids_to_ranges([138]) == [{"开始行号": 138, "结束行号": 138}]

    def test_empty_returns_empty(self) -> None:
        assert merge_row_ids_to_ranges([]) == []

    def test_string_digits_accepted(self) -> None:
        """LLM 偶尔把 rowID 输出成字符串。"""
        assert merge_row_ids_to_ranges(["3", "4"]) == [{"开始行号": 3, "结束行号": 4}]

    def test_none_dropped(self) -> None:
        """None 会被过滤（实现里显式判了 `if r is not None`）。"""
        assert merge_row_ids_to_ranges([3, None, 4]) == [{"开始行号": 3, "结束行号": 4}]

    def test_non_numeric_raises(self) -> None:
        """非数字字符串会让 int() 抛 ValueError——上游 schema 校验负责挡住这种输入，
        本函数不做兜底。记录现状以免误以为它能容错。"""
        with pytest.raises(ValueError):
            merge_row_ids_to_ranges([3, "abc"])

    def test_duplicate_ids_collapse_into_single_range(self) -> None:
        """重复 rowID 被区间合并自然去重（相邻判据对重复值恒成立）。"""
        out = merge_row_ids_to_ranges([3, 3, 4, 4, 5])
        assert out == [{"开始行号": 3, "结束行号": 5}]


# ═══════════════════ rowID → 时间 ═══════════════════

class TestConvertRowRangeToTime:
    """返回值保留原 citation 的其它字段，并补上收缩后的行号 + 时间。"""

    def test_range_maps_to_start_and_end_time(self, subtitle_l: list[SubtitleRow]) -> None:
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        out = convert_row_range_to_time({"开始行号": 2, "结束行号": 4}, m)
        assert len(out) == 1
        assert out[0]["开始时间"] == "00:00:03"
        assert out[0]["结束时间"] == "00:00:12"
        assert out[0]["开始行号"] == 2 and out[0]["结束行号"] == 4

    def test_other_citation_fields_preserved(self, subtitle_l: list[SubtitleRow]) -> None:
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        out = convert_row_range_to_time(
            {"开始行号": 1, "结束行号": 1, "原文": "占位"}, m,
        )
        assert out[0]["原文"] == "占位"

    def test_out_of_range_shrinks_to_valid_bounds(self, subtitle_l: list[SubtitleRow]) -> None:
        """越界边界会双向收缩到最近的有效行号，而不是直接判空。"""
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        out = convert_row_range_to_time({"开始行号": 0, "结束行号": 99}, m)
        assert out[0]["开始行号"] == 1 and out[0]["结束行号"] == 5

    def test_fully_out_of_range_returns_none(self, subtitle_l: list[SubtitleRow]) -> None:
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        assert convert_row_range_to_time({"开始行号": 99, "结束行号": 100}, m) is None

    def test_reversed_range_returns_none(self, subtitle_l: list[SubtitleRow]) -> None:
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        assert convert_row_range_to_time({"开始行号": 4, "结束行号": 2}, m) is None

    def test_empty_map_returns_none(self) -> None:
        assert convert_row_range_to_time({"开始行号": 1, "结束行号": 2}, {}) is None

    def test_gap_splits_into_multiple_segments(self) -> None:
        """字幕经清洗后 rowID 不连续：连续缺号 ≤ max_gap(2) 不拆，
        超过才拆成多段。"""
        sparse = [
            SubtitleRow(rowID=1, start_time="00:00:01", end_time="00:00:02", text="a"),
            SubtitleRow(rowID=2, start_time="00:00:02", end_time="00:00:03", text="b"),
            # rowID 3-6 被清洗删掉（连续缺号 4 行 > max_gap，必须断开）
            SubtitleRow(rowID=7, start_time="00:00:07", end_time="00:00:08", text="g"),
        ]
        m = convert_subtitle_l2rawID2subtitle_m(sparse)
        out = convert_row_range_to_time({"开始行号": 1, "结束行号": 7}, m)
        assert len(out) == 2
        assert (out[0]["开始行号"], out[0]["结束行号"]) == (1, 2)
        assert (out[1]["开始行号"], out[1]["结束行号"]) == (7, 7)

    def test_small_gap_kept_as_single_segment(self) -> None:
        """只缺 1 行（≤ max_gap）→ 不拆分，整段引用。"""
        sparse = [
            SubtitleRow(rowID=1, start_time="00:00:01", end_time="00:00:02", text="a"),
            SubtitleRow(rowID=2, start_time="00:00:02", end_time="00:00:03", text="b"),
            SubtitleRow(rowID=4, start_time="00:00:04", end_time="00:00:05", text="d"),
        ]
        m = convert_subtitle_l2rawID2subtitle_m(sparse)
        out = convert_row_range_to_time({"开始行号": 1, "结束行号": 4}, m)
        assert len(out) == 1
        assert (out[0]["开始行号"], out[0]["结束行号"]) == (1, 4)

    def test_missing_keys_raises_assertion(self, subtitle_l: list[SubtitleRow]) -> None:
        """函数用 assert 声明前置条件，调用方须保证键存在。"""
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        with pytest.raises(AssertionError):
            convert_row_range_to_time({}, m)

    def test_row_id_map_keys_are_ints(self, subtitle_l: list[SubtitleRow]) -> None:
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        assert set(m.keys()) == {1, 2, 3, 4, 5}

    def test_empty_subtitle_list_raises(self) -> None:
        with pytest.raises(AssertionError):
            convert_subtitle_l2rawID2subtitle_m([])


class TestExtractTextByRowRange:
    """首参是 subtitle_l（不是 rowID 映射表），拼接时无分隔符以便跨行 substring 匹配。"""

    def test_concatenates_range_text_without_separator(self, subtitle_l: list[SubtitleRow]) -> None:
        text = extract_text_by_row_range(subtitle_l, 2, 3)
        assert text == "半导体最强中芯国际涨停"

    def test_single_row(self, subtitle_l: list[SubtitleRow]) -> None:
        assert extract_text_by_row_range(subtitle_l, 1, 1) == "今天大盘涨了"

    def test_out_of_range_returns_empty(self, subtitle_l: list[SubtitleRow]) -> None:
        assert extract_text_by_row_range(subtitle_l, 90, 95) == ""

    def test_reversed_range_returns_empty(self, subtitle_l: list[SubtitleRow]) -> None:
        assert extract_text_by_row_range(subtitle_l, 4, 2) == ""

    def test_empty_subtitle_returns_empty(self) -> None:
        assert extract_text_by_row_range([], 1, 2) == ""

    def test_gap_rows_skipped(self) -> None:
        sparse = [
            SubtitleRow(rowID=1, text="a"),
            SubtitleRow(rowID=3, text="c"),
        ]
        assert extract_text_by_row_range(sparse, 1, 3) == "ac"


class TestConvertNestedRowIds:
    """analysis 结构任意嵌套，转换必须递归覆盖所有 引用 / 引用时间 字段，且不改原对象。"""

    def test_nested_int_list_citation_converted(self, subtitle_l: list[SubtitleRow]) -> None:
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        data = {
            "投资机会": [{
                "标题": "中芯国际",
                "短线": {"观点": "看多 🐂📈", "引用": [3, 4]},
            }],
        }
        out = convert_nested_row_ids(data, m)
        cits = out["投资机会"][0]["短线"]["引用"]
        assert len(cits) == 1
        assert cits[0]["开始时间"] == "00:00:06"
        assert cits[0]["结束时间"] == "00:00:12"

    def test_original_not_mutated(self, subtitle_l: list[SubtitleRow]) -> None:
        """函数承诺返回深拷贝。"""
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        data = {"x": {"引用": [1, 2]}}
        convert_nested_row_ids(data, m)
        assert data["x"]["引用"] == [1, 2]

    def test_old_format_range_objects_also_converted(self, subtitle_l: list[SubtitleRow]) -> None:
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        data = {"x": {"引用": [{"开始行号": 1, "结束行号": 2}]}}
        out = convert_nested_row_ids(data, m)
        assert out["x"]["引用"][0]["开始时间"] == "00:00:01"

    def test_引用时间_key_also_handled(self, subtitle_l: list[SubtitleRow]) -> None:
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        out = convert_nested_row_ids({"x": {"引用时间": [1]}}, m)
        assert out["x"]["引用时间"][0]["开始时间"] == "00:00:01"

    def test_section_dropped_when_all_citations_empty(self, subtitle_l: list[SubtitleRow]) -> None:
        """引用全部越界 → 该小节整条剔除，避免渲染出无支撑的条目。"""
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        data = {"投资机会": [
            {"标题": "有效", "引用": [1]},
            {"标题": "全越界", "引用": [900, 901]},
        ]}
        out = convert_nested_row_ids(data, m)
        titles = [o["标题"] for o in out["投资机会"]]
        assert titles == ["有效"]

    def test_non_dict_input_passthrough(self, subtitle_l: list[SubtitleRow]) -> None:
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        assert convert_nested_row_ids("plain", m) == "plain"

    def test_fields_without_citations_untouched(self, subtitle_l: list[SubtitleRow]) -> None:
        m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
        out = convert_nested_row_ids({"观点": "看多 🐂📈", "n": 42}, m)
        assert out == {"观点": "看多 🐂📈", "n": 42}


# ═══════════════════ schema 数字矫正 ═══════════════════

class TestCoerceNumericBySchema:
    """LLM 常把 integer 字段输出成字符串；但 ETF/港股代码是 string 且长得像数字，
    误转会把 "00700" 变成 700 从而校验失败。这组用例锁死这个边界。"""

    def test_string_int_coerced_when_schema_says_integer(self) -> None:
        schema = {"type": "object", "properties": {"rowID": {"type": "integer"}}}
        assert _coerce_numeric_by_schema({"rowID": "1741"}, schema) == {"rowID": 1741}

    def test_string_kept_when_schema_says_string(self) -> None:
        """港股代码必须保持前导零。"""
        schema = {"type": "object", "properties": {"代码": {"type": "string"}}}
        assert _coerce_numeric_by_schema({"代码": "00700"}, schema) == {"代码": "00700"}

    def test_float_coerced_for_number(self) -> None:
        schema = {"type": "object", "properties": {"涨幅": {"type": "number"}}}
        assert _coerce_numeric_by_schema({"涨幅": "3.14"}, schema) == {"涨幅": 3.14}

    def test_array_items_coerced(self) -> None:
        schema = {"type": "object", "properties": {
            "引用": {"type": "array", "items": {"type": "integer"}}}}
        assert _coerce_numeric_by_schema({"引用": ["1", "2"]}, schema) == {"引用": [1, 2]}

    def test_uncoercible_string_left_alone(self) -> None:
        schema = {"type": "object", "properties": {"rowID": {"type": "integer"}}}
        assert _coerce_numeric_by_schema({"rowID": "abc"}, schema) == {"rowID": "abc"}

    def test_oneof_schema_skipped(self) -> None:
        """含 oneOf/anyOf 时无法确定分支，应原样返回。"""
        schema = {"oneOf": [{"type": "integer"}, {"type": "string"}]}
        assert _coerce_numeric_by_schema("123", schema) == "123"

    def test_none_schema_passthrough(self) -> None:
        assert _coerce_numeric_by_schema({"a": "1"}, None) == {"a": "1"}

    def test_unknown_property_untouched(self) -> None:
        schema = {"type": "object", "properties": {"known": {"type": "integer"}}}
        out = _coerce_numeric_by_schema({"known": "1", "extra": "2"}, schema)
        assert out == {"known": 1, "extra": "2"}


class TestSchemaLoading:
    def test_load_real_stage_schema(self) -> None:
        anchor = str(Path(__file__).parent.parent
                     / "src/llm_summarize/step7_topic/anchor.py")
        schema = _load_schema("7_1_主题分析", anchor)
        assert schema is not None
        assert "投资机会" in schema.get("properties", {})

    def test_missing_schema_returns_none(self) -> None:
        anchor = str(Path(__file__).parent.parent
                     / "src/llm_summarize/step7_topic/anchor.py")
        assert _load_schema("不存在的阶段", anchor) is None

    def test_get_step_output_dir_creates(self, tmp_path: Path) -> None:
        d = get_step_output_dir(tmp_path, 4)
        assert d.exists() and d.name == "llm_output_4"


# ═══════════════════ 黑话内联 ═══════════════════

class TestJargonInlining:
    def test_format_appends_jargon_in_parens(self) -> None:
        assert _format_with_jargon("中芯国际", ["大国脊梁"]) == "中芯国际（大国脊梁）"

    def test_format_multiple_jargons_joined(self) -> None:
        out = _format_with_jargon("宁德时代", ["宁王", "宁德"])
        assert out.startswith("宁德时代（") and "宁王" in out and "宁德" in out

    def test_format_without_jargon_returns_plain(self) -> None:
        assert _format_with_jargon("中芯国际", []) == "中芯国际"

    def test_strip_suffix_is_inverse(self) -> None:
        assert _strip_jargon_suffix("中芯国际（大国脊梁）") == "中芯国际"

    def test_strip_suffix_noop_when_absent(self) -> None:
        assert _strip_jargon_suffix("中芯国际") == "中芯国际"

    def test_inline_single_field_mutates(self) -> None:
        parent = {"标题": "中芯国际"}
        inline_single_field(parent, "标题", {"中芯国际": ["大国脊梁"]})
        assert parent["标题"] == "中芯国际（大国脊梁）"

    def test_inline_single_field_skips_unknown(self) -> None:
        parent = {"标题": "贵州茅台"}
        inline_single_field(parent, "标题", {"中芯国际": ["大国脊梁"]})
        assert parent["标题"] == "贵州茅台"

    def test_inline_array_field_mutates_each(self) -> None:
        parent = {"涉及资产": ["中芯国际", "宁德时代"]}
        inline_array_field(parent, "涉及资产", {
            "中芯国际": ["大国脊梁"], "宁德时代": ["宁王"],
        })
        assert parent["涉及资产"] == ["中芯国际（大国脊梁）", "宁德时代（宁王）"]

    def test_collect_used_jargons_reads_incremental_and_matches_subtitle(self, tmp_path: Path) -> None:
        """签名是 (subtitle_l, base_dir, keyword_data)：
        增量黑话从 `base_dir.parent/incremental_database/jargon_incremental.json` 读，
        再叠加全局词表做子串匹配。这里只验证增量来源被正确读取并参与匹配。
        """
        inc_dir = tmp_path / "incremental_database"
        inc_dir.mkdir()
        # 结构见 jargon_loader 模块 docstring：{"个股黑话": {"<黑话>": {"正式名": ...}}}
        (inc_dir / "jargon_incremental.json").write_text(
            json.dumps(
                {"个股黑话": {"测试黑话甲": {"正式名": "测试正式名甲"}}},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        base_dir = tmp_path / "llm_analysis"
        base_dir.mkdir()

        used = collect_used_jargons(
            [{"rowID": 1, "text": "测试黑话甲今天涨停"}], base_dir, {},
        )
        # 增量库里的黑话应被收集（不论字幕是否命中，来源 1 是无条件加入的）
        assert "测试正式名甲" in used
        assert used["测试正式名甲"] == ["测试黑话甲"]

    def test_collect_used_jargons_tolerates_missing_incremental_file(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "llm_analysis"
        base_dir.mkdir()
        used = collect_used_jargons([{"rowID": 1, "text": "随便一句"}], base_dir, {})
        assert isinstance(used, dict)


# ═══════════════════ 修复条目整形 ═══════════════════

class TestBuildRepairItems:
    def test_none_returns_empty(self) -> None:
        assert build_repair_items(None) == []

    def test_fields_extracted(self) -> None:
        out = build_repair_items([{
            "字段": "投资机会[0].短线", "失败原因": "引用无方向性原话",
            "原始内容": {"观点": "看多 🐂📈"},
        }])
        assert out[0]["字段"] == "投资机会[0].短线"
        assert out[0]["失败原因"] == "引用无方向性原话"
        assert "看多" in out[0]["原始内容片段"]

    def test_snippet_truncated(self) -> None:
        out = build_repair_items(
            [{"字段": "x", "失败原因": "y", "原始内容": {"k": "长" * 500}}],
            snippet_chars=50,
        )
        assert len(out[0]["原始内容片段"]) == 50

    def test_empty_content_yields_empty_snippet(self) -> None:
        out = build_repair_items([{"字段": "x", "失败原因": "y", "原始内容": ""}])
        assert out[0]["原始内容片段"] == ""

    def test_snippet_is_valid_json_prefix(self) -> None:
        """片段用于 prompt 展示，不要求可解析，但不能是 Python repr。"""
        out = build_repair_items([{"字段": "x", "失败原因": "y", "原始内容": {"观点": "看多"}}])
        assert out[0]["原始内容片段"].startswith("{")
        assert "'" not in out[0]["原始内容片段"]  # 非 Python repr


# ═══════════════════ prompt 加载与字幕序列化 ═══════════════════

class TestPromptAndSubtitleText:
    def test_subtitle_serialized_as_tab_separated(self) -> None:
        out = subtitle_dict_to_llm_text([
            SubtitleRow(rowID=1, speaker=0, text="大盘涨了"),
        ])
        assert out == "1\t大盘涨了"

    def test_multiple_lines_newline_joined(self) -> None:
        out = subtitle_dict_to_llm_text([
            SubtitleRow(rowID=1, speaker=0, text="第一句"),
            SubtitleRow(rowID=2, speaker=0, text="第二句"),
        ])
        assert out.split("\n") == ["1\t第一句", "2\t第二句"]

    def test_load_real_stage_prompt(self) -> None:
        anchor = str(Path(__file__).parent.parent
                     / "src/llm_summarize/step7_topic/anchor.py")
        text = load_prompt("7_1_主题分析.txt", anchor)
        assert "投资机会" in text

    def test_load_prompt_missing_raises(self) -> None:
        anchor = str(Path(__file__).parent.parent
                     / "src/llm_summarize/step7_topic/anchor.py")
        with pytest.raises(FileNotFoundError):
            load_prompt("不存在.txt", anchor)
