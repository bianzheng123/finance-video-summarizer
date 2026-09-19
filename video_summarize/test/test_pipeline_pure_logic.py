"""流水线纯逻辑单元测试（不触网、不调 LLM）。

与 test_llm_summarize_utils.py 互补：那份覆盖行号↔时间、schema 矫正等"跨阶段工具"，
本份覆盖各阶段**自己的**纯计算部分——这些代码原本只在真跑一次完整分析时才被执行，
所以回归最难发现。选取标准是「出错会直接违反 CLAUDE.md 三条硬性要求」：

- pinyin_matcher：同音纠错。误替换会把字幕改成主讲人没说过的词 → 破坏可溯源。
- post_processor：引用时间 → 原文回填。回填错会让引用对不上原话 → 破坏可溯源。
- citation_expander：主导性检测。判错会把一笔带过的提及放大成整段引用 → 伪主题。
- segment_merger：合并决策的传递闭包。合并错会丢主题或串主题 → 破坏全面性。
- sensitivity_aggregator / sensitivity_detector：主讲人"私域再讲"信号的召回与聚合。
- topic_referencer：空子板块销毁时的引用合并。丢引用 → 破坏全面性。
"""

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_summarize.citation_check.citation_validator import (
    CitationValidator,
    _has_citations,
    _sanitize_filename,
)
from src.llm_summarize.analysis_common.post_processor import PostProcessor
from src.llm_summarize.step6_trading.sensitivity_aggregator import (
    SensitivityAggregator,
)
from src.llm_summarize.step6_trading.sensitivity_detector import (
    _SENSITIVITY_PATTERNS,
    _build_quote,
    _cluster_signals,
    _collect_hit_rowids,
    _find_anchored_topics,
    _safe_int,
)
from src.llm_summarize.subtitle_cleaning.pinyin_matcher import (
    _build_hotword_pinyin_index,
    _build_truncation_index,
    _find_suspect_candidates,
    _find_truncation_candidates,
    _is_all_han,
    _is_exact_homophone,
    _pinyin_display,
    _pinyin_distance,
    _split_syllable,
    _to_pinyin_tuple,
    row_reading,
    scan_row_candidates,
)
from src.llm_summarize.topic_segmentation.segment_merger import (
    _apply_merge_decisions,
)
from src.llm_summarize.topic_segmentation.topic_segmenter import (
    _clip_segments_to_scope,
)
from src.llm_summarize.utils.llm_text_utils import (
    N_REDUNDANT_ROWS,
    render_full_subtitle,
    render_windowed_subtitle,
)
from src.llm_summarize.utils.rowid_scope import (
    filter_citations,
    filter_items_by_rowid,
    filter_range_citations,
    filter_row_ids,
    format_scope,
)
from src.llm_summarize.clustering.citation_expander import (
    _anchors_to_singletons,
    _coerce_int_anchors,
    _merge_overlapping_ranges,
    _ranges_for_anchors,
    _segment_label_matches_topic,
)
from src.llm_summarize.clustering import ClusteringStep
from src.llm_summarize.utils.jargon_loader import load_incremental_jargon
from src.utils.time_utils import fmt_time, hms_to_seconds

@pytest.fixture
def subtitle_l() -> list[dict]:
    """3 条 5 秒一句的字幕，时间轴首尾相接（真实 ASR 输出就是这个形态）。"""
    return [
        {"rowID": 1, "start_time": "00:00:00", "end_time": "00:00:05", "text": "第一句"},
        {"rowID": 2, "start_time": "00:00:05", "end_time": "00:00:10", "text": "第二句"},
        {"rowID": 3, "start_time": "00:00:10", "end_time": "00:00:15", "text": "第三句"},
    ]


# ═══════════════════ 拼音拆分与距离 ═══════════════════

class TestSyllableSplit:
    """TONE3 音节拆成 (声母, 韵母, 声调)。拆错会让整条同音链路失准。"""

    def test_initial_and_final_and_tone(self) -> None:
        assert _split_syllable("lan2") == ("l", "an", 2)

    def test_zero_initial(self) -> None:
        """零声母音节（an1）声母为空字符串，不能误吞第一个字母。"""
        assert _split_syllable("an1") == ("", "an", 1)

    def test_two_char_initial_matched_greedily(self) -> None:
        """zh/ch/sh 必须整体当声母——按长度倒序匹配的原因。"""
        assert _split_syllable("zhong1") == ("zh", "ong", 1)

    def test_no_tone_digit_gives_tone_zero(self) -> None:
        assert _split_syllable("er") == ("", "er", 0)

    def test_empty_string(self) -> None:
        assert _split_syllable("") == ("", "", 0)


class TestPinyinTuple:
    def test_homophone_words_share_identical_tuple(self) -> None:
        """茅台/毛台 是最典型的 ASR 误识别对，拼音元组必须完全一致。"""
        assert _to_pinyin_tuple("茅台") == _to_pinyin_tuple("毛台")

    def test_tone_difference_preserved_in_tuple(self) -> None:
        """澜起 lán qǐ vs 蓝旗 lán qí：声母韵母同、声调不同。"""
        a, b = _to_pinyin_tuple("澜起"), _to_pinyin_tuple("蓝旗")
        assert [x[:2] for x in a] == [x[:2] for x in b]
        assert [x[2] for x in a] != [x[2] for x in b]

    def test_tuple_length_equals_char_count(self) -> None:
        assert len(_to_pinyin_tuple("中芯国际")) == 4

    def test_display_has_tone_marks(self) -> None:
        assert _pinyin_display("澜起") == "lán qǐ"


class TestPinyinDistance:
    def test_exact_homophone_distance_zero(self) -> None:
        assert _pinyin_distance(_to_pinyin_tuple("茅台"), _to_pinyin_tuple("毛台")) == 0

    def test_tone_sensitive_counts_tone_diff(self) -> None:
        a, b = _to_pinyin_tuple("澜起"), _to_pinyin_tuple("蓝旗")
        assert _pinyin_distance(a, b, tone_sensitive=False) == 0
        assert _pinyin_distance(a, b, tone_sensitive=True) == 1

    def test_length_mismatch_returns_none(self) -> None:
        """不跨长度匹配——否则 2 字词会被 3 字热词吸走。"""
        assert _pinyin_distance(
            _to_pinyin_tuple("茅台"), _to_pinyin_tuple("茅台酒"),
        ) is None

    def test_is_exact_homophone_ignores_tone(self) -> None:
        assert _is_exact_homophone(_to_pinyin_tuple("澜起"), _to_pinyin_tuple("蓝旗"))

    def test_is_exact_homophone_length_mismatch_false(self) -> None:
        assert not _is_exact_homophone(
            _to_pinyin_tuple("茅台"), _to_pinyin_tuple("茅台酒"),
        )


class TestIsAllHan:
    def test_pure_han_true(self) -> None:
        assert _is_all_han("茅台")

    def test_mixed_latin_false(self) -> None:
        """含字母的 token 不参与同音替换，避免动到 ETF 代码这类混排文本。"""
        assert not _is_all_han("茅台A")

    def test_punctuation_false(self) -> None:
        assert not _is_all_han("，")

    def test_empty_string_is_vacuously_true(self) -> None:
        """all() 对空串返回 True。调用方都先判了非空，这里锁定现状。"""
        assert _is_all_han("")


# ═══════════════════ 热词索引与候选查找 ═══════════════════

class TestHotwordPinyinIndex:
    """索引按词长分桶。权重/长度过滤错了会让纠错要么失灵要么过度激进。"""

    def test_indexed_by_word_length(self) -> None:
        idx = _build_hotword_pinyin_index({"100": ["茅台", "中芯国际"]}, {})
        assert set(idx.keys()) == {2, 4}

    def test_formal_name_taken_from_jargon_map(self) -> None:
        idx = _build_hotword_pinyin_index({"100": ["茅台"]}, {"茅台": ["贵州茅台"]})
        assert idx[2][0][2] == "贵州茅台"

    def test_formal_defaults_to_word_itself(self) -> None:
        idx = _build_hotword_pinyin_index({"100": ["茅台"]}, {})
        assert idx[2][0][2] == "茅台"

    def test_below_min_weight_excluded(self) -> None:
        """低权重热词不进索引——它们置信度不足以驱动自动替换。"""
        assert _build_hotword_pinyin_index({"11": ["宁德时代"]}, {}) == {}

    def test_min_weight_override_includes_it(self) -> None:
        idx = _build_hotword_pinyin_index({"11": ["宁德时代"]}, {}, min_weight=11)
        assert 4 in idx

    def test_single_char_word_excluded_by_default(self) -> None:
        """单字词默认被 min_word_len=2 挡掉，否则满篇误替。"""
        assert _build_hotword_pinyin_index({"100": ["长"]}, {}) == {}

    def test_too_long_word_excluded(self) -> None:
        assert _build_hotword_pinyin_index({"100": ["一二三四五六"]}, {}) == {}

    def test_non_numeric_weight_key_skipped(self) -> None:
        assert _build_hotword_pinyin_index({"abc": ["茅台"]}, {}) == {}

    def test_non_list_value_skipped(self) -> None:
        assert _build_hotword_pinyin_index({"100": "茅台"}, {}) == {}

    def test_non_string_entry_skipped(self) -> None:
        idx = _build_hotword_pinyin_index({"100": [None, 123, "茅台"]}, {})
        assert [w for w, _, _, _ in idx[2]] == ["茅台"]


class TestFindSuspectCandidates:
    """扫可疑同音词注入 prompt。漏扫 → 错字留在总结里；多扫 → prompt 噪声。"""

    @pytest.fixture
    def idx(self) -> dict[int, list[tuple[str, tuple, str, int]]]:
        return _build_hotword_pinyin_index(
            {"100": ["茅台", "中芯国际"]}, {"茅台": ["贵州茅台"]},
        )

    def test_homophone_typo_detected(self, idx: dict[int, list[tuple[str, tuple, str, int]]]) -> None:
        out = _find_suspect_candidates("毛台今天涨了", idx)
        assert len(out) == 1
        assert out[0]["original"] == "毛台"
        assert out[0]["suspect"] == "茅台"
        assert out[0]["formal"] == "贵州茅台"

    def test_tone_match_flag_true_for_same_tone(self, idx: dict[int, list[tuple[str, tuple, str, int]]]) -> None:
        """茅台/毛台 连声调都同 → tone_match=True，置信度更高。"""
        assert _find_suspect_candidates("毛台今天涨了", idx)[0]["tone_match"] is True

    def test_correct_word_not_flagged(self, idx: dict[int, list[tuple[str, tuple, str, int]]]) -> None:
        """本身就是热词的滑窗要跳过，否则会把正确的词报成可疑。"""
        assert _find_suspect_candidates("茅台今天涨了", idx) == []

    def test_unrelated_text_yields_nothing(self, idx: dict[int, list[tuple[str, tuple, str, int]]]) -> None:
        assert _find_suspect_candidates("今天大盘怎么样", idx) == []

    def test_empty_text_returns_empty(self, idx: dict[int, list[tuple[str, tuple, str, int]]]) -> None:
        assert _find_suspect_candidates("", idx) == []

    def test_empty_index_returns_empty(self) -> None:
        assert _find_suspect_candidates("毛台", {}) == []

    def test_max_per_line_caps_output(self) -> None:
        idx = _build_hotword_pinyin_index({"100": ["茅台", "澜起"]}, {})
        out = _find_suspect_candidates("毛台和蓝旗都涨", idx, max_per_line=1)
        assert len(out) == 1

    def test_duplicate_token_reported_once(self, idx: dict[int, list[tuple[str, tuple, str, int]]]) -> None:
        out = _find_suspect_candidates("毛台涨了毛台又涨", idx)
        assert len(out) == 1


class TestTruncationIndex:
    """吞首字（ASR 常把"中芯国际"识别成"芯国际"）的字符串匹配索引。"""

    def test_index_keyed_by_truncated_form(self) -> None:
        idx = _build_truncation_index({"100": ["中芯国际", "宁德时代"]}, {})
        assert set(idx.keys()) == {"芯国际", "德时代"}

    def test_value_is_full_word_and_formal(self) -> None:
        idx = _build_truncation_index({"100": ["中芯国际"]}, {"中芯国际": ["中芯国际股份"]})
        assert idx["芯国际"] == ("中芯国际", "中芯国际股份")

    def test_two_char_word_excluded(self) -> None:
        """2 字热词去掉一字只剩单字，误伤面太大 → min_word_len=3 挡掉。"""
        assert _build_truncation_index({"100": ["茅台"]}, {}) == {}

    def test_below_min_weight_excluded(self) -> None:
        assert _build_truncation_index({"11": ["中芯国际"]}, {}) == {}

    def test_first_wins_on_collision(self) -> None:
        idx = _build_truncation_index({"100": ["中芯国际", "大芯国际"]}, {})
        assert idx["芯国际"][0] == "中芯国际"

    def test_non_numeric_weight_skipped(self) -> None:
        assert _build_truncation_index({"x": ["中芯国际"]}, {}) == {}


class TestFindTruncationCandidates:
    @pytest.fixture
    def idx(self) -> dict[int, list[tuple[str, tuple, str, int]]]:
        return _build_truncation_index({"100": ["中芯国际"]}, {})

    def test_truncated_form_found(self, idx: dict[int, list[tuple[str, tuple, str, int]]]) -> None:
        out = _find_truncation_candidates("我看芯国际不错", idx)
        assert out == [{
            "original": "芯国际", "suspect": "中芯国际",
            "formal": "中芯国际", "type": "truncation",
        }]

    def test_no_match_returns_empty(self, idx: dict[int, list[tuple[str, tuple, str, int]]]) -> None:
        assert _find_truncation_candidates("今天大盘不错", idx) == []

    def test_empty_text_returns_empty(self, idx: dict[int, list[tuple[str, tuple, str, int]]]) -> None:
        assert _find_truncation_candidates("", idx) == []

    def test_empty_index_returns_empty(self) -> None:
        assert _find_truncation_candidates("芯国际", {}) == []


class TestRowReading:
    """整行读音（[ANNOTATION] 的 `行读音`）。"""

    def test_basic_reading(self) -> None:
        assert row_reading("澜起") == "lán qǐ"

    def test_full_row_reading(self) -> None:
        out = row_reading("我得去跟小卯叔的人投诉")
        assert "xiǎo mǎo shū" in out

    def test_empty_text(self) -> None:
        assert row_reading("") == ""


class TestScanRowCandidates:
    """批量拼音候选扫描（2_1/2_2 的 [ANNOTATION] 拼音候选）。"""

    @pytest.fixture
    def idx(self) -> dict[int, list[tuple[str, tuple, str, int]]]:
        return _build_hotword_pinyin_index(
            {"100": ["兆易"]}, {"兆易": ["兆易创新"]},
        )

    def test_candidate_carries_reading(
        self, idx: dict[int, list[tuple[str, tuple, str, int]]],
    ) -> None:
        out = scan_row_candidates({7: "赵毅创新今天涨了"}, idx)
        assert out[7] == [{"片段": "赵毅", "读音": "zhào yì", "音近": ["兆易"]}]

    def test_formal_name_withheld_by_default(
        self, idx: dict[int, list[tuple[str, tuple, str, int]]],
    ) -> None:
        """2_1 的定位是不给词表——正式名属词表知识，默认必须不带。"""
        out = scan_row_candidates({7: "赵毅创新今天涨了"}, idx)
        assert "正式名" not in out[7][0]

    def test_formal_name_included_when_requested(
        self, idx: dict[int, list[tuple[str, tuple, str, int]]],
    ) -> None:
        out = scan_row_candidates({7: "赵毅创新今天涨了"}, idx, include_formal=True)
        assert out[7][0]["正式名"] == "兆易创新"

    def test_rows_without_candidates_absent(
        self, idx: dict[int, list[tuple[str, tuple, str, int]]],
    ) -> None:
        """无候选的行整键不出现（而非给空数组），保持 [ANNOTATION] 紧凑。"""
        out = scan_row_candidates({7: "赵毅创新", 8: "今天大盘还行", 9: ""}, idx)
        assert set(out) == {7}

    def test_empty_index_returns_empty(self) -> None:
        assert scan_row_candidates({7: "赵毅创新"}, {}) == {}


# ═══════════════════ 空字段清理 ═══════════════════

class TestFilterEmptyFields:
    """空字段留在 analysis 里会渲染出空标题的条目。"""

    def test_empty_containers_dropped(self) -> None:
        out = PostProcessor.filter_empty_fields(
            {"a": [], "b": {}, "c": "", "d": None, "f": "x"},
        )
        assert out == {"f": "x"}

    def test_zero_is_kept(self) -> None:
        """0 不是"空"——引用总时长=0 这类字段必须保留。"""
        assert PostProcessor.filter_empty_fields({"e": 0}) == {"e": 0}

    def test_empty_items_pruned_from_list(self) -> None:
        out = PostProcessor.filter_empty_fields({"g": [1, None, "", {}]})
        assert out == {"g": [1]}

    def test_list_dropped_when_all_items_empty(self) -> None:
        assert PostProcessor.filter_empty_fields({"g": [None, "", {}]}) == {}

    def test_具体方法_without_citations_dropped(self) -> None:
        """交易技巧条目没引用就是无支撑内容 → 必须剔除（可溯源要求）。"""
        assert PostProcessor.filter_empty_fields(
            {"k": {"具体方法": "追高", "引用": []}},
        ) == {}

    def test_具体方法_with_citations_kept(self) -> None:
        data = {"k": {"具体方法": "追高", "引用": [1]}}
        assert PostProcessor.filter_empty_fields(data) == data

    def test_non_dict_passthrough(self) -> None:
        assert PostProcessor.filter_empty_fields("plain") == "plain"
        assert PostProcessor.filter_empty_fields([1]) == [1]


# ═══════════════════ 引用原文回填 ═══════════════════

class TestExtractSubtitleText:
    """按时间区间取原文。取错 = 引用与原话对不上 → 违反可溯源。"""

    def test_overlapping_lines_joined_with_space(self, subtitle_l: list[dict]) -> None:
        """闭区间重叠：恰压在引用终点上的邻句（第三句 start==10）也收录。"""
        assert PostProcessor._extract_subtitle_text(
            subtitle_l, "00:00:00", "00:00:10",
        ) == "第一句 第二句 第三句"

    def test_boundary_is_closed_interval(self, subtitle_l: list[dict]) -> None:
        """判据是 line_start <= end 且 line_end >= start（闭区间）：
        00:00:05–00:00:10 会把首尾相接的邻句都带进来——
        单秒行（start==end）恰落在引用边界时必须能命中。"""
        assert PostProcessor._extract_subtitle_text(
            subtitle_l, "00:00:05", "00:00:10",
        ) == "第一句 第二句 第三句"

    def test_no_overlap_returns_empty(self, subtitle_l: list[dict]) -> None:
        assert PostProcessor._extract_subtitle_text(
            subtitle_l, "00:01:00", "00:02:00",
        ) == ""

    def test_empty_subtitle_returns_empty(self) -> None:
        assert PostProcessor._extract_subtitle_text([], "00:00:00", "00:00:10") == ""

    def test_lines_missing_time_fields_skipped(self) -> None:
        assert PostProcessor._extract_subtitle_text(
            [{"text": "无时间戳"}], "00:00:00", "00:01:00",
        ) == ""

    def test_partial_overlap_included(self, subtitle_l: list[dict]) -> None:
        """只压到一点也算命中——宁可多带上下文，不可漏掉支撑原话。"""
        assert PostProcessor._extract_subtitle_text(
            subtitle_l, "00:00:04", "00:00:06",
        ) == "第一句 第二句"


class TestPopulateCitationTexts:
    def test_citation_text_filled_across_sections(self, subtitle_l: list[dict]) -> None:
        analysis = {"半导体": {"投资机会": [
            {"标题": "T", "引用": [{"开始时间": "00:00:00", "结束时间": "00:00:06"}]},
        ]}}
        PostProcessor.populate_citation_texts(analysis, subtitle_l)
        cit = analysis["半导体"]["投资机会"][0]["引用"][0]
        assert cit["原文"] == "第一句 第二句"

    def test_deeply_nested_citations_filled(self, subtitle_l: list[dict]) -> None:
        section = {
            "引用": [{"开始时间": "00:00:00", "结束时间": "00:00:03"}],
            "深": {"引用": [{"开始时间": "00:00:10", "结束时间": "00:00:14"}]},
        }
        PostProcessor.populate_citation_texts_in_section(section, subtitle_l)
        assert section["引用"][0]["原文"] == "第一句"
        # 闭区间边界：第二句 end==10 压在引用起点上，一并收录
        assert section["深"]["引用"][0]["原文"] == "第二句 第三句"

    def test_citation_without_time_fields_skipped(self, subtitle_l: list[dict]) -> None:
        section = {"引用": [{"开始行号": 1}]}
        PostProcessor.populate_citation_texts_in_section(section, subtitle_l)
        assert "原文" not in section["引用"][0]

    def test_mutates_in_place_and_returns_same_object(self, subtitle_l: list[dict]) -> None:
        analysis = {"S": {"引用": [{"开始时间": "00:00:00", "结束时间": "00:00:03"}]}}
        assert PostProcessor.populate_citation_texts(analysis, subtitle_l) is analysis


# ═══════════════════ 片段合并的传递闭包 ═══════════════════

class TestApplyMergeDecisions:
    """LLM 给每个相邻重要段对一份完整 result（含 `是否合并` 与合并后段的
    `新话题标签`/`事件`/`是否重要`/`困惑`/`困惑原因`），连续 `是否合并=true` 的
    段对把整条链合成一段（中间可夹不重要段）。
    合并错会把两个主题串成一个（丢主题）或切断一个主题（丢内容）。"""

    @pytest.fixture
    def segments(self) -> list[dict]:
        return [
            {"开始行号": 1, "结束行号": 10, "话题标签": "A"},
            {"开始行号": 11, "结束行号": 20, "话题标签": "B"},
            {"开始行号": 21, "结束行号": 30, "话题标签": "C"},
        ]

    @staticmethod
    def _res(overrides: dict | None = None) -> dict:
        """构造一个 `是否合并=true` 的 pair result（字段可覆盖）。"""
        base = {
            "是否合并": True, "理由": "", "新话题标签": "",
            "事件": "", "是否重要": True, "困惑": False, "困惑原因": "",
        }
        if overrides:
            base.update(overrides)
        return base

    def test_no_decisions_keeps_all(self, segments: list[dict]) -> None:
        assert _apply_merge_decisions(segments, {}) == segments

    def test_all_true_collapses_to_one(self, segments: list[dict]) -> None:
        out = _apply_merge_decisions(
            segments, {(0, 1): self._res(), (1, 2): self._res()},
        )
        assert out == [{
            "开始行号": 1, "结束行号": 30, "话题标签": "A | B | C",
            "事件": "", "是否重要": True, "困惑": False, "困惑原因": "",
        }]

    def test_partial_merge(self, segments: list[dict]) -> None:
        out = _apply_merge_decisions(segments, {(0, 1): self._res()})
        assert len(out) == 2
        assert out[0] == {
            "开始行号": 1, "结束行号": 20, "话题标签": "A | B",
            "事件": "", "是否重要": True, "困惑": False, "困惑原因": "",
        }
        assert out[1] == segments[2]

    def test_second_pair_only(self, segments: list[dict]) -> None:
        out = _apply_merge_decisions(segments, {(1, 2): self._res()})
        assert [s["话题标签"] for s in out] == ["A", "B | C"]

    def test_boundaries_come_from_first_and_last(self, segments: list[dict]) -> None:
        out = _apply_merge_decisions(
            segments, {(0, 1): self._res(), (1, 2): self._res()},
        )
        assert out[0]["开始行号"] == 1 and out[0]["结束行号"] == 30

    def test_bridge_absorbs_unimportant_segment(self) -> None:
        """跨「不重要」段桥接：中间段被吸收（行号并入），标签只拼重要段。"""
        out = _apply_merge_decisions([
            {"开始行号": 1, "结束行号": 10, "话题标签": "A"},
            {"开始行号": 11, "结束行号": 15, "话题标签": "闲聊", "是否重要": False},
            {"开始行号": 16, "结束行号": 20, "话题标签": "B"},
        ], {(0, 2): self._res()})
        assert out == [{
            "开始行号": 1, "结束行号": 20, "话题标签": "A | B",
            "事件": "", "是否重要": True, "困惑": False, "困惑原因": "",
        }]

    def test_empty_segments(self) -> None:
        assert _apply_merge_decisions([], {(0, 1): self._res()}) == []

    def test_single_segment_untouched(self) -> None:
        seg = [{"开始行号": 1, "结束行号": 5, "话题标签": "A"}]
        assert _apply_merge_decisions(seg, {}) == seg

    def test_blank_labels_fall_back_to_placeholder(self) -> None:
        out = _apply_merge_decisions([
            {"开始行号": 1, "结束行号": 2, "话题标签": ""},
            {"开始行号": 3, "结束行号": 4, "话题标签": "  "},
        ], {(0, 1): self._res()})
        assert out[0]["话题标签"] == "未命名"

    def test_unmerged_segment_is_copied_not_aliased(self, segments: list[dict]) -> None:
        """未合并段返回的是 dict 拷贝，改返回值不应污染入参。"""
        out = _apply_merge_decisions(segments, {})
        out[0]["话题标签"] = "改了"
        assert segments[0]["话题标签"] == "A"

    def test_new_label_from_llm_used(self, segments: list[dict]) -> None:
        """合并段优先采信 LLM 的 `新话题标签`，不再拼旧标签。"""
        out = _apply_merge_decisions(
            segments, {(0, 1): self._res({"新话题标签": "银行长期持有"})},
        )
        assert out[0]["话题标签"] == "银行长期持有"

    def test_llm_fields_adopted(self, segments: list[dict]) -> None:
        """`事件`/`是否重要`/`困惑`/`困惑原因` 直接采信 LLM。"""
        out = _apply_merge_decisions(
            segments,
            {(0, 1): self._res({
                "事件": "某国加征关税", "是否重要": False,
                "困惑": True, "困惑原因": "指代不明",
            })},
        )
        assert out[0]["事件"] == "某国加征关税"
        assert out[0]["是否重要"] is False
        assert out[0]["困惑"] is True
        assert out[0]["困惑原因"] == "指代不明"

    def test_confusion_or_with_group_members(self) -> None:
        """组内任一段困惑，即便 LLM 标 false，合并段仍困惑（OR 保证进 3_2）。"""
        out = _apply_merge_decisions([
            {"开始行号": 1, "结束行号": 10, "话题标签": "A",
             "困惑": True, "困惑原因": "x"},
            {"开始行号": 11, "结束行号": 20, "话题标签": "B"},
        ], {(0, 1): self._res({"困惑": False})})
        assert out[0]["困惑"] is True
        assert out[0]["困惑原因"] == "x"


class TestClipSegmentsToScope:
    """3_2 分块：模型看到完整字幕，可能把话题段切到邻块去，必须裁回本块范围。"""

    def test_in_scope_segment_untouched(self) -> None:
        segs = [{"开始行号": 10, "结束行号": 20, "话题标签": "A"}]
        assert _clip_segments_to_scope(segs, 1, 100, "t") == segs

    def test_segment_fully_outside_dropped(self) -> None:
        segs = [{"开始行号": 200, "结束行号": 300, "话题标签": "越界"}]
        assert _clip_segments_to_scope(segs, 1, 100, "t") == []

    def test_segment_crossing_upper_bound_clipped(self) -> None:
        out = _clip_segments_to_scope(
            [{"开始行号": 90, "结束行号": 150, "话题标签": "A"}], 1, 100, "t",
        )
        assert out == [{"开始行号": 90, "结束行号": 100, "话题标签": "A"}]

    def test_segment_crossing_lower_bound_clipped(self) -> None:
        out = _clip_segments_to_scope(
            [{"开始行号": 50, "结束行号": 120, "话题标签": "A"}], 100, 200, "t",
        )
        assert out == [{"开始行号": 100, "结束行号": 120, "话题标签": "A"}]

    def test_label_preserved_when_clipping(self) -> None:
        out = _clip_segments_to_scope(
            [{"开始行号": 1, "结束行号": 999, "话题标签": "半导体"}], 5, 50, "t",
        )
        assert out[0]["话题标签"] == "半导体"


# ═══════════════════ 行号作用域：区间渲染与越界过滤 ═══════════════════

class TestFormatScope:
    """`[TASK]` 内行号清单的载荷：JSON 区间数组 `[{"start": int, "end": int}, ...]`。"""

    def test_consecutive_ids_merged(self) -> None:
        assert format_scope([1, 2, 3, 4, 5]) == [{"start": 1, "end": 5}]

    def test_gaps_split_into_multiple_ranges(self) -> None:
        assert format_scope([3, 4, 5, 8, 10, 11]) == [
            {"start": 3, "end": 5},
            {"start": 8, "end": 8},
            {"start": 10, "end": 11},
        ]

    def test_single_id_becomes_degenerate_range(self) -> None:
        """单行也写成 start == end 的区间，形态与多行一致。"""
        assert format_scope([42]) == [{"start": 42, "end": 42}]

    def test_empty_returns_empty_list(self) -> None:
        assert format_scope([]) == []

    def test_unsorted_and_duplicated_input_normalized(self) -> None:
        assert format_scope([5, 3, 4, 3, 5]) == [{"start": 3, "end": 5}]

    def test_json_serializable(self) -> None:
        """模板用 `| tojson` 渲染，载荷必须能直接 json.dumps。"""
        assert json.loads(json.dumps(format_scope([1, 2, 9]))) == [
            {"start": 1, "end": 2}, {"start": 9, "end": 9},
        ]


class TestFilterCitations:
    """整数数组形态的 `引用` 越界过滤（1_1 / 2_1 / 4_1 用）。"""

    def test_out_of_scope_row_dropped(self) -> None:
        obj = {"引用": [1, 2, 999]}
        assert filter_citations(obj, {1, 2}, "t") == 1
        assert obj["引用"] == [1, 2]

    def test_nested_citations_filtered(self) -> None:
        obj = {"投资机会": [{"短线": {"引用": [1, 500]}}]}
        filter_citations(obj, {1}, "t")
        assert obj["投资机会"][0]["短线"]["引用"] == [1]

    def test_all_in_scope_reports_zero_dropped(self) -> None:
        assert filter_citations({"引用": [1, 2]}, {1, 2, 3}, "t") == 0

    def test_non_int_elements_left_alone(self) -> None:
        """已转成时间区间的引用不是整数，不该被这条路径动。"""
        obj = {"引用": [{"开始时间": "00:00:01", "结束时间": "00:00:05"}]}
        filter_citations(obj, {1}, "t")
        assert obj["引用"] == [{"开始时间": "00:00:01", "结束时间": "00:00:05"}]

    def test_union_sibling_row_skipped_silently(self, caplog) -> None:
        """批并集内但属兄弟主题的引用：静默跳过，不打越界 WARNING。"""
        obj = {"引用": [1, 2]}
        with caplog.at_level(logging.WARNING):
            dropped = filter_citations(obj, {1}, "t", union={1, 2})
        assert dropped == 0
        assert obj["引用"] == [1]
        assert not caplog.records

    def test_union_out_of_scope_still_warns(self, caplog) -> None:
        """并集外的引用是真越界：丢弃并告警。"""
        obj = {"引用": [999]}
        with caplog.at_level(logging.WARNING):
            assert filter_citations(obj, {1}, "t", union={1, 2}) == 1
        assert obj["引用"] == []
        assert any("越界" in r.message for r in caplog.records)


class TestFilterRangeCitations:
    """区间形态的 `引用` 越界过滤（4_2 重写用）。"""

    def test_range_with_no_overlap_dropped(self) -> None:
        obj = {"引用": [{"开始行号": 500, "结束行号": 600}]}
        assert filter_range_citations(obj, {1, 2, 3}, "t") == 1
        assert obj["引用"] == []

    def test_range_clipped_to_intersection(self) -> None:
        obj = {"引用": [{"开始行号": 1, "结束行号": 100}]}
        filter_range_citations(obj, {5, 6, 7}, "t")
        assert obj["引用"] == [{"开始行号": 5, "结束行号": 7}]

    def test_extra_fields_preserved(self) -> None:
        obj = {"引用": [{"开始行号": 5, "结束行号": 6, "原文": "abc"}]}
        filter_range_citations(obj, {5, 6}, "t")
        assert obj["引用"][0]["原文"] == "abc"

    def test_empty_scope_is_noop(self) -> None:
        obj = {"引用": [{"开始行号": 1, "结束行号": 2}]}
        assert filter_range_citations(obj, set(), "t") == 0
        assert obj["引用"] == [{"开始行号": 1, "结束行号": 2}]


class TestFilterItemsByRowid:
    """`[{rowID: ...}]` 形态的条目过滤（1_1 修改列表 / 1_2 need_react行）。"""

    def test_out_of_scope_item_dropped(self) -> None:
        items = [{"rowID": 1, "x": "a"}, {"rowID": 999, "x": "b"}]
        assert filter_items_by_rowid(items, {1}, "t") == [{"rowID": 1, "x": "a"}]

    def test_non_numeric_rowid_dropped(self) -> None:
        assert filter_items_by_rowid([{"rowID": "abc"}], {1}, "t") == []

    def test_string_numeric_rowid_accepted(self) -> None:
        assert filter_items_by_rowid([{"rowID": "1"}], {1}, "t") == [{"rowID": "1"}]

    def test_union_sibling_row_skipped_silently(self, caplog) -> None:
        """批并集内但属兄弟单元的条目：静默跳过，不打越界 WARNING。"""
        items = [{"rowID": 1, "x": "a"}, {"rowID": 2, "x": "b"}]
        with caplog.at_level(logging.WARNING):
            kept = filter_items_by_rowid(items, {1}, "t", union={1, 2})
        assert kept == [{"rowID": 1, "x": "a"}]
        assert not caplog.records

    def test_union_out_of_scope_still_warns(self, caplog) -> None:
        """并集外的条目是真越界：丢弃并告警。"""
        items = [{"rowID": 999, "x": "b"}]
        with caplog.at_level(logging.WARNING):
            assert filter_items_by_rowid(items, {1}, "t", union={1, 2}) == []
        assert any("越界" in r.message for r in caplog.records)

    def test_union_none_keeps_legacy_behavior(self, caplog) -> None:
        """union=None：不在 allowed 的条目一律告警（与旧行为逐字节一致）。"""
        items = [{"rowID": 2, "x": "b"}]
        with caplog.at_level(logging.WARNING):
            assert filter_items_by_rowid(items, {1}, "t") == []
        assert any("越界" in r.message for r in caplog.records)


class TestFilterRowIds:
    """裸 rowID 数组过滤（2_1 无关/噪音行号）。"""

    def test_keeps_allowed_drops_out_of_scope(self) -> None:
        assert filter_row_ids([1, 2, 999], {1, 2}, "t") == [1, 2]

    def test_union_sibling_row_skipped_silently(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            kept = filter_row_ids([1, 2], {1}, "t", union={1, 2})
        assert kept == [1]
        assert not caplog.records

    def test_union_out_of_scope_still_warns(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            assert filter_row_ids([999], {1}, "t", union={1, 2}) == []
        assert any("越界" in r.message for r in caplog.records)

    def test_union_none_keeps_legacy_behavior(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            assert filter_row_ids([2], {1}, "t") == []
        assert any("越界" in r.message for r in caplog.records)


class TestRenderFullSubtitle:
    """全流水线 `[DATA]` 的唯一渲染入口——各阶段必须逐字节一致才能命中前缀缓存。"""

    def test_empty_text_rows_skipped(self) -> None:
        out = render_full_subtitle([
            {"rowID": 1, "speaker": 0, "text": "有内容"},
            {"rowID": 2, "speaker": 0, "text": ""},
            {"rowID": 3, "speaker": 0, "text": "也有内容"},
        ])
        assert out == "1\t有内容\n3\t也有内容"

    def test_deterministic_across_calls(self) -> None:
        sub = [{"rowID": 1, "speaker": 0, "text": "x"}]
        assert render_full_subtitle(sub) == render_full_subtitle(sub)


class TestRenderWindowedSubtitle:
    """窗口化 `[DATA]`：作用域由代码侧过滤表达，取代已删除的 `[ROWID]` 块。"""

    # rowID 故意缺号（空文本行被跳过后的真实形态）
    SUBS = [{"rowID": r, "text": f"line{r}"} for r in [1, 2, 3, 10, 11, 12, 50, 51]]

    def test_window_centered_on_scope(self) -> None:
        out = render_windowed_subtitle(self.SUBS, [11], pad=1)
        assert out == "10\tline10\n11\tline11\n12\tline12"

    def test_pad_counts_list_positions_not_rowid_arithmetic(self) -> None:
        """pad 必须按列表位置扩展：rowID 有缺号，`rid ± pad` 会少给行。

        scope=10、pad=1 时，数值口径下 9/11 只有 11 存在（给 2 行），
        位置口径给出前后各一行 → 3/10/11。
        """
        out = render_windowed_subtitle(self.SUBS, [10], pad=1)
        assert out == "3\tline3\n10\tline10\n11\tline11"

    def test_disjoint_windows_get_elision_marker(self) -> None:
        """断裂处必须显式标记——窗口化打破了「相邻 rowID 时间连续」的前提。

        计数口径是**被省略的字幕行数**（此处 rowID 2/3/10/11/12 共 5 行），
        而非 rowID 数值跨度（50-1）——rowID 有缺号，跨度会虚报。
        """
        out = render_windowed_subtitle(self.SUBS, [1, 50], pad=0)
        assert out == "1\tline1\n……（此处省略 5 行）\n50\tline50"

    def test_adjacent_windows_merge_without_marker(self) -> None:
        out = render_windowed_subtitle(self.SUBS, [1, 2, 3], pad=0)
        assert "省略" not in out
        assert out == "1\tline1\n2\tline2\n3\tline3"

    def test_empty_scope_degrades_to_full(self) -> None:
        """空作用域绝不能产出空 [DATA]。"""
        assert render_windowed_subtitle(self.SUBS, [], pad=1) == render_full_subtitle(self.SUBS)

    def test_nonexistent_scope_degrades_to_full(self) -> None:
        assert render_windowed_subtitle(self.SUBS, [9999], pad=1) == render_full_subtitle(self.SUBS)

    def test_extra_span_ids_included_beyond_pad(self) -> None:
        """1_2 桥接判定要读两段之间的完整间隔（上限 50 行 > pad）。"""
        out = render_windowed_subtitle(self.SUBS, [1], pad=0, extra_span_ids=[50, 51])
        assert out.endswith("50\tline50\n51\tline51")
        assert "1\tline1" in out

    def test_renders_materialized_inline_text(self) -> None:
        """渲染读 `text_llm`（第 2 步物化的指代内联版），不在渲染期重算。

        旧实现是 `inline=True` 开关、每次渲染重跑一遍替换，且只在步骤 4 起生效。
        物化后第 3 步也能看到已消解的指代——3_2 话题重置判断"这段在讲什么"要靠它。
        """
        from src.llm_summarize.utils.llm_text_utils import (
            build_inlined_text,
            materialize_llm_text,
        )

        subs = [{
            "rowID": 1, "text": "那个东西涨了",
            "annotation": {"指代": [{"原先的词": "那个东西", "指代": "认沽期权"}]},
        }]
        assert build_inlined_text(subs[0]) == "那个东西（认沽期权）涨了"

        baked = materialize_llm_text(subs, [
            {"rowID": 1, "原先的词": "那个东西", "指代": "认沽期权"},
        ])
        assert baked[0]["text_llm"] == "那个东西（认沽期权）涨了"
        assert baked[0]["text"] == "那个东西涨了", "`text` 不得被内联污染（要写进 SRT）"
        assert render_windowed_subtitle(baked, [1], pad=0) == (
            "1\t那个东西（认沽期权）涨了"
        )

    def test_falls_back_to_text_before_materialization(self) -> None:
        """`text_llm` 缺席时回退 `text`——步骤 1-2 自身还没有物化字段。"""
        assert render_windowed_subtitle(
            [{"rowID": 1, "text": "原始文本"}], [1], pad=0,
        ) == "1\t原始文本"

    def test_deterministic_across_calls(self) -> None:
        """同一阶段的并行调用必须逐字节一致，否则前缀缓存失效。"""
        a = render_windowed_subtitle(self.SUBS, [11], pad=N_REDUNDANT_ROWS)
        b = render_windowed_subtitle(self.SUBS, [11], pad=N_REDUNDANT_ROWS)
        assert a == b


# ═══════════════════ 引用区间与主导性检测 ═══════════════════

class TestSegmentLabelMatchesTopic:
    """判断锚点所在段"是否真的在聊本主题"。判错 → 一笔带过被放大成整段引用。"""

    def test_alias_substring_matches(self) -> None:
        assert _segment_label_matches_topic({"话题标签": "聊聊半导体行情"}, {"半导体"})

    def test_unrelated_label_does_not_match(self) -> None:
        assert not _segment_label_matches_topic({"话题标签": "白酒板块"}, {"半导体"})

    def test_empty_aliases_is_conservative_true(self) -> None:
        """别名集为空时不过滤——宁可保留也不误伤（全面性优先）。"""
        assert _segment_label_matches_topic({"话题标签": "白酒"}, set())

    def test_missing_label_is_conservative_true(self) -> None:
        assert _segment_label_matches_topic({}, {"半导体"})

    def test_any_one_alias_suffices(self) -> None:
        assert _segment_label_matches_topic({"话题标签": "宁王产业链"}, {"宁德时代", "宁王"})


class TestAnchorCoercion:
    def test_ints_and_digit_strings_kept(self) -> None:
        assert _coerce_int_anchors([1, "2", 3]) == [1, 2, 3]

    def test_non_numeric_dropped(self) -> None:
        assert _coerce_int_anchors([1, None, "abc"]) == [1]

    def test_float_truncated(self) -> None:
        assert _coerce_int_anchors([3.7]) == [3]

    def test_singletons_are_point_ranges(self) -> None:
        assert _anchors_to_singletons([1, "2"]) == [
            {"开始行号": 1, "结束行号": 1},
            {"开始行号": 2, "结束行号": 2},
        ]

    def test_singletons_skip_junk(self) -> None:
        assert _anchors_to_singletons(["x"]) == []


class TestMergeOverlappingRanges:
    def test_true_overlap_merged(self) -> None:
        out = _merge_overlapping_ranges([
            {"开始行号": 1, "结束行号": 5}, {"开始行号": 4, "结束行号": 8},
        ])
        assert out == [{"开始行号": 1, "结束行号": 8}]

    def test_adjacent_ranges_NOT_merged(self) -> None:
        """相邻但不重叠的段之间是 3_3 判定过的话题边界，禁止跨界合并——
        合了就把两个不同话题的引用粘成一段（破坏主题聚合）。"""
        out = _merge_overlapping_ranges([
            {"开始行号": 1, "结束行号": 5}, {"开始行号": 6, "结束行号": 8},
        ])
        assert len(out) == 2

    def test_unsorted_input_sorted_first(self) -> None:
        out = _merge_overlapping_ranges([
            {"开始行号": 10, "结束行号": 12}, {"开始行号": 1, "结束行号": 3},
        ])
        assert out[0]["开始行号"] == 1

    def test_reversed_bounds_swapped(self) -> None:
        assert _merge_overlapping_ranges([{"开始行号": 8, "结束行号": 3}]) == [
            {"开始行号": 3, "结束行号": 8},
        ]

    def test_contained_range_absorbed(self) -> None:
        out = _merge_overlapping_ranges([
            {"开始行号": 1, "结束行号": 20}, {"开始行号": 5, "结束行号": 8},
        ])
        assert out == [{"开始行号": 1, "结束行号": 20}]

    def test_junk_entries_dropped(self) -> None:
        assert _merge_overlapping_ranges([
            {"开始行号": "a", "结束行号": 1}, "notadict", {},
        ]) == []

    def test_empty_input(self) -> None:
        assert _merge_overlapping_ranges([]) == []


class TestRangesForAnchors:
    """主导性检测的总入口：标签命中 → 整段引用；未命中 → 退化为单点。"""

    @pytest.fixture
    def segments(self) -> list[dict]:
        return [
            {"开始行号": 1, "结束行号": 10, "话题标签": "半导体行情"},
            {"开始行号": 11, "结束行号": 20, "话题标签": "白酒"},
        ]

    def test_dominant_anchor_expands_to_full_segment(self, segments: list[dict]) -> None:
        out = _ranges_for_anchors([3], segments, "半导体", {"半导体"})
        assert out == [{"开始行号": 1, "结束行号": 10}]

    def test_non_dominant_anchor_degrades_to_singleton(self, segments: list[dict]) -> None:
        """锚点落在"白酒"段里 → 半导体只是被一笔带过，不能吃下整段。"""
        out = _ranges_for_anchors([15], segments, "半导体", {"半导体"})
        assert out == [{"开始行号": 15, "结束行号": 15}]

    def test_orphan_anchor_falls_back_to_singleton(self, segments: list[dict]) -> None:
        out = _ranges_for_anchors([99], segments, "半导体", {"半导体"})
        assert out == [{"开始行号": 99, "结束行号": 99}]

    def test_multiple_anchors_in_same_segment_dedup(self, segments: list[dict]) -> None:
        out = _ranges_for_anchors([2, 3, 4], segments, "半导体", {"半导体"})
        assert out == [{"开始行号": 1, "结束行号": 10}]

    def test_mixed_dominant_and_singleton(self, segments: list[dict]) -> None:
        out = _ranges_for_anchors([3, 15], segments, "半导体", {"半导体"})
        assert {"开始行号": 1, "结束行号": 10} in out
        assert {"开始行号": 15, "结束行号": 15} in out

    def test_empty_anchors(self, segments: list[dict]) -> None:
        assert _ranges_for_anchors([], segments, "半导体", {"半导体"}) == []

    def test_no_aliases_keeps_whole_segment(self, segments: list[dict]) -> None:
        """aliases 为 None → 不做主导性过滤，整段保留。"""
        out = _ranges_for_anchors([15], segments, "半导体", None)
        assert out == [{"开始行号": 11, "结束行号": 20}]


# ═══════════════════ 敏感信号：正则粗筛 ═══════════════════

def _first_signal(text: str) -> str | None:
    """按表顺序返回首个命中的信号类型，模拟 detect_and_annotate 的粗筛。"""
    for pattern, label in _SENSITIVITY_PATTERNS:
        if pattern.search(text):
            return label
    return None


class TestSensitivityPatterns:
    """主讲人"这个下次会员课再讲"这类话是高价值信号（暗示有未公开观点）。
    粗筛漏了 → 信号整条丢失；这层宽松召回，LLM 在第二层做语义复核。"""

    @pytest.mark.parametrize(("text", "expected"), [
        ("这个下次会员课再讲", "下次会员课预告"),
        ("下次会员群里讲", "下次会员课预告"),
        ("咱们下期再聊这个", "下次再讲"),
        ("留到下次讲", "下次再讲"),
        ("私域里可以讲", "私域可讲"),
        ("公域里讲不了", "公域讲不了"),
        ("这个具体的公司名字讲不了", "具体标的不便明说"),
        ("在这儿说不了", "公域讲不了"),
        ("咱们讲不了", "公域讲不了"),
        ("这能说吗", "暗示敏感"),
        ("懂的都懂", "暗示敏感"),
        ("不便明说", "暗示敏感"),
    ])
    def test_signal_recognized(self, text: str, expected: str) -> None:
        assert _first_signal(text) == expected

    @pytest.mark.parametrize("text", ["私欲里说", "思域里聊", "死域里讲"])
    def test_asr_homophone_variants_of_私域_recognized(self, text: str) -> None:
        """ASR 常把"私域"识别成"私欲/思域/死域"——变体不覆盖就等于信号丢失。"""
        assert _first_signal(text) == "私域可讲"

    @pytest.mark.parametrize("text", ["攻域讲不了", "共域讲不了"])
    def test_asr_variants_of_公域_recognized(self, text: str) -> None:
        assert _first_signal(text) == "公域讲不了"

    @pytest.mark.parametrize("text", [
        "今天大盘涨了", "会员很多", "私域", "下次", "这个公司很好",
    ])
    def test_ordinary_speech_not_flagged(self, text: str) -> None:
        """"会员"、"私域"单独出现是高频普通词，不能当信号——否则满屏误报。"""
        assert _first_signal(text) is None


# ═══════════════════ 敏感信号：锚定与聚合 ═══════════════════

class TestFindAnchoredTopics:
    """信号要挂到 rowID 邻近的主题上（回看 10 行 / 前看 5 行）。"""

    def test_nearby_topic_matched(self) -> None:
        kd = {"近": {"引用": [{"开始行号": 48, "结束行号": 52}]}}
        assert _find_anchored_topics(kd, 50) == ["近"]

    def test_far_topic_excluded(self) -> None:
        """引用区间 30–35 距信号行 50 已超出回看窗口。"""
        kd = {"远": {"引用": [{"开始行号": 30, "结束行号": 35}]}}
        assert _find_anchored_topics(kd, 50) == []

    def test_sorted_by_distance(self) -> None:
        kd = {
            "远些": {"引用": [{"开始行号": 41, "结束行号": 42}]},
            "最近": {"引用": [{"开始行号": 49, "结束行号": 51}]},
        }
        assert _find_anchored_topics(kd, 50) == ["最近", "远些"]

    def test_reversed_range_normalized(self) -> None:
        kd = {"倒": {"引用": [{"开始行号": 55, "结束行号": 45}]}}
        assert _find_anchored_topics(kd, 50) == ["倒"]

    def test_non_list_citations_skipped(self) -> None:
        assert _find_anchored_topics({"坏": {"引用": "notalist"}}, 50) == []

    def test_non_dict_topic_skipped(self) -> None:
        assert _find_anchored_topics({"坏": "notadict"}, 50) == []

    def test_missing_bounds_skipped(self) -> None:
        assert _find_anchored_topics({"缺": {"引用": [{"开始行号": 50}]}}, 50) == []

    def test_empty_keyword_data(self) -> None:
        assert _find_anchored_topics({}, 50) == []


class TestQuoteAndHits:
    @pytest.fixture
    def rowid_to_text(self) -> dict:
        return {i: f"句{i}" for i in range(1, 30)}

    def test_quote_spans_window_around_signal(self, rowid_to_text: dict) -> None:
        out = _build_quote(rowid_to_text, sorted(rowid_to_text), 10)
        assert out.startswith("句2")
        assert out.endswith("句13")

    def test_quote_truncated_with_ellipsis(self) -> None:
        long_map = {i: "字" * 60 for i in range(1, 20)}
        out = _build_quote(long_map, sorted(long_map), 10)
        assert out.endswith("…")
        assert len(out) == 241

    def test_quote_skips_blank_lines(self) -> None:
        out = _build_quote({9: "", 10: "有内容", 11: "  "}, [9, 10, 11], 10)
        assert out == "有内容"

    def test_hit_rowids_only_existing_lines(self) -> None:
        """窗口是 [signal-8, signal+3]；只回落在窗口内且字幕里真实存在的 rowID。"""
        assert _collect_hit_rowids([1, 5, 10, 50], 10) == [5, 10]

    def test_hits_empty_when_nothing_in_window(self) -> None:
        assert _collect_hit_rowids([100, 200], 10) == []


class TestSafeIntAndClustering:
    def test_safe_int_parses(self) -> None:
        assert _safe_int("5") == 5

    def test_safe_int_none_and_junk(self) -> None:
        assert _safe_int(None) is None
        assert _safe_int("abc") is None

    def test_cluster_keeps_first_of_each_run(self) -> None:
        """间距 <= 15 视为同一次表述，只留首条，避免同一句话报三遍。"""
        out = _cluster_signals([{"rowID": 10}, {"rowID": 20}, {"rowID": 100}])
        assert [s["rowID"] for s in out] == [10, 100]

    def test_cluster_unsorted_input(self) -> None:
        out = _cluster_signals([{"rowID": 100}, {"rowID": 10}])
        assert [s["rowID"] for s in out] == [10, 100]

    def test_cluster_empty(self) -> None:
        assert _cluster_signals([]) == []

    def test_cluster_single(self) -> None:
        assert _cluster_signals([{"rowID": 7}]) == [{"rowID": 7}]


class TestSensitivityAggregator:
    """同一信号会被多个主题各挂一份，按 锚定行号 去重、把源主题汇总到 板块。"""

    @pytest.fixture
    def keyword_data(self) -> dict:
        hint = {
            "锚定行号": 2, "信号类型": "私域可讲", "原话": "去私域看",
            "原话总结": "引流", "命中行号": [1, 2, 3], "LLM理由": "r",
        }
        return {
            "半导体": {
                "敏感提示": [hint],
                "引用": [{"开始时间": "00:00:00", "结束时间": "00:00:15"}],
            },
            "白酒": {
                "敏感提示": [dict(hint)],
                "引用": [{"开始时间": "00:00:00", "结束时间": "00:00:05"}],
            },
        }

    def test_same_anchor_deduped_into_one_record(
        self, keyword_data: dict, subtitle_l: list[dict],
    ) -> None:
        out = SensitivityAggregator.aggregate(keyword_data, subtitle_l)
        assert len(out["主讲人暗示重要话题"]) == 1

    def test_source_topics_collected_into_板块(
        self, keyword_data: dict, subtitle_l: list[dict],
    ) -> None:
        out = SensitivityAggregator.aggregate(keyword_data, subtitle_l)
        assert out["主讲人暗示重要话题"][0]["板块"] == ["半导体", "白酒"]

    def test_板块_sorted_by_citation_duration_desc(self, subtitle_l: list[dict]) -> None:
        """引用时长长的主题排前面——它更可能是这条信号真正指向的对象。"""
        hint = {"锚定行号": 2, "命中行号": [1, 2, 3]}
        kd = {
            "半导体": {
                "敏感提示": [hint],
                "引用": [{"开始时间": "00:00:00", "结束时间": "00:00:05"}],
            },
            "白酒": {
                "敏感提示": [dict(hint)],
                "引用": [{"开始时间": "00:00:00", "结束时间": "00:00:15"}],
            },
        }
        out = SensitivityAggregator.aggregate(kd, subtitle_l)
        assert out["主讲人暗示重要话题"][0]["板块"] == ["白酒", "半导体"]

    def test_citation_built_from_hit_rowids(self, keyword_data: dict, subtitle_l: list[dict]) -> None:
        out = SensitivityAggregator.aggregate(keyword_data, subtitle_l)
        cit = out["主讲人暗示重要话题"][0]["引用"][0]
        assert cit["开始时间"] == "00:00:00"
        assert cit["结束时间"] == "00:00:15"
        assert cit["原文"] == "第一句第二句第三句"

    def test_no_hints_returns_empty(self, subtitle_l: list[dict]) -> None:
        out = SensitivityAggregator.aggregate({"半导体": {}}, subtitle_l)
        assert "主讲人暗示重要话题" not in out

    def test_fixed_sections_skipped_as_source(self, subtitle_l: list[dict]) -> None:
        kd = {"宏观分析": {"敏感提示": [{"锚定行号": 1, "命中行号": [1]}]}}
        out = SensitivityAggregator.aggregate(kd, subtitle_l)
        assert "主讲人暗示重要话题" not in out

    def test_missing_signal_type_defaults(self, subtitle_l: list[dict]) -> None:
        kd = {"半导体": {"敏感提示": [{"锚定行号": 1, "命中行号": [1]}]}}
        out = SensitivityAggregator.aggregate(kd, subtitle_l)
        assert out["主讲人暗示重要话题"][0]["信号类型"] == "敏感提示"

    def test_citation_none_when_no_valid_rowids(self, subtitle_l: list[dict]) -> None:
        kd = {"半导体": {"敏感提示": [{"锚定行号": 1, "命中行号": [999]}]}}
        out = SensitivityAggregator.aggregate(kd, subtitle_l)
        assert out["主讲人暗示重要话题"][0]["引用"] == []

    def test_non_dict_hint_skipped(self, subtitle_l: list[dict]) -> None:
        kd = {"半导体": {"敏感提示": ["不是字典"]}}
        out = SensitivityAggregator.aggregate(kd, subtitle_l)
        assert "主讲人暗示重要话题" not in out


# ═══════════════════ 校验器辅助 ═══════════════════

class TestValidatorHelpers:
    def test_sanitize_filename_replaces_all_illegal_chars(self) -> None:
        """主题名会直接当日志文件名，含 / : * 等会让写盘失败。"""
        assert _sanitize_filename('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"

    def test_sanitize_keeps_chinese(self) -> None:
        assert _sanitize_filename("半导体·行情") == "半导体·行情"

    def test_has_citations_direct(self) -> None:
        assert _has_citations({"a": {"引用": [1]}})

    def test_has_citations_empty_list_is_false(self) -> None:
        assert not _has_citations({"a": {"引用": []}})

    def test_has_citations_nested_in_list(self) -> None:
        assert _has_citations([{"x": {"引用": [1]}}])

    def test_has_citations_scalar_false(self) -> None:
        assert not _has_citations("s")
        assert not _has_citations({"a": 1})


class TestRewrittenItemShape:
    """引用校验重写产物的形状归一：无名条目绝不能进 analysis。

    真实事故（钱博士 2026.8.6 半导体）：模型把【待校验条目清单】的 `{类型, 条目}`
    包装原样复述进 `重写后条目`，写进 analysis 后该条目没有 `标题`；渲染层拿空串
    做双向子串回退匹配（`"" in 任意字符串` 恒真），把该 section 全部 17 条观点
    吸到这一行，名称列显示"无"。
    """

    _INV_TASK = {"kind": "投资机会", "label": "半导体 / 投资机会[1]"}

    def test_unwraps_type_item_envelope(self) -> None:
        wrapped = {"类型": "投资机会", "条目": {"标题": "光纤", "短线": {"观点": "看空 🐻📉"}}}
        out = CitationValidator._unwrap_rewritten(wrapped, self._INV_TASK)
        assert out == {"标题": "光纤", "短线": {"观点": "看空 🐻📉"}}

    def test_unwrap_preserves_normal_item(self) -> None:
        """正常条目不该被动：只有恰好 {类型, 条目} 两键才剥。"""
        normal = {"标题": "光纤", "类型": "板块", "短线": {}}
        assert CitationValidator._unwrap_rewritten(normal, self._INV_TASK) is normal

    def test_unwrap_ignores_non_dict_inner(self) -> None:
        weird = {"类型": "投资机会", "条目": "不是字典"}
        assert CitationValidator._unwrap_rewritten(weird, self._INV_TASK) is weird

    def test_shape_rejects_missing_title(self) -> None:
        assert not CitationValidator._rewritten_shape_ok({"短线": {}}, self._INV_TASK)

    def test_shape_rejects_blank_title(self) -> None:
        assert not CitationValidator._rewritten_shape_ok({"标题": "  "}, self._INV_TASK)

    def test_shape_accepts_valid_title(self) -> None:
        assert CitationValidator._rewritten_shape_ok({"标题": "光纤"}, self._INV_TASK)

    def test_shape_checks_opinion_name_key(self) -> None:
        task = {"kind": "观点", "label": "t"}
        assert not CitationValidator._rewritten_shape_ok({"观点": "x"}, task)
        assert CitationValidator._rewritten_shape_ok({"标的名称": "光纤"}, task)

    def test_shape_skips_unknown_kinds(self) -> None:
        """宏观/敏感类条目的名称字段各异，不强校验（避免误丢有效重写）。"""
        assert CitationValidator._rewritten_shape_ok(
            {"观点": "x"}, {"kind": "宏观分析", "label": "t"},
        )


# ═══════════════════ 时间工具 ═══════════════════

class TestTimeUtils:
    @pytest.mark.parametrize(("seconds", "expected"), [
        (0, "00:00:00"),
        (60, "00:01:00"),
        (3600, "01:00:00"),
        (3661.7, "01:01:02"),
        ("", "00:00:00"),
        ("125", "00:02:05"),
    ])
    def test_fmt_time(self, seconds: int | float | str, expected: str) -> None:
        assert fmt_time(seconds) == expected

    @pytest.mark.parametrize(("hms", "expected"), [
        ("00:00:05", 5),
        ("1:02:03", 3723),
        ("02:30", 150),          # MM:SS 简短格式
        (" 00:00:01 ", 1),       # 两端空白
        ("00:00:01.5", 1),       # 小数秒截断
        ("bad", 0),              # 无法解析回 0
    ])
    def test_hms_to_seconds(self, hms: str, expected: int) -> None:
        assert hms_to_seconds(hms) == expected

    def test_roundtrip(self) -> None:
        assert hms_to_seconds(fmt_time(3725)) == 3725


# ═══════════════════ 黑话增量加载 ═══════════════════

class TestLoadIncrementalJargon:
    """增量黑话来自本次会话 WSA 消歧的产出，路径是 base_dir 的**兄弟**目录。
    加载失败必须静默降级为空——不能因为一个可选文件把整轮分析带崩。"""

    def test_missing_file_returns_empty_maps(self, tmp_path: Path) -> None:
        base = tmp_path / "llm_analysis"
        base.mkdir()
        assert load_incremental_jargon(base) == ({}, {})

    def _write(self, tmp_path: Path, payload: dict | str) -> Path:
        inc = tmp_path / "incremental_database"
        inc.mkdir(exist_ok=True)
        path = inc / "jargon_incremental.json"
        path.write_text(
            payload if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        base = tmp_path / "llm_analysis"
        base.mkdir(exist_ok=True)
        return base

    def test_stock_and_sector_parsed(self, tmp_path: Path) -> None:
        base = self._write(tmp_path, {
            "个股黑话": {"宁王": {"正式名": "宁德时代"}},
            "板块黑话": {"光": {"正式板块名": "光通讯"}},
        })
        stock, sector = load_incremental_jargon(base)
        assert stock == {"宁王": ["宁德时代"]}
        assert sector == {"光": ["光通讯"]}

    def test_blank_formal_name_dropped(self, tmp_path: Path) -> None:
        base = self._write(tmp_path, {"个股黑话": {"坏": {"正式名": "   "}}})
        assert load_incremental_jargon(base) == ({}, {})

    def test_non_dict_info_skipped(self, tmp_path: Path) -> None:
        base = self._write(tmp_path, {"个股黑话": {"坏": "不是字典"}})
        assert load_incremental_jargon(base) == ({}, {})

    def test_missing_formal_key_skipped(self, tmp_path: Path) -> None:
        base = self._write(tmp_path, {"个股黑话": {"空": {}}})
        assert load_incremental_jargon(base) == ({}, {})

    def test_broken_json_degrades_silently(self, tmp_path: Path) -> None:
        base = self._write(tmp_path, "{ 不是合法 json")
        assert load_incremental_jargon(base) == ({}, {})

    def test_unrelated_keys_ignored(self, tmp_path: Path) -> None:
        base = self._write(tmp_path, {"未解析": ["老灯"]})
        assert load_incremental_jargon(base) == ({}, {})
