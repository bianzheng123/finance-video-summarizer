"""步骤 5 宏观分析：5_1 选段 → 5_2 分字段组分析 → 5_3 内联引用校验。

渐进式披露取代「整场字幕一次性生成六个字段」，引用校验内联取代「步骤 6 事后校验」，
修复粒度由整字段缩到单条条目。详见 `macro_step.py` 与 `macro_analyzer.py`。
"""

from .macro_analyzer import MACRO_FIELD_GROUPS, MacroAnalyzer, MacroFieldGroup
from .macro_segment_selector import MacroSegmentSelector
from .macro_step import MacroAnalysisStep

__all__ = [
    "MACRO_FIELD_GROUPS",
    "MacroAnalysisStep",
    "MacroAnalyzer",
    "MacroFieldGroup",
    "MacroSegmentSelector",
]
