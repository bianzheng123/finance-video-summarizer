"""经济版 step1 宏观分析：整场字幕 + [DATA] 前 + 分字段组分析 + 内联引用校验。

与完整版 ``step5_macro`` 完全独立：无选段（经济版无 keyword_data/话题段），直接对
整场字幕按四个字段组（大盘情绪/宏观事件/宏观观点/仓位建议）分析，prompt 结构为
`[DATA]` 前（完整字幕成为 KV 缓存稳定前缀）。
"""

from .macro_step import MacroAnalysisStep

__all__ = ["MacroAnalysisStep"]
