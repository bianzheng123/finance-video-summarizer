"""经济版 step2 板块分析：整场字幕 + [DATA] 前 + 分字段组分析 + 内联引用校验。

与宏观分析（经济版 step1）完全独立：无选段，直接对整场字幕按三个字段组
（板块情绪/板块事件/板块观点）分析，prompt 结构为 `[DATA]` 前。
"""

from .sector_step import SectorAnalysisStep

__all__ = ["SectorAnalysisStep"]
