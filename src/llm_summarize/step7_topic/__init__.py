"""步骤 7 主题分析：7_1 主题生成 → 7_2 内联引用校验 → 收尾落盘。

管线最后一环：只承担主题分析、跨 section 后处理与 `7_analysis_result.json` 落盘
（敏感信号判定/聚合已迁入步骤 6；最终 `llm_analysis.json` 由主流程合并写）。
"""

from .topic_analyzer import TopicAnalyzer
from .topic_step import TopicAnalysisStep

__all__ = [
    "TopicAnalysisStep",
    "TopicAnalyzer",
]
