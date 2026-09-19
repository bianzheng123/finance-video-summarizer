"""字幕处理模块 - 七步管线：话题分割、词级三阶段、剪枝终判与话题重置、聚类、
宏观分析、交易技巧与敏感信号、主题分析。

第 2 步「词级三阶段」（2_1 粗筛 / 2_2 词表判定 / 2_3 联网确认）一次完成字面纠错：
「是黑话还是错字」本是同一个判断，需要同一份词表与同一份上下文；拆成两步会让前一步
在没有词表时先猜，猜错方向就白烧大量联网搜索。收尾把纠错写进 `text`、把已确认指代
物化进 `text_llm`，后者是第 3 步起所有阶段的 [DATA] 底稿。

第 3 步「剪枝终判与话题重置」的两个阶段**并行**：3_1 在已纠错字幕上做关键词提取与
逐行剪枝终判，3_2 对 1_1 自报「困惑」（没读懂）的话题段重新判定分割方式。两者改的
维度不同（逐行类型 vs 段边界）且互不依赖，故并行；详见 `step3_refine/refine_step.py`。

后半段的步骤 5（宏观）、6（交易技巧与敏感信号）、7（主题）**自带内联引用校验**：
生成 → 校验 → 不支撑当场重写或重跑修复 → 定稿，不跨步骤回滚。校验组件本身不是
步骤（`citation_check/`）。**步骤 6 的交易技巧不做引用校验**——它是权重最低的
section，只经确定性越界过滤；但步骤 6 产出的 `主讲人暗示重要话题` 照常经 6_3 校验。
"""

from .topic_segmentation import TopicSegmentationStep
from .subtitle_cleaning import WordLevelStep
from .step3_refine import RefineStep
from .clustering import ClusteringStep
from .step5_macro import MacroAnalysisStep
from .step6_trading import TradingSensitivityStep
from .step7_topic import TopicAnalysisStep

__all__ = [
    'TopicSegmentationStep',
    'WordLevelStep',
    'RefineStep',
    'ClusteringStep',
    'MacroAnalysisStep',
    'TradingSensitivityStep',
    'TopicAnalysisStep',
]
