"""第 3 步：剪枝与关键词终判（3_1）∥ 话题重置（3_2）。

两个阶段并行，都在**已纠错 + 指代已内联**的字幕上重做判定，但改的是不同维度：
3_1 改逐行类型与关键词（按 rowID 键控），3_2 改话题段的分割方式（按段区间键控）。
详见 `refine_step.py` 的模块 docstring。
"""

from .refine_stages import RefineStages
from .refine_step import RefineStep
from .topic_reset import TopicResetter, build_reset_units

__all__ = ["RefineStep", "RefineStages", "TopicResetter", "build_reset_units"]
