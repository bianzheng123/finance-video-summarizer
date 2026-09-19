"""步骤 5/6/7 与引用校验共用的分析设施（不属于任何单一步骤）。

装的是「多个步骤都要用、但本身不构成一个阶段」的东西：

- `post_processor`：行号→时间、引用原文回填、空字段过滤（步骤 5/6/7 与引用校验共用）
- `cross_step_dedup`：跨步骤归属仲裁（仓位建议 × 交易技巧判重），只能在步骤 5/6/7
  三路 join 之后调——三步并行时彼此看不到对方产物
- `_jargon_utils`：黑话内联注入（步骤 5/7 共用）
- `_duration_utils`：视频/主题时长口径（步骤 5/6/7 的分级日志共用）
- `_repair_utils` + `_repair_block.jinja2`：修复模式的待修条目整形与模板宏（5 处调用共用）

`_repair_block.jinja2` 不在本包的 jinja2 搜索路径里时无法被 `{% from %}` 引用，
故各步骤构造 `Environment` 时必须把本包目录一并加进 `FileSystemLoader`
（见各步骤模块里的 `_jinja_env`）。
"""

from ._duration_utils import (
    calculate_topic_duration,
    calculate_video_total_duration,
)
from ._jargon_utils import (
    collect_used_jargons,
    inline_array_field,
    inline_single_field,
)
from ._repair_utils import build_repair_items
from .cross_step_dedup import dedup_position_advice_against_skills
from .post_processor import PostProcessor

__all__ = [
    "PostProcessor",
    "build_repair_items",
    "dedup_position_advice_against_skills",
    "calculate_topic_duration",
    "calculate_video_total_duration",
    "collect_used_jargons",
    "inline_array_field",
    "inline_single_field",
]
