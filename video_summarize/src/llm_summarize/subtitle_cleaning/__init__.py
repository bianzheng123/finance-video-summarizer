"""步骤 2 词级三阶段（2_1 粗筛 / 2_2 词表判定 / 2_3 联网确认）。

原第四阶段「2_4 关键词与剪枝」已迁为第 3 步的 `3_1`（见 `step3_refine/`）：它面对的是
本步产出的已纠错字幕，与「话题重置 3_2」并行，属"重做判定"而非词级纠错。

一次判断「是黑话还是错字」：这两件事需要同一份词表与同一份上下文，拆成"先猜字面、
后解语义"两步会让前一步在信息不全时先猜，猜错方向就白烧大量联网搜索。

- `WordLevelStep`：编排器（预清洗 + 四阶段 + 字幕回写 + 注释生成）
- `WordLevelStages`：单阶段执行器（渲染 → 调用 → 语义校验）
- `HotwordResources` / `build_row_annotations`：拼音索引与词表资源、逐行注释组装
"""

from .common import HotwordResources, build_row_annotations
from .word_level_stages import WordLevelStages
from .word_level_step import WordLevelStep

__all__ = [
    "WordLevelStep",
    "WordLevelStages",
    "HotwordResources",
    "build_row_annotations",
]
