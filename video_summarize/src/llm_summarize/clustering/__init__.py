"""步骤 4 聚类（构建主题层级结构 / 4_1 向量主题去重 / 4_2 关键词合并 / 4_3 精确去重 / 4_4 候选粗筛 / 4_5_1 存疑候选精筛 / 4_5_2 事件候选精筛 / 4_6 关系去环 / 4_7 祖先父生成 / 4_8 候选剪枝 / 4_9 两层压缩 / 4_10 引用扩展）。

`event_topics` 是「段落事件只能当父」这一不变量的共享判据（类型判定 / 事件区间 / 共现率打分）。
"""

from .topic_dedup import TopicDeduper
from .precise_dedup import PreciseDeduper
from .coarse_parent_filter import (
    CoarseParentFilter,
    precompute_event_candidates,
    recall_candidate_parents_by_embedding,
)
from .refined_parent_filter import DoubtParentFilter, EventParentFilter
from .cycle_resolver import CycleResolver
from .ancestor_parent_generator import AncestorParentGenerator
from .event_topics import (
    is_keyword_event,
    is_segment_event,
    resolve_event_ranges,
)
from .parent_ranker import ParentRanker
from .citation_expander import CitationExpander
from .clustering_step import ClusteringStep

__all__ = [
    "TopicDeduper",
    "PreciseDeduper",
    "CoarseParentFilter",
    "DoubtParentFilter",
    "EventParentFilter",
    "CycleResolver",
    "AncestorParentGenerator",
    "ParentRanker",
    "CitationExpander",
    "ClusteringStep",
    "precompute_event_candidates",
    "recall_candidate_parents_by_embedding",
    "is_keyword_event",
    "is_segment_event",
    "resolve_event_ranges",
]
