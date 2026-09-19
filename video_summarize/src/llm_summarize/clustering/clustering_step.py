"""步骤 4 聚类编排器。

关键词提取与黑话消歧已在第 3 步完成（逐行注释的 `关键词` 字段）；本步只做
聚类：读取第 3 步分组数组并构建主题层级结构 → 向量主题去重（4_1，embedding 粗筛
候选等价对，零 LLM）→ 关键词合并（4_2，候选对批量并行三态判定）→ 精确去重
（4_3，困惑对 + 上下文 + 联网）→ 候选粗筛（4_4，无字幕、世界知识逐对判定）→
候选精筛（4_5_1 存疑对联网、4_5_2 事件候选父零搜索）→ 关系去环（4_6，SCC 环
检测 + 联网 1 轮三态判定）→ 祖先父生成（4_7，根节点 >10 时数据库拟父 + 建缺席桶补图）→
候选剪枝（4_8，共现率选唯一父，纯 Python）→ 两层压缩（4_9，挂载 + 辅助桶折叠，纯 Python）→
引用扩展递归段并集（4_10）→ 行号转时间。

最终产出 `引用` 已扩展为时间区间的 keyword_data。不同阶段**串行**，阶段内
（4_2 批 / 4_3 / 4_4 / 4_5_1 / 4_5_2 / 4_6）按工作单元**批量并行**。

断点（存在即跳过）：
- `4_3_keyword_extraction.json`：构建主题层级结构 + 4_1 + 4_2 + 4_3 完成（命中则跳过整个聚合阶段）；
- `4_1_embedding_dedup.json`（「结果」= keyword_data +「候选对」+「threshold」）、
  `4_2_topic_merge.json`（「结果」= 4_2 合并后 keyword_data +「困惑对」）、
  `4_3_precise_dedup.json`（「结果」= 4_3 后 keyword_data）、
  `4_4_coarse_candidates.json`（「结果」= 4_4 粗筛候选 + 存疑）、
  `4_5_1_doubt_refined.json`（「结果」= 4_5_1 存疑精筛候选）、
  `4_5_2_event_refined.json`（「结果」= 4_5_2 事件精筛候选）、
  `4_6_cycle_resolved.json`（「结果」= 4_6 去环后候选）、
  `4_7_ancestor_generation.json`（「结果」= 4_7 祖先父生成后候选）；
- `4_topic_consolidated.json`：引用扩展与时间映射完成（命中则整步跳过）。
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ...log_config import step_logger
from ...llm_infra.io import read_json, write_integrated
from ...llm_infra.config import get_embedding_recall_threshold
from ..schemas import SubtitleRow, TopicData, TopicSegment
from ..utils.row_id_2_time_range import (
    convert_row_range_to_time,
    convert_subtitle_l2rawID2subtitle_m,
)
from ..step3_refine.topic_aggregator import (
    aggregate_topics_by_type,
    build_keyword_data_from_grouped,
)
from .ancestor_parent_generator import AncestorParentGenerator
from .citation_expander import CitationExpander
from .coarse_parent_filter import (
    CoarseParentFilter,
    precompute_event_candidates,
    recall_candidate_parents_by_embedding,
)
from .cycle_resolver import CycleResolver
from .parent_ranker import ParentRanker
from .precise_dedup import PreciseDeduper
from .refined_parent_filter import DoubtParentFilter, EventParentFilter
from .topic_dedup import TopicDeduper

logger = logging.getLogger(__name__)


class ClusteringStep:
    """步骤 4 编排器：注释聚合去重 → 候选筛选 → 候选剪枝 + 两层压缩 + 引用扩展递归段并集。"""

    def __init__(self) -> None:
        self._deduper = TopicDeduper()
        self._precise_deduper = PreciseDeduper()
        self._coarse_filter = CoarseParentFilter()
        self._doubt_filter = DoubtParentFilter()
        self._event_filter = EventParentFilter()
        self._cycle_resolver = CycleResolver()
        self._ancestor_gen = AncestorParentGenerator()
        self._ranker = ParentRanker()
        self._citation_expander = CitationExpander()

    def run(
        self,
        subtitle_l: list[SubtitleRow],
        segments: list[TopicSegment],
        base_dir: Path | None = None,
    ) -> dict[str, TopicData]:
        """主入口。返回 `引用` 已扩展为时间区间的 keyword_data。"""
        with step_logger("第4步-关键词召回"):
            keyword_data = self._extract_keywords(subtitle_l, segments, base_dir)
        with step_logger("第4步-候选生成"):
            keyword_data, candidates = self._generate_candidates(
                keyword_data, subtitle_l, base_dir, segments,
            )
        return self._resolve_references(
            keyword_data, candidates, segments, subtitle_l, base_dir,
        )

    # ============ 构建主题层级结构 + 4_1 分块去重 + 4_2 合并去重 + 4_3 精确去重 ============

    def _extract_keywords(
        self,
        subtitle_l: list[SubtitleRow],
        segments: list[TopicSegment],
        base_dir: Path | None,
    ) -> dict[str, TopicData]:
        """读取第 3 步分组数组 → 构建主题层级结构初始态 → 4_1 向量主题去重（embedding 粗筛候选对，零 LLM）→
        4_2 关键词合并（候选对批量并行三态判定）→ 4_3 精确去重（困惑对 + 上下文 + 联网）→ 过滤空引用 → 落盘。"""
        if base_dir is not None:
            keyword_path = base_dir / "4_3_keyword_extraction.json"
            if keyword_path.exists():
                data = read_json(keyword_path)
                if isinstance(data, dict):
                    result = data.get("结果")
                    if isinstance(result, dict):
                        logger.info("检测到已有 4_3_keyword_extraction.json，跳过整个聚合阶段")
                        result.pop("rank", None)
                        return {
                            n: TopicData.model_validate(td)
                            for n, td in result.items()
                        }

        grouped = self._load_grouped_topics(subtitle_l, segments, base_dir)
        keyword_data = build_keyword_data_from_grouped(grouped, segments)
        if not keyword_data:
            logger.warning("注释中无有效关键词，返回空关键词")
            return {}

        logger.info("从分组数组构建主题层级结构 %d 个主题，开始主题去重", len(keyword_data))

        uncertain_pairs: list[dict] = []

        # 4_1 向量主题去重：embedding + 余弦相似度粗筛候选等价对（零 LLM）。
        candidate_pairs: list[dict] = []
        with step_logger("第4_1步-向量主题去重"):
            dedup_path = base_dir / "4_1_embedding_dedup.json" if base_dir is not None else None
            cached_kw: dict[str, TopicData] | None = None
            if dedup_path is not None and dedup_path.exists():
                cached = read_json(dedup_path)
                if isinstance(cached, dict):
                    if isinstance(cached.get("结果"), dict):
                        cached_kw = {
                            n: TopicData.model_validate(td)
                            for n, td in cached["结果"].items()
                        }
                    pairs = cached.get("候选对")
                    if isinstance(pairs, list):
                        candidate_pairs = pairs
            if cached_kw is not None:
                logger.info("检测到 4_1_embedding_dedup.json，跳过向量主题去重")
                keyword_data = cached_kw
            else:
                keyword_data, candidate_pairs = self._deduper.dedup_topics_vector(
                    keyword_data, base_dir,
                )

        # 4_2 关键词合并：对候选等价对批量并行三态判定（等价合并 / 困惑导出 / 不等价丢弃）。
        merge_pairs_2: list[dict] = []
        with step_logger("第4_2步-关键词合并"):
            merge_path = base_dir / "4_2_topic_merge.json" if base_dir is not None else None
            cached_kw: dict[str, TopicData] | None = None
            if merge_path is not None and merge_path.exists():
                cached = read_json(merge_path)
                if isinstance(cached, dict):
                    if isinstance(cached.get("结果"), dict):
                        cached_kw = {
                            n: TopicData.model_validate(td)
                            for n, td in cached["结果"].items()
                        }
                    uncertain = cached.get("困惑对")
                    if isinstance(uncertain, list):
                        uncertain_pairs.extend(uncertain)
                    mp = cached.get("合并对")
                    if isinstance(mp, list):
                        merge_pairs_2 = mp
            if cached_kw is not None:
                logger.info("检测到 4_2_topic_merge.json，跳过关键词合并")
                keyword_data = cached_kw
            else:
                keyword_data, uncertain_2, merge_pairs_2 = self._deduper.dedup_topics_merge(
                    candidate_pairs, keyword_data, base_dir,
                )
                uncertain_pairs.extend(uncertain_2)

        # 4_2 各批可能重复导出同一困惑对，去重后再交 4_3。
        uncertain_pairs = self._dedup_uncertain_pairs(uncertain_pairs)

        # 4_3 精确去重：不确定对 + 字幕上下文 + 联网搜索（最多 3 轮）。
        with step_logger("第4_3步-精确去重"):
            keyword_data, merge_pairs_3 = self._precise_deduper.dedup_precise(
                uncertain_pairs, keyword_data, subtitle_l, base_dir,
            )

        # 过滤空引用主题：没有 rowID 锚点的主题对下游无意义
        before = len(keyword_data)
        keyword_data = {
            topic: data for topic, data in keyword_data.items()
            if isinstance(data, TopicData) and data.引用
        }
        removed = before - len(keyword_data)
        if removed > 0:
            logger.info("过滤空引用主题 %d 个，剩余 %d 个", removed, len(keyword_data))

        if base_dir is not None:
            self._save_keyword_results(
                base_dir, keyword_data, merge_pairs_2, merge_pairs_3,
            )

        return keyword_data

    @staticmethod
    def _load_grouped_topics(
        subtitle_l: list[SubtitleRow],
        segments: list[TopicSegment],
        base_dir: Path | None,
    ) -> dict[str, list[dict]]:
        """读取第 3 步落盘的分组数组；缺失或 base_dir=None 时兜底内存聚合。"""
        if base_dir is not None:
            agg_path = base_dir / "3_aggregated_topics.json"
            if agg_path.exists():
                data = read_json(agg_path)
                if isinstance(data, dict):
                    logger.info("检测到 3_aggregated_topics.json，读取分组数组")
                    return data
        logger.info("未找到 3_aggregated_topics.json，兜底内存聚合")
        return aggregate_topics_by_type(subtitle_l, segments)

    @staticmethod
    def _save_keyword_results(
        base_dir: Path,
        keyword_data: dict[str, TopicData],
        merge_pairs_2: list[dict],
        merge_pairs_3: list[dict],
    ) -> None:
        """保存聚合阶段最终结果到 4_3_keyword_extraction.json。

        结构：`{"结果": keyword_data, "合并溯源": {"4_2": [...], "4_3": [...]}}`；
        合并溯源每条为 `{"被合并名": ..., "保留名": ...}`。
        """
        try:
            keyword_path = base_dir / "4_3_keyword_extraction.json"
            payload = {
                "结果": {n: td.model_dump() for n, td in keyword_data.items()},
                "合并溯源": {
                    "4_2": merge_pairs_2,
                    "4_3": merge_pairs_3,
                },
            }
            with open(keyword_path, "w", encoding="utf-8") as f:
                json.dump(payload, ensure_ascii=False, indent=2, fp=f)
            logger.info("已保存到: %s", keyword_path)
        except Exception as e:
            logger.warning("保存结果失败: %s", e)

    @staticmethod
    def _dedup_uncertain_pairs(pairs: list[dict]) -> list[dict]:
        """对困惑对按 (关键词1, 关键词2) 去重，保持首次出现顺序。"""
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for p in pairs:
            if not isinstance(p, dict):
                continue
            key = ((p.get("关键词1") or "").strip(), (p.get("关键词2") or "").strip())
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(p)
        return deduped

    # ==================== 4_4 候选粗筛 + 4_5 候选精筛 + 4_6 去环 + 4_7 祖先父生成 ====================

    def _generate_candidates(
        self,
        keyword_data: dict[str, TopicData],
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None,
        segments: list[TopicSegment],
    ) -> tuple[dict[str, TopicData], dict[str, list[str]]]:
        """4_4 候选粗筛 + 4_5 候选精筛 + 4_6 关系去环 + 4_7 祖先父生成。

        返回 (keyword_data, candidates)。断点续跑在各模块内处理。
        """

        self._snapshot(base_dir, "4_0_topics_raw.json", keyword_data)
        self._print_topics("初始", keyword_data)

        # 代码预筛：事件候选父（4_5 用，同段标注）+ 非事件候选父（4_4 用，embedding range 召回）
        event_candidates = precompute_event_candidates(keyword_data, segments)
        non_event_candidates = recall_candidate_parents_by_embedding(
            keyword_data,
            threshold=get_embedding_recall_threshold(),
        )
        self._snapshot(base_dir, "4_4_embedding_recall.json", non_event_candidates)

        # 4_4 候选粗筛（无字幕、无事件，世界知识逐对判定）
        coarse_confirmed, doubt_pairs = self._coarse_filter.generate(
            keyword_data, segments, base_dir,
            non_event_candidates=non_event_candidates,
        )
        self._snapshot(base_dir, "4_4_parent_relation.json", build_candidate_relation(coarse_confirmed))

        # 4_5_1 存疑候选精筛（存疑对，联网）
        doubt_confirmed, doubt_final = self._doubt_filter.generate(
            keyword_data, doubt_pairs, subtitle_l, base_dir,
        )
        # 4_5_2 事件候选精筛（事件候选父，零搜索）
        event_confirmed, event_final = self._event_filter.generate(
            keyword_data, event_candidates, subtitle_l, base_dir,
        )
        # 合并两个子阶段的确认父与存疑
        refined_confirmed = self._merge_candidates(doubt_confirmed, event_confirmed)
        final_doubt = doubt_final + event_final
        if base_dir is not None:
            write_integrated(
                base_dir / "4_5_refined_candidates.json", {},
                {"候选父板块": refined_confirmed},
            )
            write_integrated(
                base_dir / "4_5_doubt_relations.json", {},
                {"存疑": final_doubt},
            )

        # 合并候选图：4_4 确认父 ∪ 4_5 确认父
        candidates = self._merge_candidates(coarse_confirmed, refined_confirmed)
        self._snapshot(base_dir, "4_5_parent_relation.json", build_candidate_relation(candidates))

        logger.info(
            "主题 %d 个，含候选的主题 %d 个",
            len(keyword_data), sum(1 for v in candidates.values() if v),
        )

        # 4_6 关系去环（SCC 环检测 + 联网 1 轮三态判定）
        keyword_data, candidates = self._cycle_resolver.resolve(
            keyword_data, candidates, subtitle_l, base_dir,
        )
        self._snapshot(base_dir, "4_6_parent_relation.json", build_candidate_relation(candidates))

        # 4_7 祖先父生成（画 DAG 选根节点，根节点 >10 时数据库拟父）
        candidates = self._ancestor_gen.generate(candidates, keyword_data, base_dir)
        self._snapshot(base_dir, "4_7_parent_relation.json", build_candidate_relation(candidates))

        return keyword_data, candidates

    @staticmethod
    def _merge_candidates(
        coarse: dict[str, list[str]],
        refined: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """合并 4_4 与 4_5 的确认候选父，得到完整候选图（子 → 候选父列表）。

        coarse 含所有主题（含空列表），refined 只含 4_5 确认的主题；
        以 coarse 为基底，把 refined 的确认父并入。
        """
        merged: dict[str, list[str]] = {
            topic: list(parents) if isinstance(parents, list) else []
            for topic, parents in coarse.items()
        }
        for topic, parents in refined.items():
            if not isinstance(parents, list):
                continue
            lst = merged.setdefault(topic, [])
            for p in parents:
                if isinstance(p, str) and p and p != topic and p not in lst:
                    lst.append(p)
        return merged

    # ============ 4_8 候选剪枝 + 4_9 两层压缩 + 4_10 引用扩展 ============

    def _resolve_references(
        self,
        keyword_data: dict[str, TopicData],
        candidates: dict[str, list[str]],
        segments: list[TopicSegment],
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None,
    ) -> dict[str, TopicData]:
        """4_8 候选剪枝 → 4_9 两层压缩 → 4_10 引用扩展+行号转时间。
        断点：4_topic_consolidated.json（存在即整段跳过）。"""
        if base_dir is not None:
            consolidated_path = base_dir / "4_topic_consolidated.json"
            if consolidated_path.exists():
                data = read_json(consolidated_path)
                if isinstance(data, dict):
                    logger.info("检测到 4_topic_consolidated.json，跳过引用扩展阶段")
                    return {
                        n: TopicData.model_validate(td)
                        for n, td in data.items()
                    }

        # 4_8 候选剪枝（共现率选唯一父）+ 4_9 两层压缩（挂载 + 辅助桶折叠）
        keyword_data = self._ranker.assign_and_merge(keyword_data, candidates, segments)
        self._snapshot(base_dir, "4_9_after_compress.json", keyword_data)
        self._snapshot(base_dir, "4_9_parent_relation.json", build_parent_relation(keyword_data))
        self._print_topics("4_9 两层压缩 后", keyword_data)

        # 4_10 引用扩展 + 行号转时间（根主题递归段并集 → 行号区间 → 时间区间）
        with step_logger("第4步-引用扩展"):
            keyword_data = self._citation_expander.expand(keyword_data, segments, subtitle_l)
            self._snapshot(base_dir, "4_after_expansion.json", keyword_data)
            self._print_topics("引用扩展 后", keyword_data)
            rowID2subtitle_m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
            keyword_data = self.add_time_ranges(keyword_data, rowID2subtitle_m)
        relation = build_parent_relation(keyword_data)
        for topic, td in keyword_data.items():
            if topic in relation:
                relation[topic]["引用"] = td.引用
        self._snapshot(base_dir, "4_10_parent_relation.json", relation)
        self._snapshot(base_dir, "4_10_summary_topics.json", build_summary_topics(keyword_data))
        self._snapshot(base_dir, "4_topic_consolidated.json", keyword_data)

        return keyword_data

    @staticmethod
    def add_time_ranges(
        keyword_data: dict[str, TopicData], rowID2subtitle_m: dict,
    ) -> dict[str, TopicData]:
        """把行号区间 [{开始行号, 结束行号}] 转换成时间区间。"""
        for topic_data in keyword_data.values():
            if isinstance(topic_data.引用, list):
                new_ranges = []
                for tr in topic_data.引用:
                    ranges = convert_row_range_to_time(tr, rowID2subtitle_m)
                    if ranges:
                        if isinstance(ranges, list):
                            new_ranges.extend(ranges)
                        else:
                            new_ranges.append(ranges)
                topic_data.引用 = new_ranges
        return keyword_data

    @staticmethod
    def _snapshot(base_dir: Path | None, filename: str, data: Any) -> None:
        if base_dir is None:
            return
        try:
            (base_dir / filename).write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=_dump_default),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("写 %s 失败：%s", filename, e)

    @staticmethod
    def _print_topics(phase: str, keyword_data: dict[str, TopicData]) -> None:
        nodes = list(keyword_data)
        summary = build_summary_topics(keyword_data)
        logger.info(
            "[%s] 共 %d 个节点，其中 %d 个主题（非叶子/段落事件）：%s",
            phase, len(nodes), len(summary),
            [t["主题名"] for t in summary],
        )


def build_candidate_relation(candidates: dict[str, list[str]]) -> dict[str, dict]:
    """从多父候选关系生成候选父关系（4_4 / 4_5）。

    Args:
        candidates: {主题名: [候选父名, ...]}（多父，未定单父）

    Returns:
        {主题名: {"候选父": [父名, ...]}}
    """
    relation: dict[str, dict] = {}
    for topic, parents in candidates.items():
        if not isinstance(parents, list):
            continue
        relation[topic] = {
            "候选父": [p for p in parents if isinstance(p, str)],
        }
    return relation


def _topic_depth(topic: str, child2parent: dict[str, str]) -> int:
    """节点到根的深度（根为 0），沿 child2parent 祖先链走，环兜底。"""
    chain = [topic]
    cur = topic
    visited = {topic}
    while cur in child2parent:
        cur = child2parent[cur]
        if cur in visited:
            break
        visited.add(cur)
        chain.append(cur)
    return len(chain) - 1


def build_parent_relation(keyword_data: dict[str, TopicData]) -> dict[str, dict]:
    """从 keyword_data 的「子项」反推确定树关系（父 + 子项 + 高度 + 类型）。

    Args:
        keyword_data: {主题名: TopicData}

    Returns:
        {主题名: {"父": 直接父名或 None, "子项": [子名, ...], "高度": 深度, "类型": 类型}}
    """
    child2parent: dict[str, str] = {}
    for topic, td in keyword_data.items():
        for child in td.子项 or []:
            if isinstance(child, str) and child in keyword_data:
                child2parent[child] = topic

    relation: dict[str, dict] = {}
    for topic, td in keyword_data.items():
        relation[topic] = {
            "父": child2parent.get(topic),
            "子项": [
                c for c in (td.子项 or [])
                if isinstance(c, str) and c in keyword_data
            ],
            "高度": _topic_depth(topic, child2parent),
            "类型": td.类型,
        }
    return relation


def build_summary_topics(keyword_data: dict[str, TopicData]) -> list[dict]:
    """两层压缩后生成「主题 + 子项」结构。

    主题 = 段落事件（恒主题）+ 有普通子项的非叶子节点。
    关键词事件不占子项位，从「子项」拆出后挂到所属节点的「关键词事件」列表。

    Args:
        keyword_data: {主题名: {"类型": ..., "引用": [...], "子项": [子名, ...]}}

    Returns:
        主题列表（按 高度, 主题名 排序），每项：
        {"主题名", "类型", "高度", "关键词事件": [str, ...],
         "子项": [{"名称", "类型", "关键词事件": [str, ...], "高度"}, ...]}
    """
    relation = build_parent_relation(keyword_data)

    def split_children(children: list) -> tuple[list[str], list[dict]]:
        """把子项列表二分为 (关键词事件名列表, 普通子项列表)。

        普通子项对象的「关键词事件」字段 = 该子项自己「子项」里的关键词事件。
        """
        keyword_events: list[str] = []
        regular: list[dict] = []
        for child in children:
            child_rel = relation.get(child) if isinstance(child, str) else None
            if not isinstance(child_rel, dict):
                continue
            if child_rel.get("类型") == "关键词事件":
                keyword_events.append(child)
                continue
            sub_events = [
                c for c in (child_rel.get("子项") or [])
                if isinstance(c, str)
                and isinstance(relation.get(c), dict)
                and relation[c].get("类型") == "关键词事件"
            ]
            regular.append({
                "名称": child,
                "类型": child_rel.get("类型"),
                "关键词事件": sub_events,
                "高度": child_rel.get("高度"),
            })
        return keyword_events, regular

    topics: list[dict] = []
    for topic, rel in relation.items():
        if not isinstance(rel, dict):
            continue
        keyword_events, children = split_children(rel.get("子项") or [])
        if rel.get("类型") == "段落事件" or children:
            topics.append({
                "主题名": topic,
                "类型": rel.get("类型"),
                "高度": rel.get("高度"),
                "关键词事件": keyword_events,
                "子项": children,
            })
    topics.sort(key=lambda t: (t.get("高度") or 0, t.get("主题名") or ""))
    return topics


def _dump_default(o: Any) -> Any:
    """json.dumps 的 default 钩子：pydantic 模型转 model_dump，其余退回 str。"""
    if isinstance(o, BaseModel):
        return o.model_dump()
    return str(o)
