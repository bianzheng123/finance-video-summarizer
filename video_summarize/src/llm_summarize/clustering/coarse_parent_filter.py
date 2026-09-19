"""4_4 候选粗筛：embedding top-k 召回候选父 + LLM 凭世界知识逐对判定（无字幕、无事件）。

与旧 4_4（带字幕上下文筛选）的区别：
- 不再输入字幕，LLM 只凭世界知识判断「父是否在范畴上包含子」（排除上下游/配套关系）；
- 不涉及事件：事件候选父拆到 4_5 候选精筛处理。

判定三态：作为父节点（进 `候选父板块`）/ 不作为父节点（不输出、代码丢弃）/ 存疑（进 4_5）。

`recall_candidate_parents_by_embedding` 产出 `non_event_candidates`（embedding top-k 召回，
供本阶段）；`precompute_event_candidates` 产出 `event_candidates`（同段标注事件，供 4_5）。
"""

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..utils.llm_text_utils import load_prompt
from ..schemas import TopicData, TopicSegment
from ...llm_infra.batch import round_robin_groups
from ...llm_infra.io import read_json, write_integrated
from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ...llm_infra.clients import EmbeddingClient
from ...llm_infra.config import (
    get_embedding_recall_threshold,
    get_group_count,
)
from ...llm_infra.resource import fanout
from ...log_config import step_logger
from .event_topics import is_segment_event
from .bucket_utils import is_non_industry_concept

logger = logging.getLogger(__name__)

# 4_4 候选粗筛的并行分组数（沿用旧 4_4 的历史常量）。
CANDIDATE_GROUP_COUNT = 5

# 非事件候选父类型白名单（同段共现，走 4_4 无字幕世界知识判定）：个股/个股黑话/
# 外汇/大宗商品等叶子类型不作父；段落事件走 4_5 事件候选父通道，不在此列。
_PARENTABLE_TYPES = {"板块", "板块黑话", "加密货币", "债券"}

# 子类型白名单（同段共现，能作子、参与候选父生成）：段落事件只能作父、不在其内；
# 债券/加密货币只作父不作子、也不在其内；个股黑话/板块黑话（未锁定黑话）不作子、
# 不在其内（避免被误挂到板块）；关键词事件只能作子，在列。
_CHILDABLE_TYPES = {
    "股票", "外汇", "板块", "大宗商品", "关键词事件",
}


def _cosine(a: list[float], b: list[float]) -> float:
    """两向量的余弦相似度（零向量返回 0.0，避免除零）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def recall_candidate_parents_by_embedding(
    keyword_data: dict[str, TopicData],
    threshold: float,
) -> dict[str, list[str]]:
    """对所有 childable 主题，在 parentable 主题池里做 embedding range 候选父召回。

    候选父池 = 本场已提取的「非段落事件且类型 ∈ `_PARENTABLE_TYPES`」主题（板块/指数/
    板块黑话/加密货币/债券）；召回对象 = 「非段落事件且类型 ∈ `_CHILDABLE_TYPES`」主题。
    一次性 embed 全部对象与池，本地算两两余弦，每个 childable 主题召回相似度严格大于
    `threshold` 的全部候选父（range search，不设 k 上限），按相似度降序、排除自身。

    返回结构对齐 4_4 输入：所有主题都有 key；段落事件与不可作子类型（债券/加密货币）
    的候选父恒空（它们只作父不作子）。
    """
    result: dict[str, list[str]] = {}
    childable: list[str] = []
    parentable: list[str] = []
    for topic, td in keyword_data.items():
        if not isinstance(td, TopicData):
            continue
        # 非行业概念（科创板/红利/次新股等）不参与父子挂载：既不作父、也不作子，
        # 其下个股由 4_7 全量候选改归真实一级行业。
        is_concept = is_non_industry_concept(topic)
        if is_segment_event(td) or td.类型 not in _CHILDABLE_TYPES or is_concept:
            result[topic] = []
        else:
            childable.append(topic)
        if not is_segment_event(td) and td.类型 in _PARENTABLE_TYPES and not is_concept:
            parentable.append(topic)

    if not childable or not parentable:
        return result

    client = EmbeddingClient()
    child_vecs = client.embed(childable)
    parent_vecs = client.embed(parentable)
    for ci, child in enumerate(childable):
        scored: list[tuple[str, float]] = []
        for pi, parent in enumerate(parentable):
            if parent == child:
                continue
            sim = _cosine(child_vecs[ci], parent_vecs[pi])
            if sim <= threshold:
                continue
            scored.append((parent, sim))
        scored.sort(key=lambda x: (-x[1], x[0]))
        result[child] = [p for p, _ in scored]
    return result


def precompute_event_candidates(
    keyword_data: dict[str, TopicData],
    segments: list[TopicSegment],
) -> dict[str, list[str]]:
    """代码预筛事件候选父（纯 Python，零 LLM）。

    对每个主题，把锚点行号映射到话题段，候选父 = 段上标注的段落事件主题（走 4_5
    字幕+联网判定「当事资产」）。类型白名单硬约束：段落事件只能作父（候选父恒空）；
    不在子类型白名单的类型（债券/加密货币）只作父不作子（候选父恒空）。
    """
    seg_by_row: dict[int, int] = {}
    seg_events: dict[int, str] = {}
    for seg_idx, seg in enumerate(segments):
        if not isinstance(seg, TopicSegment) or seg.是否重要 is False:
            continue
        try:
            start = int(seg.开始行号)
            end = int(seg.结束行号)
        except (KeyError, TypeError, ValueError):
            continue
        if start > end:
            start, end = end, start
        for r in range(start, end + 1):
            seg_by_row[r] = seg_idx
        event_name = str(seg.事件 or "").strip()
        if event_name:
            seg_events[seg_idx] = event_name

    topic_segs: dict[str, set[int]] = {}
    for topic, td in keyword_data.items():
        if not isinstance(td, TopicData):
            continue
        segs: set[int] = set()
        for r in td.引用 or []:
            try:
                rid = int(r)
            except (TypeError, ValueError):
                continue
            if rid in seg_by_row:
                segs.add(seg_by_row[rid])
        topic_segs[topic] = segs

    event: dict[str, list[str]] = {}
    for topic, td in keyword_data.items():
        if not isinstance(td, TopicData):
            continue
        # 段落事件只能作父；不在子类型白名单的类型（债券/加密货币等）只作父不作子。
        if is_segment_event(td) or td.类型 not in _CHILDABLE_TYPES:
            event[topic] = []
            continue
        ev: set[str] = set()
        for s in topic_segs.get(topic, set()):
            e = seg_events.get(s)
            if e and e in keyword_data and is_segment_event(keyword_data.get(e)):
                ev.add(e)
        event[topic] = sorted(ev)
    return event


class CoarseParentFilter:
    """4_4 候选粗筛：代码预筛候选父 + LLM 凭世界知识逐对判定（无字幕、无事件）。"""

    def __init__(self) -> None:
        self._prompt = load_prompt("4_4_候选粗筛.txt", __file__)
        self._schema = _load_schema("4_4_候选粗筛", __file__)
        self._llm_client = LLMClient()
        _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("4_4_候选粗筛_user.jinja2")

    def generate(
        self,
        keyword_data: dict[str, TopicData],
        segments: list[TopicSegment],
        base_dir: Path | None = None,
        non_event_candidates: dict[str, list[str]] | None = None,
    ) -> tuple[dict[str, list[str]], list[dict]]:
        """候选父粗筛。返回 `(confirmed, doubt_pairs)`。

        - `confirmed`: `{主题名: [确认候选父]}`，仅含「作为父节点」的对；
        - `doubt_pairs`: `[{"子": 主题名, "父": 候选父名}]`，存疑的对，交给 4_5。
        整合落盘 `4_4_coarse_candidates.json`。

        `non_event_candidates` 可由编排层预筛后传入，避免重复计算；None 时内部自行预筛。
        """
        with step_logger("第4_4步-候选粗筛"):
            # 断点：4_4 结果（候选父板块 + 存疑）存在即复用
            if base_dir is not None:
                coarse_path = base_dir / "4_4_coarse_candidates.json"
                if coarse_path.exists():
                    cached = read_json(coarse_path)
                    if isinstance(cached, dict) and isinstance(cached.get("结果"), dict):
                        result = cached["结果"]
                        confirmed = result.get("候选父板块")
                        doubt = result.get("存疑")
                        if isinstance(confirmed, dict) and isinstance(doubt, list):
                            logger.info("检测到 4_4_coarse_candidates.json，跳过候选粗筛")
                            return confirmed, doubt

            if non_event_candidates is None:
                non_event_candidates = recall_candidate_parents_by_embedding(
                    keyword_data,
                    threshold=get_embedding_recall_threshold(),
                )
            topics = [t for t, td in keyword_data.items() if isinstance(td, TopicData)]
            if not topics:
                if base_dir is not None:
                    write_integrated(
                        base_dir / "4_4_coarse_candidates.json", {},
                        {"候选父板块": {}, "存疑": []},
                    )
                return {}, []

            # 短路：非事件主题候选父全空时，无候选可筛，跳过 LLM
            if not any(
                non_event_candidates.get(t)
                for t in topics
                if not is_segment_event(keyword_data.get(t))
            ):
                logger.info("代码预筛候选父全部为空，跳过 4_4 LLM 粗筛")
                if base_dir is not None:
                    write_integrated(
                        base_dir / "4_4_coarse_candidates.json", {},
                        {"候选父板块": {}, "存疑": []},
                    )
                return {}, []

            topic_list: list[dict] = []
            for t in topics:
                topic_list.append({
                    "主题名": t,
                    "类型": keyword_data[t].类型,
                    "候选父": non_event_candidates.get(t, []),
                })
            topic_list_sorted = sorted(
                topic_list, key=lambda t: -len(t.get("候选父") or []),
            )
            groups = round_robin_groups(
                topic_list_sorted,
                get_group_count("4_4_候选粗筛", CANDIDATE_GROUP_COUNT),
            )
            groups = [g for g in groups if g]

            log_dir = get_step_output_dir(base_dir, 4) if base_dir is not None else None
            futures = fanout("4_4", [
                (
                    self._filter_group,
                    (g_i, [t["主题名"] for t in groups[g_i]], topic_list, log_dir),
                    {},
                )
                for g_i in range(len(groups))
            ])
            merged_confirmed: dict[str, list[str]] = {}
            merged_doubt: list[dict] = []
            group_outputs: dict[str, dict] = {}
            for future, g_i in zip(futures, range(len(groups))):
                try:
                    result, group_confirmed, group_doubt = future.result()
                except Exception as e:  # noqa: BLE001 —— 单组失败降级，不拖垮整步
                    logger.warning("4_4 第 %d 组调用失败: %s", g_i, e)
                    continue
                if result is not None:
                    group_outputs[f"group_{g_i}"] = result
                for topic, parents in group_confirmed.items():
                    if not isinstance(parents, list):
                        continue
                    for p in parents:
                        if not isinstance(p, str):
                            continue
                        if p not in merged_confirmed.setdefault(topic, []):
                            merged_confirmed[topic].append(p)
                for d in group_doubt:
                    if isinstance(d, dict) and d.get("子") and d.get("父"):
                        merged_doubt.append({"子": d["子"], "父": d["父"]})

            confirmed, doubt_pairs = self._refine(
                merged_confirmed, merged_doubt, keyword_data, non_event_candidates,
            )
            logger.info(
                "4_4 粗筛后：确认父 %d 个主题、存疑 %d 对",
                sum(1 for v in confirmed.values() if v), len(doubt_pairs),
            )
            if base_dir is not None:
                write_integrated(
                    base_dir / "4_4_coarse_candidates.json", group_outputs,
                    {"候选父板块": confirmed, "存疑": doubt_pairs},
                )
            return confirmed, doubt_pairs

    def _filter_group(
        self,
        group_i: int,
        group_scope_names: list[str],
        full_topic_list: list[dict],
        log_dir: Path | None,
    ) -> tuple[dict | None, dict[str, list[str]], list[dict]]:
        """单组 4_4 粗筛调用，返回 (result dict, 确认候选父 dict, 存疑 list)。

        完整主题清单（含候选父）进上下文，仅【本批范围】限制输出键范围；越界键由
        `_refine` 兜底丢弃。
        """
        user_prompt = self._user_template.render(
            task_rules=self._prompt,
            topic_list=full_topic_list,
            batch_scope=group_scope_names,
        )
        logger.info(
            "4_4 第 %d 组 prompt 长度: %d，本批主题数: %d",
            group_i, len(user_prompt), len(group_scope_names),
        )
        result = self._llm_client.call(
            user_prompt=user_prompt,
            temperature=0.0,
            schema=self._schema,
            log_dir=log_dir,
            log_name=f"4_4_候选粗筛_group{group_i}",
            step_name="4_4_候选粗筛",
        )
        if not isinstance(result, dict):
            return None, {}, []
        confirmed = result.get("候选父板块", {})
        doubt = result.get("存疑", [])
        return result, confirmed if isinstance(confirmed, dict) else {}, (
            doubt if isinstance(doubt, list) else []
        )

    @staticmethod
    def _refine(
        merged_confirmed: dict[str, list[str]],
        merged_doubt: list[dict],
        keyword_data: dict[str, TopicData],
        non_event_candidates: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]], list[dict]]:
        """清洗 LLM 输出：确认父只保留预筛候选、事件候选父恒空；存疑对去重并圈定范围。"""
        refined: dict[str, list[str]] = {}
        for topic in keyword_data:
            if not isinstance(keyword_data.get(topic), TopicData):
                continue
            if is_segment_event(keyword_data.get(topic)):
                refined[topic] = []
                continue
            allowed = set(non_event_candidates.get(topic, []))
            parents = merged_confirmed.get(topic)
            if not isinstance(parents, list):
                refined[topic] = []
                continue
            kept: list[str] = []
            for p in parents:
                if not isinstance(p, str):
                    continue
                p = p.strip()
                if not p or p == topic or p not in allowed or p in kept:
                    continue
                kept.append(p)
            refined[topic] = kept

        refined_doubt: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for d in merged_doubt:
            child = (d.get("子") or "").strip()
            parent = (d.get("父") or "").strip()
            if not child or not parent or child == parent:
                continue
            if child not in keyword_data or parent not in keyword_data:
                continue
            if is_segment_event(keyword_data.get(child)):
                continue
            if parent not in non_event_candidates.get(child, []):
                continue
            if parent in refined.get(child, []):
                continue  # 已确认为父的不再存疑
            key = (child, parent)
            if key in seen:
                continue
            seen.add(key)
            refined_doubt.append({"子": child, "父": parent})
        return refined, refined_doubt
