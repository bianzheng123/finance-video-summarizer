"""4_7 祖先父生成：画 DAG 选顶层根节点，根节点过多时结合申万一级行业库拟父。

4_4~4_6 已筛出可多父的父子关系 DAG（child → 候选父列表）并去环。
本阶段：
1. 画 DAG、选出顶层根节点（DAG 里没有任何父节点的节点）。
2. 非事件根节点超过 `ANCESTOR_TRIGGER_THRESHOLD`（10）个时触发拟父，流程：
   a. 候选父池 = root_sector.json 的全部申万一级行业键（不再做 embedding 召回——
      embedding 余弦相似度对「子行业 → 一级行业」的包含关系召回太弱，如
      「覆铜板→电子」「电池→电力设备」都召不回，导致子行业漏成孤儿根）；
   b. 每个根节点的候选父 = 全量候选父池，批量 LLM 凭世界知识逐对判定
      候选父是否范畴包含根节点；
   c. 建缺席桶 + 并入 DAG。
   已是申万一级行业的根节点（如「有色金属」「银行」）无需再拟父，直接跳过；
   事件根节点不参与（事件恒为顶层，不需要拟父）。

只拟**一层**父，不递归拟父链——一级行业名是否还需归并到更大范畴，交由下游
`ParentRanker` / 层级压缩统一处理。
"""

import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..utils.llm_text_utils import load_prompt
from ..schemas import TopicData
from ...llm_infra.batch import round_robin_groups
from ...llm_infra.io import read_json, write_integrated
from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ...llm_infra.config import get_group_count
from ...llm_infra.resource import fanout
from ...log_config import step_logger
from .event_topics import is_segment_event, is_unresolved_jargon
from .bucket_utils import create_bucket, is_non_industry_concept, is_non_industry_type

logger = logging.getLogger(__name__)

# 非事件根节点超过该阈值才触发 4_7 拟父（祖先节点过多才归并，成本控制）。
ANCESTOR_TRIGGER_THRESHOLD = 10

_DATABASE_DIR = Path(__file__).parent.parent.parent.parent / "database"


def _load_root_sector() -> dict[str, list[str]]:
    """加载 root_sector.json，结构 `{一级行业: [二级/三级子行业]}`。"""
    tree_path = _DATABASE_DIR / "root_sector.json"
    raw = json.loads(tree_path.read_text(encoding="utf-8"))
    root: dict[str, list[str]] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            root[k] = [s for s in v if isinstance(s, str)]
        else:
            root[k] = []
    return root


def select_root_nodes(
    candidates: dict[str, list[str]],
    keyword_data: dict[str, TopicData],
) -> list[str]:
    """从候选 DAG 选出顶层根节点（没有任何候选父的节点）。

    DAG 是 child → 候选父列表；根节点 = 候选父为空的节点（source）。
    """
    return sorted(
        t for t in keyword_data
        if isinstance(keyword_data.get(t), TopicData)
        and not candidates.get(t)
    )


class AncestorParentGenerator:
    """4_7 祖先父生成：根节点过多时结合一级行业库拟一层父。"""

    def __init__(self) -> None:
        self._prompt = load_prompt("4_7_候选父粗筛.txt", __file__)
        self._schema = _load_schema("4_7_候选父粗筛", __file__)
        self._root_sectors = _load_root_sector()
        self._llm_client = LLMClient()
        _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("4_7_候选父粗筛_user.jinja2")

    def generate(
        self,
        candidates: dict[str, list[str]],
        keyword_data: dict[str, TopicData],
        base_dir: Path | None = None,
    ) -> dict[str, list[str]]:
        """画 DAG 选根节点，根节点过多时为非事件根节点拟父，返回补充后的 candidates。

        整合落盘 `4_7_ancestor_generation.json`（输出 + 补充后的 candidates）。
        """
        with step_logger("第4_7步-祖先父生成"):
            # 断点：4_5 结果（补充后的 candidates）存在即复用
            if base_dir is not None:
                ancestor_path = base_dir / "4_7_ancestor_generation.json"
                if ancestor_path.exists():
                    cached = read_json(ancestor_path)
                    if isinstance(cached, dict) and isinstance(cached.get("结果"), dict):
                        logger.info("检测到 4_7_ancestor_generation.json，跳过祖先父生成")
                        return cached["结果"]

            roots = select_root_nodes(candidates, keyword_data)
            root_sector_keys = set(self._root_sectors)
            non_event_roots = [
                r for r in roots
                if not is_segment_event(keyword_data.get(r))
                and not is_unresolved_jargon(keyword_data.get(r))
                and r not in root_sector_keys  # 已是申万一级行业的根无需再拟父
                and not is_non_industry_concept(r)  # 非行业概念（科创板等）不拟父
                and not is_non_industry_type(keyword_data.get(r))  # 指数/风格因子不拟父
            ]
            if len(non_event_roots) <= ANCESTOR_TRIGGER_THRESHOLD:
                logger.info(
                    "非事件根节点 %d 个（总根节点 %d）未超过阈值 %d，跳过 4_7 拟父",
                    len(non_event_roots), len(roots), ANCESTOR_TRIGGER_THRESHOLD,
                )
                return candidates

            logger.info(
                "非事件根节点 %d 个超过阈值，触发 4_7 拟父", len(non_event_roots),
            )
            group_outputs: dict[str, dict] = {}
            parent_pool = self._build_candidate_parent_pool()
            full_candidates = self._full_candidates(non_event_roots, parent_pool)
            if base_dir is not None:
                write_integrated(
                    base_dir / "4_7_full_candidates.json",
                    {},
                    full_candidates,
                    extra={"候选父池": parent_pool},
                )
            ancestor_candidates = self._filter_parents_batch(
                non_event_roots, full_candidates, base_dir, group_outputs,
            )

            # 建缺席桶：候选父不在 keyword_data 里时立即建桶（引用为空、打辅助标记），
            # 使 4_7 结束即得到完整父子关系图（4_8 候选剪枝无需再补桶）。
            created_buckets: list[str] = []
            for parents in ancestor_candidates.values():
                if not isinstance(parents, list):
                    continue
                for p in parents:
                    if not isinstance(p, str):
                        continue
                    p = p.strip()
                    if not p or p in keyword_data:
                        continue
                    create_bucket(keyword_data, p)
                    created_buckets.append(p)
            if created_buckets:
                logger.info(
                    "4_7 建缺席桶 %d 个：%s", len(created_buckets), sorted(created_buckets),
                )

            merged = {k: list(v) for k, v in candidates.items()}
            for root, parents in ancestor_candidates.items():
                if root not in keyword_data or not isinstance(parents, list):
                    continue
                for p in parents:
                    if not isinstance(p, str):
                        continue
                    p = p.strip()
                    if not p or p == root:
                        continue
                    if p in keyword_data and is_segment_event(keyword_data.get(p)):
                        continue  # 4_7 只拟板块父，不拟段落事件父
                    if p not in merged.setdefault(root, []):
                        merged[root].append(p)

            if base_dir is not None:
                write_integrated(
                    base_dir / "4_7_ancestor_generation.json", group_outputs, merged,
                )
            return merged

    def _build_candidate_parent_pool(self) -> list[str]:
        """候选父池 = root_sector.json 的全部申万一级行业键。

        不做 embedding 召回、也不排除根节点——每个待拟父的根节点都拿到全量
        一级行业候选，由 LLM 凭世界知识判定归属。排除根节点自身的职责由
        `_full_candidates` 完成（根节点本身不在 root_sector 键中，天然不会自指）。
        """
        pool = list(self._root_sectors)
        logger.info("4_7 候选父池 %d 个（root_sector.json 全部一级行业键）", len(pool))
        return pool

    def _full_candidates(
        self,
        roots: list[str],
        parent_pool: list[str],
    ) -> dict[str, list[str]]:
        """每个根节点的候选父 = 全量 parent_pool（排除自身，防自指）。

        替代旧的 embedding 召回：embedding 余弦相似度对「子行业 → 一级行业」的
        包含关系召回太弱（「覆铜板→电子」「电池→电力设备」都召不回），改为把
        全部一级行业候选交给 LLM 凭世界知识判定，召回短板由模型知识兜底。
        """
        out: dict[str, list[str]] = {}
        for r in roots:
            out[r] = [p for p in parent_pool if p != r]
        logger.info(
            "4_7 全量候选：%d 个根节点 × %d 个候选父",
            len(roots), len(parent_pool),
        )
        return out

    def _filter_parents_batch(
        self,
        root_nodes: list[str],
        full_candidates: dict[str, list[str]],
        base_dir: Path | None,
        group_outputs: dict[str, dict],
    ) -> dict[str, list[str]]:
        """对全量候选父做批量 LLM 世界知识粗筛（同 4_4）。

        只保留判定为「作为父」的候选父（`候选父板块`）；存疑对在 4_7 不做精筛，
        直接丢弃（拿不准就不挂父，根节点保持顶层）。

        Returns: `{根节点: [确认候选父]}`；整合落盘 `4_7_batch_filter.json`。
        """
        topic_list = [
            {"主题名": r, "候选父": full_candidates.get(r, [])}
            for r in root_nodes
        ]
        with_candidates = [t for t in topic_list if t["候选父"]]
        if not with_candidates:
            logger.info("4_7 无候选父，跳过批量 LLM 粗筛")
            return {r: [] for r in root_nodes}

        topic_list_sorted = sorted(
            with_candidates, key=lambda t: -len(t["候选父"]),
        )
        groups = round_robin_groups(
            topic_list_sorted,
            get_group_count("4_7_候选父粗筛", 5),
        )
        groups = [g for g in groups if g]
        log_dir = get_step_output_dir(base_dir, 4) if base_dir is not None else None
        logger.info(
            "4_7 批量粗筛：%d 个根节点带候选父 → %d 组并行",
            len(with_candidates), len(groups),
        )

        futures = fanout("4_7", [
            (
                self._filter_group,
                (g_i, [t["主题名"] for t in groups[g_i]], topic_list, log_dir),
                {},
            )
            for g_i in range(len(groups))
        ])
        merged: dict[str, list[str]] = {r: [] for r in root_nodes}
        for future, g_i in zip(futures, range(len(groups))):
            try:
                result, group_confirmed = future.result()
            except Exception as e:  # noqa: BLE001 —— 单组失败降级，不拖垮整步
                logger.warning("4_7 第 %d 组候选父粗筛失败: %s", g_i, e)
                continue
            if result is not None:
                group_outputs[f"group_{g_i}"] = result
            for topic, parents in group_confirmed.items():
                if not isinstance(parents, list):
                    continue
                for p in parents:
                    if not isinstance(p, str):
                        continue
                    if p not in merged.setdefault(topic, []):
                        merged[topic].append(p)

        confirmed = self._refine_confirmed(merged, full_candidates)
        logger.info(
            "4_7 批量粗筛：确认父 %d 个根节点",
            sum(1 for v in confirmed.values() if v),
        )
        if base_dir is not None:
            write_integrated(
                base_dir / "4_7_batch_filter.json", group_outputs, confirmed,
            )
        return confirmed

    def _filter_group(
        self,
        group_i: int,
        group_scope_names: list[str],
        full_topic_list: list[dict],
        log_dir: Path | None,
    ) -> tuple[dict | None, dict[str, list[str]]]:
        """单组 4_7 候选父粗筛调用，返回 (result dict, 确认候选父 dict)。

        完整主题清单（含候选父）进上下文，仅【本批范围】限制输出键范围；越界键由
        `_refine_confirmed` 兜底丢弃。
        """
        user_prompt = self._user_template.render(
            task_rules=self._prompt,
            topic_list=full_topic_list,
            batch_scope=group_scope_names,
        )
        logger.info(
            "4_7 第 %d 组 prompt 长度: %d，本批主题数: %d",
            group_i, len(user_prompt), len(group_scope_names),
        )
        result = self._llm_client.call(
            user_prompt=user_prompt,
            temperature=0.0,
            schema=self._schema,
            log_dir=log_dir,
            log_name=f"4_7_候选父粗筛_group{group_i}",
            step_name="4_7_候选父粗筛",
        )
        if not isinstance(result, dict):
            return None, {}
        confirmed = result.get("候选父板块", {})
        return result, confirmed if isinstance(confirmed, dict) else {}

    @staticmethod
    def _refine_confirmed(
        merged: dict[str, list[str]],
        full_candidates: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """清洗 LLM 输出：确认父只保留全量候选池里的父（root_sector 一级行业键）。"""
        refined: dict[str, list[str]] = {}
        for topic, allowed_list in full_candidates.items():
            allowed = set(allowed_list)
            parents = merged.get(topic)
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
        return refined
