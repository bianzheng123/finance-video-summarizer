"""4_6 关系去环：对候选父图检测强连通分量（环），环内边用联网判定三态并应用。

4_4 + 4_5 合并后的候选图可能仍含环（A→B、B→A 或更长循环），环意味着父子方向
矛盾。本阶段：

1. Tarjan 强连通分量（SCC）检测，大小 >1 的 SCC 视为环；
2. 每个环（SCC）整体一次 `call_conversation` + `WEB_SEARCH_TOOL`（1 轮额度），
   输出该环内部**所有有向边**的三态判定「父子 / 无关 / 相同」；
3. 应用判定：`无关` 删边、`相同` 合并两主题（复用 `TopicDeduper._rename_topic`）、
   `父子` 按 `父` 字段定向保留单向边。

迭代处理直到无环（「相同」合并与「父子」定向都可能改变图结构，需重扫），
极端残留（环上所有边都被判父子导致逻辑矛盾）由两层压缩的路径压缩兜底。
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ...llm_infra.io import read_json, write_integrated
from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ...llm_infra.resource import fanout
from ...llm_infra.tools import WEB_SEARCH_TOOL
from ...llm_infra.clients import WSASearchClient
from ...log_config import step_logger
from ..schemas import SubtitleRow, TopicData
from ..utils.llm_text_utils import (
    load_prompt,
    render_annotations,
    render_windowed_subtitle,
    window_row_ids,
)
from .topic_dedup import TopicDeduper

logger = logging.getLogger(__name__)

# call_conversation 的 max_rounds（含收尾轮）：1 轮联网搜索 + 1 轮收尾。
CYCLE_RESOLVE_MAX_ROUNDS = 2

# 去环迭代上限（「相同」合并 /「父子」定向都可能改变图，重扫直到无环或达到上限）。
_MAX_ITERATIONS = 5

# 单条查询回灌给模型的搜索结果条数上限。
_WSA_TOPK = 5


def _find_sccs(candidates: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan 强连通分量，返回大小 >1 的 SCC（环）节点列表。"""
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in candidates.get(v, []):
            if w not in candidates:
                continue  # 只遍历候选图内的节点
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(scc)

    for v in candidates:
        if v not in indices:
            strongconnect(v)
    return sccs


def _scc_edges(
    candidates: dict[str, list[str]],
    scc: list[str],
    keyword_data: dict[str, TopicData],
) -> list[dict]:
    """收集 SCC 内部所有有向边（child → parent，child 把 parent 列为候选父），带主题类型。"""
    scc_set = set(scc)
    edges: list[dict] = []
    for child in scc:
        for parent in candidates.get(child, []):
            if parent not in scc_set:
                continue
            child_td = keyword_data.get(child)
            parent_td = keyword_data.get(parent)
            edges.append({
                "主题1": child,
                "主题1类型": child_td.类型 if child_td is not None else "",
                "主题2": parent,
                "主题2类型": parent_td.类型 if parent_td is not None else "",
            })
    return edges


def _merge_candidates(
    candidates: dict[str, list[str]],
    merge_name: str,
    keep_name: str,
) -> dict[str, list[str]]:
    """把 merge_name 的候选关系并入 keep_name，并重写全图指向。"""
    merged_parents: list[str] = list(candidates.get(keep_name, []))
    for p in candidates.get(merge_name, []):
        if p != keep_name and p not in merged_parents:
            merged_parents.append(p)
    candidates[keep_name] = merged_parents
    candidates.pop(merge_name, None)
    for child in list(candidates.keys()):
        new_ps: list[str] = []
        for p in candidates.get(child, []):
            np = keep_name if p == merge_name else p
            if np == child or np in new_ps:
                continue
            new_ps.append(np)
        candidates[child] = new_ps
    return candidates


class CycleResolver:
    """4_6 关系去环：SCC 环检测 + 环内边联网三态判定 + 应用。"""

    def __init__(self) -> None:
        self._prompt = load_prompt("4_6_关系去环.txt", __file__)
        self._schema = _load_schema("4_6_关系去环", __file__)
        self._wsa: WSASearchClient | None = None
        self._llm_client = LLMClient()
        _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("4_6_关系去环_user.jinja2")

    def resolve(
        self,
        keyword_data: dict[str, TopicData],
        candidates: dict[str, list[str]],
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None = None,
    ) -> tuple[dict[str, TopicData], dict[str, list[str]]]:
        """检测并打破候选图环。返回 `(去环后 keyword_data, 去环后 candidates)`。

        落盘 `4_6_cycle_resolved.json`（含合并对）。
        """
        with step_logger("第4_6步-关系去环"):
            # 断点：4_6 结果（去环后 candidates + 合并对）存在即复用
            if base_dir is not None:
                cycle_path = base_dir / "4_6_cycle_resolved.json"
                if cycle_path.exists():
                    cached = read_json(cycle_path)
                    if isinstance(cached, dict) and isinstance(cached.get("结果"), dict):
                        candidates = cached["结果"]
                        merge_pairs = cached.get("合并对")
                        keyword_data = self._apply_merges(keyword_data, merge_pairs)
                        logger.info("检测到 4_6_cycle_resolved.json，跳过关系去环")
                        return keyword_data, candidates

            if not candidates:
                if base_dir is not None:
                    write_integrated(base_dir / "4_6_cycle_resolved.json", {}, {}, extra={"合并对": []})
                return keyword_data, candidates

            log_dir = get_step_output_dir(base_dir, 4) if base_dir is not None else None
            all_group_outputs: dict[str, dict] = {}
            merge_pairs: list[dict[str, str]] = []
            total_judged = 0

            for _ in range(_MAX_ITERATIONS):
                sccs = _find_sccs(candidates)
                if not sccs:
                    break
                scc_edges = [_scc_edges(candidates, scc, keyword_data) for scc in sccs]
                logger.info(
                    "4_6 检测到 %d 个环（共 %d 条环内边），开始联网判定",
                    len(sccs), sum(len(e) for e in scc_edges),
                )

                futures = fanout("4_6", [
                    (
                        self._judge_cycle,
                        (idx, edges, keyword_data, subtitle_l, log_dir),
                        {},
                    )
                    for idx, edges in enumerate(scc_edges)
                ])
                decisions: list[dict] = []
                for future, (idx, edges) in zip(futures, enumerate(scc_edges)):
                    try:
                        result, items = future.result()
                    except Exception as e:  # noqa: BLE001 —— 单元失败降级
                        logger.warning("4_6 第 %d 个环判定失败：%s", idx, e)
                        continue
                    if result is not None:
                        all_group_outputs[f"cycle_{idx}"] = result
                    decisions.extend(items)

                before_nodes = len(candidates)
                candidates, keyword_data, applied_merges = self._apply_decisions(
                    keyword_data, candidates, decisions,
                )
                merge_pairs.extend(applied_merges)
                total_judged += len(decisions)
                if len(candidates) == before_nodes and not applied_merges:
                    # 无删边、无合并（即所有边都判父子定向、环仍残留），避免死循环
                    logger.warning("4_6 本轮判定未减少任何边/节点（全判父子仍成环），停止迭代、进入兜底合并")
                    break

            remaining = _find_sccs(candidates)
            if remaining:
                keyword_data, candidates, collapse_merges = self._collapse_remaining_cycles(
                    keyword_data, candidates, remaining,
                )
                merge_pairs.extend(collapse_merges)
            logger.info(
                "4_6 去环完成：共判定 %d 条边、合并 %d 对",
                total_judged, len(merge_pairs),
            )
            if base_dir is not None:
                write_integrated(
                    base_dir / "4_6_cycle_resolved.json", all_group_outputs, candidates,
                    extra={"合并对": merge_pairs},
                )
            return keyword_data, candidates

    def _judge_cycle(
        self,
        idx: int,
        edges: list[dict],
        keyword_data: dict[str, TopicData],
        subtitle_l: list[SubtitleRow],
        log_dir: Path | None,
    ) -> tuple[dict | None, list[dict]]:
        """单环判定：渲染环内边 + 环上节点锚点窗口 + call_conversation。"""
        anchor_ids = self._anchor_ids_of(keyword_data, edges)
        subtitle_text = render_windowed_subtitle(subtitle_l, anchor_ids, with_type=True)
        annotation_text = render_annotations(
            subtitle_l, row_scope=window_row_ids(subtitle_l, anchor_ids),
        )
        user_prompt = self._user_template.render(
            task_rules=self._prompt,
            edges=edges,
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
        )
        logger.info("4_6 第 %d 个环 prompt 长度: %d，环内边: %d", idx, len(user_prompt), len(edges))
        result, _trace = self._llm_client.call_conversation(
            user_prompt=user_prompt,
            tools=[WEB_SEARCH_TOOL],
            tool_executor=self._make_tool_executor(),
            schema=self._schema,
            max_rounds=CYCLE_RESOLVE_MAX_ROUNDS,
            dedup_directions=True,
            max_tool_calls=len(edges),
            temperature=0.0,
            log_dir=log_dir,
            log_name=f"4_6_关系去环_p{idx}",
            step_name="4_6_关系去环",
        )
        if not isinstance(result, dict):
            return None, []
        items = result.get("判定")
        return result, items if isinstance(items, list) else []

    def _make_tool_executor(self) -> Callable[[str, dict], str]:
        """4_6 的 web_search 工具执行器（WSA 检索 + URL 去重 + TopK 截断）。"""

        def executor(name: str, args: dict) -> str:
            if name != "web_search":
                return json.dumps({"错误": f"未知工具 {name}"}, ensure_ascii=False)
            query = (args or {}).get("query") or ""
            if not query:
                return json.dumps({"错误": "缺少 query"}, ensure_ascii=False)
            if self._wsa is None:
                self._wsa = WSASearchClient()
            try:
                results = self._wsa.search(query) or []
            except Exception as e:  # noqa: BLE001
                logger.warning("WSA 搜索「%s」失败：%s", query, e)
                return json.dumps({"搜索词": query, "错误": str(e)}, ensure_ascii=False)
            seen: set[str] = set()
            deduped: list[dict] = []
            for r in results:
                url = (r.get("url") or "").strip()
                if url and url in seen:
                    continue
                if url:
                    seen.add(url)
                deduped.append(r)
                if len(deduped) >= _WSA_TOPK:
                    break
            return json.dumps({"搜索词": query, "搜索结果": deduped}, ensure_ascii=False)

        return executor

    @staticmethod
    def _anchor_ids_of(
        keyword_data: dict[str, TopicData], edges: list[dict],
    ) -> set[int]:
        """取环上所有节点的锚点行号并集（窗口 [DATA] 的中心）。"""
        names: set[str] = set()
        for e in edges:
            names.add(e.get("主题1", ""))
            names.add(e.get("主题2", ""))
        anchor_ids: set[int] = set()
        for name in names:
            td = keyword_data.get(name)
            if td is None:
                continue
            for r in td.引用 or []:
                try:
                    anchor_ids.add(int(r))
                except (TypeError, ValueError):
                    continue
        return anchor_ids

    @staticmethod
    def _apply_merges(
        keyword_data: dict[str, TopicData], merge_pairs: list[dict],
    ) -> dict[str, TopicData]:
        """断点恢复时重放「相同」合并到 keyword_data。"""
        for mp in merge_pairs or []:
            if not isinstance(mp, dict):
                continue
            merge_name = (mp.get("被合并名") or "").strip()
            keep_name = (mp.get("保留名") or "").strip()
            if not merge_name or not keep_name:
                continue
            if merge_name in keyword_data and keep_name in keyword_data:
                keyword_data = TopicDeduper._rename_topic(keyword_data, merge_name, keep_name, "")
        return keyword_data

    @staticmethod
    def _collapse_remaining_cycles(
        keyword_data: dict[str, TopicData],
        candidates: dict[str, list[str]],
        sccs: list[list[str]],
    ) -> tuple[dict[str, TopicData], dict[str, list[str]], list[dict[str, str]]]:
        """兜底：环上所有边都被判「父子」、仍成环时，视为相同概念合并。

        保留名取 `sorted` 后第一个（「随机」的确定性实现，保证断点续跑 /
        LLM_REPLAY 回放可复现）。合并后环上节点收缩为一个，环必然消失。
        """
        merge_pairs: list[dict[str, str]] = []
        for scc in sccs:
            nodes = sorted(n for n in scc if n in keyword_data)
            if len(nodes) < 2:
                continue
            keep_name = nodes[0]
            for node in nodes[1:]:
                logger.warning(
                    "LLM 推理发现环仍存在（全判父子），视为相同概念合并「%s」→「%s」",
                    node, keep_name,
                )
                keyword_data = TopicDeduper._rename_topic(keyword_data, node, keep_name, "")
                candidates = _merge_candidates(candidates, node, keep_name)
                merge_pairs.append({"被合并名": node, "保留名": keep_name})
        return keyword_data, candidates, merge_pairs

    @staticmethod
    def _apply_decisions(
        keyword_data: dict[str, TopicData],
        candidates: dict[str, list[str]],
        decisions: list[dict],
    ) -> tuple[dict[str, TopicData], dict[str, list[str]], list[dict[str, str]]]:
        """应用三态判定：无关删边、相同合并、父子定向。"""
        merge_pairs: list[dict[str, str]] = []
        for item in decisions:
            if not isinstance(item, dict):
                continue
            t1 = (item.get("主题1") or "").strip()
            t2 = (item.get("主题2") or "").strip()
            rel = item.get("关系")
            if not t1 or not t2 or t1 == t2:
                continue
            if rel == "无关":
                candidates[t1] = [p for p in candidates.get(t1, []) if p != t2]
            elif rel == "相同":
                keep = (item.get("父") or "").strip() or t1
                if keep not in (t1, t2):
                    logger.warning("4_6 相同判定保留字段非法（%r），默认保留「%s」", keep, t1)
                    keep = t1
                merge_name = t2 if keep == t1 else t1
                if merge_name not in keyword_data or keep not in keyword_data:
                    continue
                keyword_data = TopicDeduper._rename_topic(keyword_data, merge_name, keep, "")
                candidates = _merge_candidates(candidates, merge_name, keep)
                merge_pairs.append({"被合并名": merge_name, "保留名": keep})
            elif rel == "父子":
                parent = (item.get("父") or "").strip() or t2
                if parent == t2:
                    # 原始方向正确：t1 是子、t2 是父。保留 t1→t2，删反向 t2→t1
                    candidates[t2] = [p for p in candidates.get(t2, []) if p != t1]
                    lst = candidates.setdefault(t1, [])
                    if t2 not in lst:
                        lst.append(t2)
                elif parent == t1:
                    # 方向反：t1 是父、t2 是子。删 t1→t2，加 t2→t1
                    candidates[t1] = [p for p in candidates.get(t1, []) if p != t2]
                    lst = candidates.setdefault(t2, [])
                    if t1 not in lst:
                        lst.append(t1)
                else:
                    logger.warning("4_6 父子判定父字段非法（%r），默认「%s」为父", parent, t2)
                    candidates[t2] = [p for p in candidates.get(t2, []) if p != t1]
                    lst = candidates.setdefault(t1, [])
                    if t2 not in lst:
                        lst.append(t2)
        return candidates, keyword_data, merge_pairs
