"""4_5 候选精筛拆分为两个串行子阶段：按「是否需要联网」分流。

- 4_5_1 ``DoubtParentFilter``：4_4 存疑对（板块/指数/板块黑话/加密货币/债券父），
  ``call_conversation + web_search`` 联网判定「父是否包含子」。
- 4_5_2 ``EventParentFilter``：段落事件候选父，零搜索 ``call`` 读字幕上下文判定
  「子是否为该事件的当事资产」。

并行方式：每个有待判对的主题一个工作单元，``fanout`` 提交到全局线程池并行。
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

logger = logging.getLogger(__name__)

# 4_5_1 call_conversation 的 max_rounds（含收尾轮）：1 轮联网搜索 + 1 轮收尾。
DOUBT_FILTER_MAX_ROUNDS = 2

# 单条查询回灌给模型的搜索结果条数上限（与 2_3 / 4_3 的 WSA_TOPK 对齐）。
_WSA_TOPK = 5


def _anchor_ids_of(keyword_data: dict[str, TopicData], names: list[str]) -> set[int]:
    """取若干主题的锚点行号并集（窗口 [DATA] 的中心）。"""
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


def _render_topic_context(
    keyword_data: dict[str, TopicData],
    topic: str,
    pairs: list[dict],
    subtitle_l: list[SubtitleRow],
) -> tuple[str, str]:
    """渲染单主题的 [DATA] 窗口与 [ANNOTATION]，返回 `(subtitle_text, annotation_text)`。"""
    anchor_ids = _anchor_ids_of(keyword_data, [topic] + [p["父"] for p in pairs])
    subtitle_text = render_windowed_subtitle(subtitle_l, anchor_ids, with_type=True)
    annotation_text = render_annotations(
        subtitle_l, row_scope=window_row_ids(subtitle_l, anchor_ids),
    )
    return subtitle_text, annotation_text


def _collect_pairs(
    keyword_data: dict[str, TopicData],
    raw_pairs: list[tuple[str, str]],
) -> dict[str, list[dict]]:
    """把 `(子, 父)` 对列表归并为「主题 → 待判对列表」（去重、过滤非法）。"""
    pending_by_topic: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for child, parent in raw_pairs:
        child = child.strip()
        parent = parent.strip()
        if not child or not parent or child == parent:
            continue
        if child not in keyword_data or parent not in keyword_data:
            continue
        if (child, parent) in seen:
            continue
        seen.add((child, parent))
        pending_by_topic.setdefault(child, []).append({
            "子": child,
            "子类型": keyword_data[child].类型,
            "父": parent,
            "父类型": keyword_data[parent].类型,
        })
    return pending_by_topic


def _refine(
    merged_confirmed: dict[str, list[str]],
    merged_doubt: list[dict],
    keyword_data: dict[str, TopicData],
) -> tuple[dict[str, list[str]], list[dict]]:
    """清洗 LLM 输出：确认父必须仍为主题、不能是子自身；存疑去重。"""
    refined: dict[str, list[str]] = {}
    for topic, parents in merged_confirmed.items():
        if topic not in keyword_data:
            continue
        kept: list[str] = []
        for p in parents:
            if not isinstance(p, str):
                continue
            p = p.strip()
            if not p or p == topic or p not in keyword_data or p in kept:
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
        if parent in refined.get(child, []):
            continue  # 已确认为父的不再存疑
        key = (child, parent)
        if key in seen:
            continue
        seen.add(key)
        refined_doubt.append({"子": child, "父": parent})
    return refined, refined_doubt


def _merge_group_results(
    futures: list,
    topic_items: list[tuple[str, list[dict]]],
) -> tuple[dict[str, list[str]], list[dict], dict[str, dict]]:
    """合并 fanout 结果：`(merged_confirmed, merged_doubt, group_outputs)`。"""
    merged_confirmed: dict[str, list[str]] = {}
    merged_doubt: list[dict] = []
    group_outputs: dict[str, dict] = {}
    for future, (idx, (topic, _pairs)) in zip(futures, enumerate(topic_items)):
        try:
            result, confirmed, doubt = future.result()
        except Exception as e:  # noqa: BLE001 —— 单元失败降级，不拖垮整步
            logger.warning("主题「%s」精筛失败：%s", topic, e)
            continue
        if result is not None:
            group_outputs[f"topic_{idx}"] = result
        for t, parents in confirmed.items():
            if not isinstance(parents, list):
                continue
            for p in parents:
                if not isinstance(p, str):
                    continue
                if p not in merged_confirmed.setdefault(t, []):
                    merged_confirmed[t].append(p)
        for d in doubt:
            if isinstance(d, dict) and d.get("子") and d.get("父"):
                merged_doubt.append({"子": d["子"], "父": d["父"]})
    return merged_confirmed, merged_doubt, group_outputs


class DoubtParentFilter:
    """4_5_1 存疑候选精筛：4_4 存疑对 + 字幕上下文 + 联网判定「父包含子」。"""

    def __init__(self) -> None:
        self._prompt = load_prompt("4_5_1_存疑候选精筛.txt", __file__)
        self._schema = _load_schema("4_5_1_存疑候选精筛", __file__)
        self._wsa: WSASearchClient | None = None
        self._llm_client = LLMClient()
        _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("4_5_1_存疑候选精筛_user.jinja2")

    def generate(
        self,
        keyword_data: dict[str, TopicData],
        doubt_pairs: list[dict],
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None = None,
    ) -> tuple[dict[str, list[str]], list[dict]]:
        """存疑对精筛。返回 `(confirmed, final_doubt)`。

        - `confirmed`: `{主题名: [确认候选父]}`；
        - `final_doubt`: `[{"子": ..., "父": ...}]`。
        """
        with step_logger("第4_5_1步-存疑候选精筛"):
            if base_dir is not None:
                refined_path = base_dir / "4_5_1_doubt_refined.json"
                if refined_path.exists():
                    cached = read_json(refined_path)
                    if isinstance(cached, dict) and isinstance(cached.get("结果"), dict):
                        confirmed = cached["结果"].get("候选父板块")
                        if isinstance(confirmed, dict):
                            logger.info("检测到 4_5_1_doubt_refined.json，跳过存疑候选精筛")
                            return confirmed, []

            raw_pairs: list[tuple[str, str]] = [
                ((d.get("子") or ""), (d.get("父") or ""))
                for d in doubt_pairs or [] if isinstance(d, dict)
            ]
            pending_by_topic = _collect_pairs(keyword_data, raw_pairs)
            if not pending_by_topic:
                logger.info("4_5_1 无存疑对，跳过存疑候选精筛")
                if base_dir is not None:
                    write_integrated(base_dir / "4_5_1_doubt_refined.json", {}, {})
                return {}, []

            log_dir = get_step_output_dir(base_dir, 4) if base_dir is not None else None
            topic_items = sorted(pending_by_topic.items())
            futures = fanout("4_5_1", [
                (
                    self._filter_topic,
                    (idx, topic, pairs, keyword_data, subtitle_l, log_dir),
                    {},
                )
                for idx, (topic, pairs) in enumerate(topic_items)
            ])
            merged_confirmed, merged_doubt, group_outputs = _merge_group_results(futures, topic_items)
            confirmed, final_doubt = _refine(merged_confirmed, merged_doubt, keyword_data)
            logger.info(
                "4_5_1 精筛后：确认父 %d 个主题、存疑 %d 对",
                sum(1 for v in confirmed.values() if v), len(final_doubt),
            )
            if base_dir is not None:
                write_integrated(
                    base_dir / "4_5_1_doubt_refined.json", group_outputs,
                    {"候选父板块": confirmed},
                )
            return confirmed, final_doubt

    def _filter_topic(
        self,
        idx: int,
        topic: str,
        pairs: list[dict],
        keyword_data: dict[str, TopicData],
        subtitle_l: list[SubtitleRow],
        log_dir: Path | None,
    ) -> tuple[dict | None, dict[str, list[str]], list[dict]]:
        """单主题精筛：渲染主题锚点窗口 + call_conversation，返回该主题判定。"""
        subtitle_text, annotation_text = _render_topic_context(keyword_data, topic, pairs, subtitle_l)
        user_prompt = self._user_template.render(
            task_rules=self._prompt,
            pairs=pairs,
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
        )
        logger.info(
            "4_5_1 主题「%s」prompt 长度: %d，待判对: %d",
            topic, len(user_prompt), len(pairs),
        )
        result, _trace = self._llm_client.call_conversation(
            user_prompt=user_prompt,
            tools=[WEB_SEARCH_TOOL],
            tool_executor=self._make_tool_executor(),
            schema=self._schema,
            max_rounds=DOUBT_FILTER_MAX_ROUNDS,
            dedup_directions=True,
            max_tool_calls=len(pairs),
            temperature=0.0,
            log_dir=log_dir,
            log_name=f"4_5_1_存疑候选精筛_p{idx}",
            step_name="4_5_1_存疑候选精筛",
        )
        if not isinstance(result, dict):
            return None, {}, []
        confirmed = result.get("候选父板块", {})
        doubt = result.get("存疑", [])
        return result, confirmed if isinstance(confirmed, dict) else {}, (
            doubt if isinstance(doubt, list) else []
        )

    def _make_tool_executor(self) -> Callable[[str, dict], str]:
        """4_5_1 的 web_search 工具执行器（WSA 检索 + URL 去重 + TopK 截断）。"""

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
            except Exception as e:  # noqa: BLE001 —— 网络请求，失败回灌错误让模型换词再搜
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


class EventParentFilter:
    """4_5_2 事件候选精筛：段落事件候选父 + 字幕上下文零搜索判定「当事资产」。"""

    def __init__(self) -> None:
        self._prompt = load_prompt("4_5_2_事件候选精筛.txt", __file__)
        self._schema = _load_schema("4_5_2_事件候选精筛", __file__)
        self._llm_client = LLMClient()
        _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("4_5_2_事件候选精筛_user.jinja2")

    def generate(
        self,
        keyword_data: dict[str, TopicData],
        event_candidates: dict[str, list[str]],
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None = None,
    ) -> tuple[dict[str, list[str]], list[dict]]:
        """事件候选父精筛。返回 `(confirmed, final_doubt)`。"""
        with step_logger("第4_5_2步-事件候选精筛"):
            if base_dir is not None:
                refined_path = base_dir / "4_5_2_event_refined.json"
                if refined_path.exists():
                    cached = read_json(refined_path)
                    if isinstance(cached, dict) and isinstance(cached.get("结果"), dict):
                        confirmed = cached["结果"].get("候选父板块")
                        if isinstance(confirmed, dict):
                            logger.info("检测到 4_5_2_event_refined.json，跳过事件候选精筛")
                            return confirmed, []

            raw_pairs: list[tuple[str, str]] = []
            for topic, parents in (event_candidates or {}).items():
                if not isinstance(parents, list):
                    continue
                for p in parents:
                    if isinstance(p, str):
                        raw_pairs.append((topic, p))
            pending_by_topic = _collect_pairs(keyword_data, raw_pairs)
            if not pending_by_topic:
                logger.info("4_5_2 无事件候选父，跳过事件候选精筛")
                if base_dir is not None:
                    write_integrated(base_dir / "4_5_2_event_refined.json", {}, {})
                return {}, []

            log_dir = get_step_output_dir(base_dir, 4) if base_dir is not None else None
            topic_items = sorted(pending_by_topic.items())
            futures = fanout("4_5_2", [
                (
                    self._filter_topic,
                    (idx, topic, pairs, keyword_data, subtitle_l, log_dir),
                    {},
                )
                for idx, (topic, pairs) in enumerate(topic_items)
            ])
            merged_confirmed, merged_doubt, group_outputs = _merge_group_results(futures, topic_items)
            confirmed, final_doubt = _refine(merged_confirmed, merged_doubt, keyword_data)
            logger.info(
                "4_5_2 精筛后：确认父 %d 个主题、存疑 %d 对",
                sum(1 for v in confirmed.values() if v), len(final_doubt),
            )
            if base_dir is not None:
                write_integrated(
                    base_dir / "4_5_2_event_refined.json", group_outputs,
                    {"候选父板块": confirmed},
                )
            return confirmed, final_doubt

    def _filter_topic(
        self,
        idx: int,
        topic: str,
        pairs: list[dict],
        keyword_data: dict[str, TopicData],
        subtitle_l: list[SubtitleRow],
        log_dir: Path | None,
    ) -> tuple[dict | None, dict[str, list[str]], list[dict]]:
        """单主题精筛：渲染主题锚点窗口 + 零搜索 call，返回该主题判定。"""
        subtitle_text, annotation_text = _render_topic_context(keyword_data, topic, pairs, subtitle_l)
        user_prompt = self._user_template.render(
            task_rules=self._prompt,
            pairs=pairs,
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
        )
        logger.info(
            "4_5_2 主题「%s」prompt 长度: %d，待判对: %d",
            topic, len(user_prompt), len(pairs),
        )
        result = self._llm_client.call(
            user_prompt=user_prompt,
            temperature=0.0,
            schema=self._schema,
            log_dir=log_dir,
            log_name=f"4_5_2_事件候选精筛_p{idx}",
            step_name="4_5_2_事件候选精筛",
        )
        if not isinstance(result, dict):
            return None, {}, []
        confirmed = result.get("候选父板块", {})
        doubt = result.get("存疑", [])
        return result, confirmed if isinstance(confirmed, dict) else {}, (
            doubt if isinstance(doubt, list) else []
        )
