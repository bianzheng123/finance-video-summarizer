"""4_3 精确去重：对 4_2 导出的困惑等价对，结合字幕上下文 + 联网搜索判定等价。

4_2 仅凭世界知识把「像等价」的主题对导出为困惑对（如「钴出口禁令」↔「刚果金
出口禁令」——刚果金是钴主产国，后者可能确实指前者，也可能是另一条独立的出口管制）。
本阶段对每一对渲染两个主题锚点行号窗口的字幕作上下文，用 `call_conversation` +
`WEB_SEARCH_TOOL`（联网最多 1 轮）判定；只有「极大确信等价」（输出 是否等价=true）
的对才由代码合并，其余保持各自独立。

并行方式：每个困惑对一个单元，`fanout` 提交到全局线程池并行（LLM 并发由
`RESOURCE_MANAGER.llm` 闸限制），单元间互不干扰。
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
from .event_topics import is_segment_event
from .topic_dedup import TopicDeduper

logger = logging.getLogger(__name__)

# call_conversation 的 max_rounds（含收尾轮）：1 轮联网搜索 + 1 轮收尾。
PRECISE_DEDUP_MAX_ROUNDS = 2

# 单条查询回灌给模型的搜索结果条数上限（与 2_3 的 WSA_TOPK 对齐）。
_WSA_TOPK = 5


class PreciseDeduper:
    """4_3 精确去重：困惑等价对 + 字幕上下文 + 联网判定。"""

    def __init__(self) -> None:
        self._prompt = load_prompt("4_3_精确去重.txt", __file__)
        self._schema = _load_schema("4_3_精确去重", __file__)
        self._wsa: WSASearchClient | None = None
        self._llm_client = LLMClient()
        _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("4_3_精确去重_user.jinja2")

    def dedup_precise(
        self,
        uncertain_pairs: list[dict],
        keyword_data: dict[str, TopicData],
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None,
    ) -> tuple[dict[str, TopicData], list[dict]]:
        """对 4_2 导出的困惑对精判，确定等价的对合并，返回合并后 keyword_data。

        每个困惑对一个单元并行调用；整对失败 / 无有效判定即跳过（不合并该对），
        保证「宁缺勿滥」——拿不准的宁可保持独立。
        """
        with step_logger("第4_3步-精确去重"):
            # 断点：4_3 结果（合并后 keyword_data）存在即复用
            if base_dir is not None:
                precise_path = base_dir / "4_3_precise_dedup.json"
                if precise_path.exists():
                    cached = read_json(precise_path)
                    if isinstance(cached, dict) and isinstance(cached.get("结果"), dict):
                        merge_pairs = cached.get("合并对")
                        logger.info("检测到 4_3_precise_dedup.json，跳过精确去重")
                        return (
                            {n: TopicData.model_validate(td) for n, td in cached["结果"].items()},
                            merge_pairs if isinstance(merge_pairs, list) else [],
                        )

            pairs = self._valid_pairs(uncertain_pairs, keyword_data)
            if not pairs:
                logger.info("无有效困惑对，跳过 4_3 精确去重")
                if base_dir is not None:
                    write_integrated(
                        base_dir / "4_3_precise_dedup.json", {},
                        {n: td.model_dump() for n, td in keyword_data.items()},
                        extra={"合并对": []},
                    )
                return keyword_data, []

            log_dir = get_step_output_dir(base_dir, 4) if base_dir is not None else None
            futures = fanout("4_3", [
                (
                    self._judge_pair,
                    (pair_idx, pair, keyword_data, subtitle_l, log_dir),
                    {},
                )
                for pair_idx, pair in enumerate(pairs)
            ])
            group_outputs: dict[str, dict] = {}
            decisions: list[dict] = []
            for future, (pair_idx, pair) in zip(futures, enumerate(pairs)):
                try:
                    item = future.result()
                except Exception as e:  # noqa: BLE001 —— 单元失败降级，不拖垮整步
                    logger.warning(
                        "4_3 对「%s」↔「%s」判定失败：%s",
                        pair["关键词1"], pair["关键词2"], e,
                    )
                    continue
                if isinstance(item, dict):
                    group_outputs[f"pair_{pair_idx}"] = item
                    if item.get("是否等价") is True:
                        decisions.append(item)
                else:
                    logger.warning(
                        "4_3 对「%s」↔「%s」无有效判定，跳过",
                        pair["关键词1"], pair["关键词2"],
                    )

            keyword_data, merge_pairs = self._apply_decisions(keyword_data, decisions)
            if base_dir is not None:
                write_integrated(
                    base_dir / "4_3_precise_dedup.json", group_outputs,
                    {n: td.model_dump() for n, td in keyword_data.items()},
                    extra={"合并对": merge_pairs},
                )
            return keyword_data, merge_pairs

    def _judge_pair(
        self,
        pair_idx: int,
        pair: dict,
        keyword_data: dict[str, TopicData],
        subtitle_l: list[SubtitleRow],
        log_dir: Path | None,
    ) -> dict | None:
        """单对判定：渲染两主题锚点窗口 + call_conversation，返回该对的判定 item。"""
        k1 = pair["关键词1"]
        k2 = pair["关键词2"]
        anchor_ids = self._anchor_ids_of(keyword_data, [k1, k2])
        subtitle_text = render_windowed_subtitle(subtitle_l, anchor_ids, with_type=True)
        annotation_text = render_annotations(
            subtitle_l, row_scope=window_row_ids(subtitle_l, anchor_ids),
        )
        user_prompt = self._user_template.render(
            task_rules=self._prompt,
            pairs=[pair],
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
        )
        result, _trace = self._llm_client.call_conversation(
            user_prompt=user_prompt,
            tools=[WEB_SEARCH_TOOL],
            tool_executor=self._make_tool_executor(),
            schema=self._schema,
            max_rounds=PRECISE_DEDUP_MAX_ROUNDS,
            dedup_directions=True,
            max_tool_calls=2,
            log_dir=log_dir,
            log_name=f"4_3_精确去重_p{pair_idx}",
            step_name="4_3_精确去重",
        )
        if not isinstance(result, dict):
            return None
        for item in result.get("判定") or []:
            if isinstance(item, dict) and item.get("关键词1") == k1 and item.get("关键词2") == k2:
                return item
        return None

    def _make_tool_executor(self) -> Callable[[str, dict], str]:
        """4_3 的 web_search 工具执行器（WSA 检索 + URL 去重 + TopK 截断）。"""

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

    @staticmethod
    def _valid_pairs(
        uncertain_pairs: list[dict], keyword_data: dict[str, TopicData],
    ) -> list[dict]:
        """过滤出关键词1与关键词2都仍在 keyword_data 里的困惑对。"""
        pairs: list[dict] = []
        for p in uncertain_pairs or []:
            if not isinstance(p, dict):
                continue
            k1 = (p.get("关键词1") or "").strip()
            k2 = (p.get("关键词2") or "").strip()
            if not k1 or not k2 or k1 == k2:
                continue
            if k1 not in keyword_data or k2 not in keyword_data:
                logger.warning(
                    "4_3 跳过主题不存在的困惑对：「%s」↔「%s」", k1, k2,
                )
                continue
            pairs.append({"关键词1": k1, "关键词2": k2, "理由": p.get("理由") or ""})
        return pairs

    @staticmethod
    def _anchor_ids_of(
        keyword_data: dict[str, TopicData], names: list[str],
    ) -> set[int]:
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

    @staticmethod
    def _apply_decisions(
        keyword_data: dict[str, TopicData], decisions: list[dict],
    ) -> tuple[dict[str, TopicData], list[dict]]:
        """把「是否等价=true」的对合并（方向由 `保留` 决定、段落事件优先，复用 _rename_topic 保事件类型保护）。

        返回 `(keyword_data, 合并对)`，合并对为 `[{"被合并名": ..., "保留名": ...}]`。
        """
        applied: list[dict[str, str]] = []
        for item in decisions:
            k1 = (item.get("关键词1") or "").strip()
            k2 = (item.get("关键词2") or "").strip()
            keep = (item.get("保留") or "").strip()
            if not k1 or not k2 or k1 == k2:
                continue
            if k1 not in keyword_data or k2 not in keyword_data:
                continue
            # 段落事件 vs 关键词事件：强制保留段落事件名（关键词事件并入段落事件）。
            k1_seg = is_segment_event(keyword_data.get(k1))
            k2_seg = is_segment_event(keyword_data.get(k2))
            if k1_seg and not k2_seg:
                keep = k1
            elif k2_seg and not k1_seg:
                keep = k2
            if keep == k2:
                keyword_data = TopicDeduper._rename_topic(keyword_data, k1, k2, "")
                applied.append({"被合并名": k1, "保留名": k2})
            else:
                if keep != k1:
                    logger.warning(
                        "4_3 判定等价但保留字段非法（%r），默认保留「关键词1」", keep,
                    )
                keyword_data = TopicDeduper._rename_topic(keyword_data, k2, k1, "")
                applied.append({"被合并名": k2, "保留名": k1})
        if applied:
            logger.info("4_3 精确去重合并 %d 对：%s", len(applied), applied)
        else:
            logger.info("4_3 无确定等价对需合并")
        return keyword_data, applied
