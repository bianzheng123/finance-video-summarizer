"""5_1 宏观分析选段：渐进式披露的第一跳——先只给段索引，让模型挑相关段。

索引由**现有产物拼成、零 LLM 成本**：优先用 3_2 话题重置后的最终话题段
（`3_2_topic_reset.json` 的「结果」，含 `话题标签` / `事件` / `开始行号` /
`结束行号`），退回 1_2 合并段（`1_2_segment_merging.json`），再退回用
keyword_data 的引用区间（主题名当标签）。

每行给出「段号 + 行号范围 + 话题标签 + 事件 + 首行原文片段」——模型凭这几项就够
判断该段与某个宏观字段组是否相关，不必看全文；选中后第二跳才把段全文喂进去。

选段结果只是**候选范围的提示**，不是硬约束：下游仍按合法行号集合过滤引用，
选错段最多是漏召回（由「宁可选多」的指引兜底），不会引入幻觉。
"""

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ...llm_infra.io import read_json
from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ...log_config import step_logger
from ..schemas import SubtitleRow, TopicData, TopicSegment
from ..utils.llm_text_utils import llm_text_of
from .macro_analyzer import MacroFieldGroup, _sanitize_filename

logger = logging.getLogger(__name__)

STEP_NUMBER_STR = "5_1"
STEP_NUMBER = 5
# 索引里每段首行片段的长度：够判断话题，又不至于把字幕复制一遍
_SNIPPET_CHARS = 60


def _rowid_to_text(subtitle_l: list[SubtitleRow]) -> dict[int, str]:
    out: dict[int, str] = {}
    for sub in subtitle_l:
        rid = sub.rowID
        if rid is None:
            continue
        try:
            out[int(rid)] = llm_text_of(sub)
        except (TypeError, ValueError):
            continue
    return out


def _index_entries_from_segments(segments: list[TopicSegment]) -> list[dict]:
    """步骤一合并段 → 索引条目；跳过「不重要」段与缺行号的畸形段。"""
    entries: list[dict] = []
    for seg in segments:
        if seg.是否重要 is False:
            continue
        start = int(seg.开始行号)
        end = int(seg.结束行号)
        if end < start:
            start, end = end, start
        label = str(seg.话题标签 or "").strip() or "未命名"
        event = str(seg.事件 or "").strip()
        entries.append({
            "开始行号": start, "结束行号": end,
            "话题标签": label, "事件": event,
        })
    return entries


def _index_entries_from_keyword_data(
    keyword_data: dict[str, TopicData],
) -> list[dict]:
    """退化路径：keyword_data 每个主题的引用区间各成一段（主题名当标签）。"""
    entries: list[dict] = []
    for topic, data in (keyword_data or {}).items():
        if not isinstance(data, TopicData):
            continue
        ranges = [
            (int(r["开始行号"]), int(r["结束行号"]))
            for r in (data.引用 or [])
            if isinstance(r, dict)
            and r.get("开始行号") is not None
            and r.get("结束行号") is not None
        ]
        for start, end in ranges:
            if end < start:
                start, end = end, start
            entries.append({
                "开始行号": start, "结束行号": end,
                "话题标签": str(topic), "事件": "",
            })
    entries.sort(key=lambda e: e["开始行号"])
    return entries


def build_segment_index(
    base_dir: Path | None,
    keyword_data: dict[str, TopicData],
    subtitle_l: list[SubtitleRow],
) -> tuple[str, dict[int, set[int]]]:
    """拼话题段索引，返回 (索引文本, {段号: 该段 rowID 集合})。

    段号从 1 开始连续编号，与模型输出的 `选中段号` 直接对应；空索引返回空串
    （调用方据此跳过选段、退回整场字幕）。
    """
    entries: list[dict] = []
    # 回退链按「谁的分割更新」排序：3_2 话题重置的产物是段列表的**最终**形态，
    # 1_2 只是它的输入。读错顺序会让选段用上被 3_2 修掉的旧边界。
    if base_dir is not None:
        for name in ("3_2_topic_reset.json", "1_2_segment_merging.json"):
            data = read_json(base_dir / name)
            if isinstance(data, dict) and isinstance(data.get("结果"), list):
                entries = _index_entries_from_segments(
                    [TopicSegment.model_validate(seg) for seg in data["结果"]]
                )
                if entries:
                    break
    if not entries:
        if base_dir is not None:
            logger.warning(
                "3_2_topic_reset.json / 1_2_segment_merging.json 均缺失或无效，"
                "改用 keyword_data 引用区间拼索引",
            )
        entries = _index_entries_from_keyword_data(keyword_data)
    if not entries:
        return "", {}

    entries.sort(key=lambda e: (e["开始行号"], e["结束行号"]))
    rid2text = _rowid_to_text(subtitle_l)
    lines: list[str] = []
    seg_rows: dict[int, set[int]] = {}
    for i, e in enumerate(entries, start=1):
        rows = set(range(e["开始行号"], e["结束行号"] + 1))
        seg_rows[i] = rows
        head = (
            f"- 段{i} [行号 {e['开始行号']}-{e['结束行号']}]"
            f" 话题标签：{e['话题标签']}"
            + (f"｜事件：{e['事件']}" if e["事件"] else "")
        )
        snippet = (rid2text.get(e["开始行号"]) or "").strip()
        if snippet:
            head += f"\n  首行：{snippet[:_SNIPPET_CHARS]}"
        lines.append(head)
    return "\n".join(lines), seg_rows


class MacroSegmentSelector:
    """5_1 选段：每个字段组一次调用，只吃段索引。"""

    def __init__(self, llm_client: LLMClient, prompt: str) -> None:
        self._llm_client = llm_client
        self._prompt = prompt
        self._schema = _load_schema("5_1_宏观分析选段", __file__)
        _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("5_1_宏观分析选段_user.jinja2")

    def select(
        self,
        group: MacroFieldGroup,
        segment_index: str,
        base_dir: Path,
        result_sink: dict[str, dict] | None = None,
    ) -> list[int]:
        """返回该字段组选中的段号列表（非法/缺失段号由调用方按 seg_rows 过滤）。"""
        safe_name = _sanitize_filename(group.名称)
        with step_logger(f"第{STEP_NUMBER_STR}步-选段-{group.名称}"):
            user_prompt = self._user_template.render(
                task_rules=self._prompt,
                segment_index=segment_index,
                group=group,
            )
            log_name = f"{STEP_NUMBER_STR}_宏观分析选段_{safe_name}"
            result = self._llm_client.call(
                user_prompt=user_prompt,
                temperature=0.0,
                schema=self._schema,
                log_dir=get_step_output_dir(base_dir, STEP_NUMBER)
                if base_dir is not None else None,
                log_name=log_name,
                step_name=f"{STEP_NUMBER_STR}_宏观分析选段_{group.名称}",
            )
            if result_sink is not None:
                result_sink[log_name] = result
            picked = result.get("选中段号") if isinstance(result, dict) else None
            if not isinstance(picked, list):
                logger.warning(
                    "字段组「%s」选段返回非法，退回不选段（整场字幕）", group.名称,
                )
                return []
            nums = [int(n) for n in picked if isinstance(n, int) and n >= 1]
            logger.info(
                "字段组「%s」选中 %d 个话题段：%s", group.名称, len(nums), nums,
            )
            return nums
