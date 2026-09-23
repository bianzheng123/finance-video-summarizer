"""话题分割（流水线第 1.1 小步）。

分块并行 LLM 调用：把字幕按固定大小切成 chunks，每个 chunk 独立调用一次 LLM 输出该块的话题段。
每个 chunk 的输出经过严格校验（rowID 真实存在 + 段间不重叠 + 覆盖全部 rowID），失败时最多重试
2 次，仍失败则进入自动修复（snap 越界 rowID、消除重叠、填补缺口）。

聚合后按 开始行号 排序。这样可以避免单次 LLM 调用在长字幕上输出截断。
"""

import bisect
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from jinja2 import Environment, FileSystemLoader

from ...log_config import step_logger
from ..utils.llm_text_utils import (
    load_prompt,
    render_annotations,
    render_windowed_subtitle,
    window_row_ids,
)
from ..utils.rowid_scope import format_range_scope
from ..schemas import SubtitleRow, TopicSegment
from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ...llm_infra.resource import RESOURCE_MANAGER, fanout

_env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)

# 分割块大小（行）：块越小并行度越高、单块 reasoning 量越小（尾块决定整步墙钟）。
# 由 env 覆盖（.env 里 SEGMENT_CHUNK_SIZE），默认 400。
SEGMENT_CHUNK_SIZE = int(os.getenv("SEGMENT_CHUNK_SIZE", "400"))
SEGMENT_MAX_ATTEMPTS = 3  # 1 次首发 + 2 次重试

def clean_subtitle_knowledge(
    raw_entries: Any,
    full_text: str,
    seen_words: set[str] | None = None,
) -> list[dict]:
    """校验并去重「字幕知识」条目（1_1 / 1_2 共用）。

    规则：
    - `词` 必须是 `full_text` 的连续子串（可溯源锚点，与片段子串校验同口径）；
    - `解释` 非空；`解释行号` 为 `{"开始": int, "结束": int}` 对象，解析失败时不写该字段；
    - 按 `词` 去重（同词保留首条）。

    是否脑补靠 prompt 约束，代码只做确定性形状校验（不拦截、只提示）。
    """
    seen: set[str] = seen_words if seen_words is not None else set()
    out: list[dict] = []
    for e in raw_entries or []:
        if not isinstance(e, dict):
            continue
        word = (e.get("词") or "").strip()
        expl = (e.get("解释") or "").strip()
        if not word or not expl or word not in full_text:
            continue
        if word in seen:
            continue
        seen.add(word)
        entry: dict = {"词": word, "解释": expl}
        row_range = e.get("解释行号")
        if isinstance(row_range, dict):
            try:
                start = int(row_range.get("开始"))
                end = int(row_range.get("结束"))
            except (TypeError, ValueError):
                start = end = None
            if start is not None and end is not None and start >= 1 and end >= 1:
                if start > end:
                    start, end = end, start
                entry["解释行号"] = {"开始": start, "结束": end}
        out.append(entry)
    return out


class TopicSegmenter:
    """字幕话题分割：分块并行调 LLM。"""

    def __init__(self) -> None:
        self._prompt = load_prompt("1_1_话题分割.txt", __file__)
        self._schema = _load_schema("1_1_话题分割", __file__)
        self._llm_client = LLMClient()
        _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("1_1_话题分割_user.jinja2")

    def segment(
        self,
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None = None,
    ) -> tuple[list[TopicSegment], dict[str, dict], list[dict]]:
        """话题分割，返回 `(segments, chunk_outputs, subtitle_knowledge)`。

        `chunk_outputs` 键为 `chunk_{idx}`，值为该 chunk 的 schema 校验后
        result dict（供整合文件「输出」节落盘）。
        `subtitle_knowledge` 为各块「字幕知识」聚合去重后的列表（黑话→字幕内
        解释），供 2_2 词表判定作 [DATABASE] 输入。

        1_2 片段合并只复查**跨块切口**的相邻重要段对（块内连贯性由本步保证，
        1_1 的 prompt 已按「宁粗勿细」吸收合并判据），因此不再需要向外暴露
        块切口行号。
        """
        with step_logger("第1_1步-话题分割"):
            if not subtitle_l:
                return [], {}, []

            chunks: list[tuple[int, list[dict]]] = []
            for i in range(0, len(subtitle_l), SEGMENT_CHUNK_SIZE):
                chunk = subtitle_l[i:i + SEGMENT_CHUNK_SIZE]
                chunks.append((len(chunks), chunk))
            total_chunks = len(chunks)
            logger.info(
                "字幕长度 %d，分为 %d 块（块大小 %d）",
                len(subtitle_l), total_chunks, SEGMENT_CHUNK_SIZE,
            )

            # 各块 [DATA] 按本块行 ± N_REDUNDANT_ROWS 冗余切片（在 _segment_chunk
            # 内按块渲染）：本步的工作范围就是本块，给整场只是白发 6 倍字幕。
            # 保留 [ROWID]——冗余行使 [DATA] ⊃ 工作范围，须显式声明本块负责哪段。

            # 并行执行由全局共享线程池承载，并发上限由 ResourceManager 统一管控
            chunk_results: list[tuple[int, list[dict], dict | None, list[dict]]] = []
            futures = fanout("1_1", [
                (
                    self._segment_chunk,
                    (idx, chunk, subtitle_l, base_dir),
                    {},
                )
                for idx, chunk in chunks
            ])
            for future in futures:
                chunk_results.append(future.result())
            chunk_results.sort(key=lambda x: x[0])

            # 聚合所有 chunk 的话题段（每个 chunk 内部已校验 + 修复）
            chunk_outputs: dict[str, dict] = {
                f"chunk_{idx}": result
                for idx, _, result, _ in chunk_results
                if result is not None
            }
            all_segs: list[TopicSegment] = []
            for _, segs, _, _ in chunk_results:
                all_segs.extend(segs)
            all_segs.sort(key=lambda x: x.开始行号)

            # 聚合「字幕知识」：词必须是全文字幕连续子串，按词去重
            full_text = "\n".join(
                sub.text for sub in subtitle_l if sub.text
            )
            subtitle_knowledge: list[dict] = []
            seen_words: set[str] = set()
            for _, _, _, knowledge in chunk_results:
                subtitle_knowledge.extend(
                    clean_subtitle_knowledge(knowledge, full_text, seen_words)
                )
            if subtitle_knowledge:
                logger.info("「1_1 字幕知识」聚合 %d 条", len(subtitle_knowledge))

            # 跨 chunk 整体校验
            all_row_ids = sorted({int(item.rowID) for item in subtitle_l if item.rowID is not None})
            residual = _validate_chunk_segments(all_row_ids, all_segs)
            if residual:
                logger.warning(
                    "整体聚合后仍存在问题，触发兜底修复：%s",
                    "; ".join(residual),
                )
                all_segs = _autofix_chunk_segments(all_row_ids, all_segs)

            confused = [s for s in all_segs if s.困惑]
            logger.info(
                "总计 %d 个话题段（困惑 %d 段，将由 3_2 话题重置复核）",
                len(all_segs), len(confused),
            )
            for seg in confused:
                logger.info(
                    "「1_1 困惑」段 %d-%d（%s）：%s",
                    seg.开始行号, seg.结束行号,
                    seg.话题标签, seg.困惑原因,
                )
            return all_segs, chunk_outputs, subtitle_knowledge

    def _segment_chunk(
        self,
        chunk_idx: int,
        chunk: list[SubtitleRow],
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None,
    ) -> tuple[int, list[TopicSegment], dict | None, list[dict]]:
        """对单个 chunk 调一次 LLM 拿到话题段，并做校验 + 重试 + 自动修复。

        返回 `(chunk_idx, segments, last_result, last_knowledge)`：`last_result`
        是最后一次成功调用的 schema 校验后 result dict（调用全部失败为 None），
        供整合落盘；`last_knowledge` 是最后一次成功调用输出的「字幕知识」原始列表。

        `[DATA]` 是**本块行 ± N_REDUNDANT_ROWS 冗余**的切片，本块负责的范围由
        `[ROWID]` 给出（冗余行让 [DATA] 大于工作范围，必须显式声明）。冗余的作用
        是让块首/块尾的段能看到上下文——切口处的话题往往跨块延续，没有前后文
        会把半句话切成独立话题段。模型仍可能输出越界的段，故校验前先把段裁到
        本块范围内（`_clip_segments_to_scope`）。
        """
        chunk_row_ids = sorted({int(item.rowID) for item in chunk if item.rowID is not None})
        if not chunk_row_ids:
            return chunk_idx, [], None, []

        row_id_start = chunk_row_ids[0]
        row_id_end = chunk_row_ids[-1]
        # 本块 ± 冗余行的切片；[ANNOTATION] 切到与 [DATA] 相同行集
        # （本阶段尚无逐行注释，实际恒为空 → 整块不出现）
        subtitle_text = render_windowed_subtitle(subtitle_l, chunk_row_ids)
        annotation_text = render_annotations(
            subtitle_l, row_scope=window_row_ids(subtitle_l, chunk_row_ids),
        )
        user_prompt = self._user_template.render(
            task_rules=self._prompt,
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
            rowid_scope=format_range_scope(row_id_start, row_id_end),
        )

        last_segments: list[TopicSegment] = []
        last_errors: list[str] = []
        last_result: dict | None = None
        last_knowledge: list[dict] = []

        for attempt in range(1, SEGMENT_MAX_ATTEMPTS + 1):
            log_dir = get_step_output_dir(base_dir, 1) if base_dir is not None else None
            try:
                result = self._llm_client.call(
                    user_prompt=user_prompt,
                    temperature=0.0,
                    schema=self._schema,
                    log_dir=log_dir,
                    log_name=f"1_1_话题分割_chunk_{chunk_idx}",
                    step_name=f"1_1_话题分割_chunk_{chunk_idx}",
                )
            except Exception as e:
                logger.warning(
                    "第 %d 块 LLM 调用失败（第 %d 次）：%s",
                    chunk_idx, attempt, e,
                )
                continue

            last_result = result
            last_knowledge = (
                result.get("字幕知识", []) if isinstance(result, dict) else []
            )
            raw_segments = result.get("话题分割", []) if isinstance(result, dict) else []
            cleaned = _clean_segments(raw_segments)
            cleaned = _clip_segments_to_scope(
                cleaned, row_id_start, row_id_end, f"3_2 第 {chunk_idx} 块",
            )
            errors = _validate_chunk_segments(chunk_row_ids, cleaned)

            last_segments = cleaned
            last_errors = errors

            if not errors:
                logger.info(
                    "第 %d 块 (%d-%d) → %d 段 ✓",
                    chunk_idx, row_id_start, row_id_end, len(cleaned),
                )
                break

            logger.warning(
                "第 %d 块 (%d-%d) 第 %d 次校验失败：%s",
                chunk_idx, row_id_start, row_id_end, attempt, "; ".join(errors),
            )

        if last_errors:
            fixed = _autofix_chunk_segments(chunk_row_ids, last_segments)
            residual = _validate_chunk_segments(chunk_row_ids, fixed)
            if residual:
                logger.warning(
                    "第 %d 块 自动修复后仍存在问题：%s",
                    chunk_idx, "; ".join(residual),
                )
            else:
                logger.warning(
                    "第 %d 块 (%d-%d) 自动修复完成 → %d 段",
                    chunk_idx, row_id_start, row_id_end, len(fixed),
                )
            last_segments = fixed

        return chunk_idx, last_segments, last_result, last_knowledge


def _clean_segments(raw_segments: Any) -> list[TopicSegment]:
    """规范化 LLM 返回的话题段：转 int、修正 s>e、按 开始行号 排序。"""
    cleaned: list[TopicSegment] = []
    if not isinstance(raw_segments, list):
        return cleaned
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        try:
            s = int(seg.get("开始行号"))
            e = int(seg.get("结束行号"))
        except (TypeError, ValueError):
            continue
        if s > e:
            s, e = e, s
        cleaned.append(_normalize_segment(seg, s, e))
    cleaned.sort(key=lambda x: x.开始行号)
    return cleaned


def _normalize_segment(seg: dict, start: int, end: int) -> TopicSegment:
    """规范化单个话题段为 :class:`TopicSegment`（`_clean_segments` 的段构造入口）。

    **必须是段从 raw dict 转模型的唯一入口**：段的字段集会随管线演进增长（`事件`
    之后又加了 `困惑` / `困惑原因`），各处手写字面量的写法已经漏过一次——
    集中一处后新增字段只需改这里与 :class:`TopicSegment`。

    `困惑原因` 只在 `困惑=true` 时保留：模型可能在 false 时也塞进原因文本，
    留着会让下游"有原因即困惑"的读法出错。
    """
    confused = bool(seg.get("困惑", False))
    return TopicSegment(
        开始行号=start,
        结束行号=end,
        话题标签=str(seg.get("话题标签", "")).strip() or "未命名",
        事件=str(seg.get("事件", "") or "").strip(),
        是否重要=bool(seg.get("是否重要", True)),
        困惑=confused,
        困惑原因=str(seg.get("困惑原因", "") or "").strip() if confused else "",
    )


def _clip_segments_to_scope(
    segments: list[TopicSegment],
    scope_start: int,
    scope_end: int,
    label: str,
) -> list[TopicSegment]:
    """把话题段裁进 `[ROWID]` 范围：整段越界的丢弃，跨界的收缩到边界。

    模型看到的是完整字幕，可能顺着话题一路切到邻块去。若不裁剪，越界段会带着
    邻块的行号进入聚合结果，与邻块自己的输出重叠。
    """
    kept: list[TopicSegment] = []
    dropped = 0
    clipped = 0
    for seg in segments:
        s, e = seg.开始行号, seg.结束行号
        if e < scope_start or s > scope_end:
            dropped += 1
            continue
        new_s = max(s, scope_start)
        new_e = min(e, scope_end)
        if (new_s, new_e) != (s, e):
            clipped += 1
        kept.append(seg.model_copy(update={"开始行号": new_s, "结束行号": new_e}))
    if dropped or clipped:
        logger.warning(
            "[ROWID 校验] %s：话题段越界——丢弃 %d 段、收缩 %d 段（范围 %d-%d）",
            label, dropped, clipped, scope_start, scope_end,
        )
    return kept


def _validate_chunk_segments(
    chunk_row_ids: list[int], segments: list[TopicSegment],
) -> list[str]:
    """校验单个 chunk 的话题段。返回错误消息列表（空列表代表通过）。

    规则：
    1. 行号真实：每段 开始行号 / 结束行号 ∈ chunk_row_ids。
    2. 不重叠：按 开始行号 排序后，第 i+1 段 开始行号 严格 > 第 i 段 结束行号。
    3. 全覆盖：chunk_row_ids 中每个 rowID 都被某段覆盖。
    """
    errors: list[str] = []
    if not chunk_row_ids:
        return errors

    if not segments:
        errors.append("话题段为空")
        return errors

    row_id_set = set(chunk_row_ids)

    # 1. 行号真实
    for seg in segments:
        if seg.开始行号 not in row_id_set:
            errors.append(f"开始行号 {seg.开始行号} 不在字幕中")
        if seg.结束行号 not in row_id_set:
            errors.append(f"结束行号 {seg.结束行号} 不在字幕中")
        if seg.开始行号 > seg.结束行号:
            errors.append(f"段反向 [{seg.开始行号}, {seg.结束行号}]")

    # 2. 不重叠
    for prev, curr in zip(segments, segments[1:]):
        if curr.开始行号 <= prev.结束行号:
            errors.append(
                f"段重叠 [{prev.开始行号}, {prev.结束行号}] 与 "
                f"[{curr.开始行号}, {curr.结束行号}]"
            )

    # 3. 全覆盖
    uncovered: list[int] = []
    for rid in chunk_row_ids:
        if not any(seg.开始行号 <= rid <= seg.结束行号 for seg in segments):
            uncovered.append(rid)
    if uncovered:
        sample = uncovered[:5]
        errors.append(
            f"未覆盖 rowID {len(uncovered)} 个（如 {sample}）"
        )

    return errors


def _autofix_chunk_segments(
    chunk_row_ids: list[int], segments: list[TopicSegment],
) -> list[TopicSegment]:
    """自动修复单个 chunk 的话题段，使其满足校验规则。

    步骤：
    1. snap：把不在 chunk_row_ids 的 开始/结束行号 就近吸附到最近的真实 rowID。
    2. 去除反向段后按 开始行号 排序。
    3. 解重叠：把后段 开始行号 推到前段 结束行号 之后的下一个真实 rowID；若推不动则丢弃。
    4. 填补未覆盖 rowID：把缺口扩到相邻段的边界，或新建 "未分类" 段。
    """
    if not chunk_row_ids:
        return []

    sorted_ids = chunk_row_ids  # 已排序
    row_id_set = set(sorted_ids)

    def _snap(rid: int) -> int:
        if rid in row_id_set:
            return rid
        idx = bisect.bisect_left(sorted_ids, rid)
        candidates: list[int] = []
        if idx < len(sorted_ids):
            candidates.append(sorted_ids[idx])
        if idx > 0:
            candidates.append(sorted_ids[idx - 1])
        return min(candidates, key=lambda x: abs(x - rid))

    def _next_id(rid: int) -> int | None:
        idx = bisect.bisect_right(sorted_ids, rid)
        return sorted_ids[idx] if idx < len(sorted_ids) else None

    # Step 1: snap 越界 rowID
    snapped: list[TopicSegment] = []
    for seg in segments or []:
        try:
            s = int(seg.开始行号)
            e = int(seg.结束行号)
        except (TypeError, ValueError):
            continue
        s = _snap(s)
        e = _snap(e)
        if s > e:
            s, e = e, s
        snapped.append(seg.model_copy(update={"开始行号": s, "结束行号": e}))
    snapped.sort(key=lambda x: x.开始行号)

    # Step 2: 解重叠
    deduped: list[TopicSegment] = []
    for seg in snapped:
        if not deduped:
            deduped.append(seg)
            continue
        prev = deduped[-1]
        if seg.开始行号 <= prev.结束行号:
            new_start = _next_id(prev.结束行号)
            if new_start is None or new_start > seg.结束行号:
                # 后段被前段完全吞掉 → 丢弃后段
                continue
            # 只挪开始行号，其余字段整体保留（旧实现在这里手写字面量，
            # 漏掉了 `事件`，事件名在自动修复路径上静默丢失）
            seg = seg.model_copy(update={"开始行号": new_start})
        deduped.append(seg)

    # Step 3: 填补未覆盖 rowID
    if not deduped:
        return [_normalize_segment(
            {"话题标签": "未分类"}, sorted_ids[0], sorted_ids[-1],
        )]

    filled: list[TopicSegment] = []
    cursor_idx = 0  # 指向下一个待覆盖的真实 rowID 在 sorted_ids 中的索引

    for seg in deduped:
        seg_start_idx = bisect.bisect_left(sorted_ids, seg.开始行号)
        seg_end_idx = bisect.bisect_right(sorted_ids, seg.结束行号) - 1

        # 段前的 gap
        if cursor_idx < seg_start_idx:
            gap_start = sorted_ids[cursor_idx]
            gap_end = sorted_ids[seg_start_idx - 1]
            if filled:
                # 把缺口并入前一段
                filled[-1] = filled[-1].model_copy(update={"结束行号": gap_end})
            else:
                # 没有前段，把缺口并入当前段（扩前界）
                seg = seg.model_copy(update={"开始行号": gap_start})
                seg_start_idx = cursor_idx

        filled.append(seg)
        cursor_idx = seg_end_idx + 1

    # 末尾 gap
    if cursor_idx < len(sorted_ids):
        tail_end = sorted_ids[-1]
        if filled:
            filled[-1] = filled[-1].model_copy(update={"结束行号": tail_end})
        else:
            filled.append(_normalize_segment(
                {"话题标签": "未分类"}, sorted_ids[cursor_idx], tail_end,
            ))

    return filled
