"""3_2 话题重置：对 1_1 自报困惑的话题段重新判定分割方式。

1_1 是在未纠错字幕上切分的，它标 `困惑=true` 的段就是它明确承认"没读懂这段在讲
什么"的段。读不懂就切不准：不知道代词指什么就判不出事件弧归属，不知道某词是否
ASR 误识就会把同一话题误切成两段。本阶段在**已纠错 + 指代已内联**的字幕上重来
一次——1_1 当初读不懂的点此时多半已有答案。

## 工作单元：困惑段 ± 2 个话题，且必须先聚成不重叠单元

每个困惑段带前后各 `CONTEXT_SEGMENTS` 个话题作为可调整范围（正确边界常落在邻段
身上：1_1 把本该属于前段的内容切给了困惑段，只准动困惑段就修不好）。

但**相邻困惑段的窗口会重叠**——两次调用同时改写同一批行，结果必然冲突（谁后写谁
生效，且两边各自保证"完全覆盖"的区间不同，合起来可能重叠或留洞）。故先用纯 Python
把窗口相交的困惑段**并成一个单元**，保证每一行只有一个写者。单元之间因此互不重叠、
可安全并行。

## 全量重制输出（1_1 的重制版）

3_2 现在是 1_1 话题分割的**重制版**：模型在已纠错、指代已内联的字幕上，对每个
困惑单元**重新输出完整的话题段列表**（`新话题列表`），每段含 `开始行号`/`结束行号`/
`话题标签`/`事件`/`是否重要`/`困惑`/`困惑原因`——字段与 1_1 完全一致，外加
`是否需要修正` 与 `修正说明`。输出经校验（行号真实 + 段间不重叠 + 全覆盖），
不合格打回重跑（携带上轮校验错误），仍不合格才自动修复兜底、最终保留原分割。
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path

from ...llm_infra.resource import fanout
from ...log_config import step_logger
from ..schemas import SubtitleRow, TopicSegment
from ..topic_segmentation.topic_segmenter import (
    SEGMENT_MAX_ATTEMPTS,
    _autofix_chunk_segments,
    _clean_segments,
    _clip_segments_to_scope,
    _validate_chunk_segments,
)
from ..utils.llm_text_utils import (
    render_annotations,
    render_windowed_subtitle,
    window_row_ids,
)
from ..utils.rowid_scope import format_scope
from .refine_stages import RefineStages

logger = logging.getLogger(__name__)

# 困惑段前后各带多少个话题段作为可调整的上下文
CONTEXT_SEGMENTS = 2
# 单个重置单元的行数保护上限：超过则放弃该单元（沿用 1_2 合并组的同一口径）
MAX_RESET_UNIT_ROWS = 1000


def build_reset_units(segments: list[TopicSegment]) -> list[list[int]]:
    """把困惑段聚成互不重叠的工作单元，返回每个单元的段下标列表（连续区间）。

    每个困惑段取 `[i-CONTEXT_SEGMENTS, i+CONTEXT_SEGMENTS]` 窗口，窗口相交或相邻
    即并组。**这一步是 3_2 并行安全的前提**：不并组的话，两个相邻困惑段的窗口会
    覆盖同一批行，两次调用各自输出"完全覆盖自己区间"的段列表，合并时必然重叠。

    Returns: `[[段下标...], ...]`，每个内层列表是一段**连续**的段下标（升序），
        单元之间不相交且按顺序排列。
    """
    windows: list[tuple[int, int]] = []
    for i, seg in enumerate(segments):
        if not seg.困惑:
            continue
        lo = max(0, i - CONTEXT_SEGMENTS)
        hi = min(len(segments) - 1, i + CONTEXT_SEGMENTS)
        windows.append((lo, hi))
    if not windows:
        return []

    merged: list[tuple[int, int]] = [windows[0]]
    for lo, hi in windows[1:]:
        prev_lo, prev_hi = merged[-1]
        # 相交或相邻（hi + 1 == lo）都并组：相邻窗口若分开处理，两个单元的边界段
        # 会各自被两次调用当成"自己区间的端点"，切点判定互相打架
        if lo <= prev_hi + 1:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return [list(range(lo, hi + 1)) for lo, hi in merged]


class TopicResetter:
    """3_2 话题重置执行器：困惑段聚类 → 并行调用 → 校验 → 段列表重写。"""

    def __init__(self, stages: RefineStages | None = None) -> None:
        self._stages = stages or RefineStages()

    def reset(
        self,
        segments: list[TopicSegment],
        subtitle_l: list[SubtitleRow],
        confirmed_jargon: list[dict],
        base_dir: Path | None,
    ) -> tuple[list[TopicSegment], dict[str, dict]]:
        """话题重置主入口，返回 `(修正后 segments, unit_outputs)`。

        `confirmed_jargon` 是第 2 步已确证黑话（含 `困惑=true` 的未解析项），按
        单元行号范围过滤后渲染进 3_2 的 [DATABASE]——1_1 当初读不懂的词，答案
        大多在这份结论里，不给 3_2 就会重蹈「电梯PCB 已知是胜宏科技却仍标困惑」。

        无困惑段时原样返回（零 LLM 调用）。单元失败/校验不过时该单元保留原分割。
        """
        with step_logger("第3_2步-话题重置"):
            units = build_reset_units(segments)
            confused_total = sum(1 for s in segments if s.困惑)
            if not units:
                logger.info(
                    "「3_2 话题重置」跳过：1_1/1_2 无困惑段（共 %d 段全部读懂）",
                    len(segments),
                )
                return list(segments), {}

            logger.info(
                "「3_2 话题重置」%d 个困惑段 → 聚成 %d 个不重叠单元"
                "（每段带前后各 %d 个上下文话题），单元间并行",
                confused_total, len(units), CONTEXT_SEGMENTS,
            )

            row_ids_sorted = sorted(
                int(s.rowID) for s in subtitle_l if s.rowID is not None
            )
            futures = fanout("3_2", [
                (
                    self._run_unit,
                    (
                        u_idx, idxs, segments, subtitle_l, row_ids_sorted,
                        confirmed_jargon, base_dir,
                    ),
                    {},
                )
                for u_idx, idxs in enumerate(units)
            ])

            unit_outputs: dict[str, dict] = {}
            replacements: dict[tuple[int, int], list[TopicSegment]] = {}
            revised_units = 0
            for u_idx, (future, idxs) in enumerate(zip(futures, units)):
                try:
                    result, new_segs, revised = future.result()
                except Exception as e:  # noqa: BLE001 —— 单元失败不拖垮整步
                    logger.warning(
                        "「3_2 话题重置 u%d」失败：%s——本单元保留 1_1 原分割",
                        u_idx, e,
                    )
                    continue
                if result is not None:
                    unit_outputs[f"unit_{u_idx}"] = result
                if new_segs is None:
                    continue
                replacements[(idxs[0], idxs[-1])] = new_segs
                revised_units += int(revised)

            out = _apply_replacements(segments, replacements)
            logger.info(
                "「3_2 话题重置」完成：%d 单元中 %d 个判定需修正；"
                "段数 %d → %d，剩余困惑段 %d 个",
                len(units), revised_units, len(segments), len(out),
                sum(1 for s in out if s.困惑),
            )
            return out, unit_outputs

    def _run_unit(
        self,
        unit_idx: int,
        idxs: list[int],
        segments: list[TopicSegment],
        subtitle_l: list[SubtitleRow],
        row_ids_sorted: list[int],
        confirmed_jargon: list[dict],
        base_dir: Path | None,
    ) -> tuple[dict | None, list[TopicSegment] | None, bool]:
        """跑一个重置单元，返回 `(result, 新段列表 | None, 是否判定需修正)`。

        `新段列表 is None` 表示本单元保留原分割（重跑与自动修复后仍不合规，或
        模型未产出有效段）。输出是全量重制：解析 `新话题列表` → 清洗 → 裁回单元
        区间 → 校验（行号真实 + 段间不重叠 + 全覆盖），不合格打回重跑（把上一轮
        校验错误回灌），仍不合格走自动修复兜底，最终不合规则保留原分割。
        """
        unit_segs = [segments[i] for i in idxs]
        start = int(unit_segs[0].开始行号)
        end = int(unit_segs[-1].结束行号)
        # 单元负责区间内**真实存在**的 rowID（覆盖校验的基准；rowID 有缺号）
        lo = bisect.bisect_left(row_ids_sorted, start)
        hi = bisect.bisect_right(row_ids_sorted, end)
        unit_row_ids = row_ids_sorted[lo:hi]
        if not unit_row_ids:
            logger.warning("「3_2 u%d」区间 %d-%d 无有效行，跳过", unit_idx, start, end)
            return None, None, False
        if end - start > MAX_RESET_UNIT_ROWS:
            logger.warning(
                "「3_2 u%d」单元 %d-%d（%d 行）超过 %d 行保护上限，"
                "放弃重置、保留原分割",
                unit_idx, start, end, end - start, MAX_RESET_UNIT_ROWS,
            )
            return None, None, False

        # 已确证黑话按本单元行号过滤（[DATABASE] 只给本窗口相关的结论）
        unit_row_id_set = set(unit_row_ids)
        unit_confirmed = [
            c for c in confirmed_jargon
            if c.get("rowID") is not None and int(c["rowID"]) in unit_row_id_set
        ]

        subtitle_text = render_windowed_subtitle(subtitle_l, unit_row_ids)
        annotation_text = render_annotations(
            subtitle_l, row_scope=window_row_ids(subtitle_l, unit_row_ids),
        )
        # 当前分割作为参考上下文（全量重制不靠段序号点名某段）
        spec = [
            {
                "开始行号": int(segments[i].开始行号),
                "结束行号": int(segments[i].结束行号),
                "话题标签": segments[i].话题标签,
                "事件": str(segments[i].事件 or "").strip(),
                "是否重要": bool(segments[i].是否重要),
                "困惑": bool(segments[i].困惑),
                "困惑原因": str(segments[i].困惑原因 or "").strip(),
            }
            for i in idxs
        ]
        scope = format_scope(unit_row_ids)

        # 校验失败打回重跑：第 2 次起把上一轮校验错误回灌给模型，让它修正
        result: dict | None = None
        new_segs: list[TopicSegment] | None = None
        last_errors: list[str] = []
        for attempt in range(1, SEGMENT_MAX_ATTEMPTS + 1):
            retry_hint = "；".join(last_errors) if attempt > 1 else ""
            try:
                result = self._stages.run_3_2(
                    unit_idx, spec, scope,
                    subtitle_text, annotation_text, unit_confirmed, base_dir,
                    retry_hint=retry_hint,
                )
            except Exception as e:  # noqa: BLE001 —— 单次调用失败可重试
                logger.warning("「3_2 u%d」第 %d 次调用失败：%s", unit_idx, attempt, e)
                last_errors = [f"调用失败：{e}"]
                continue

            if not isinstance(result, dict):
                last_errors = ["LLM 返回非 dict 结果"]
                continue

            cleaned = _clean_segments(result.get("新话题列表"))
            cleaned = _clip_segments_to_scope(
                cleaned, unit_row_ids[0], unit_row_ids[-1], f"3_2 u{unit_idx}",
            )
            last_errors = _validate_chunk_segments(unit_row_ids, cleaned)
            if not last_errors:
                new_segs = cleaned
                break
            logger.warning(
                "「3_2 u%d」第 %d 次校验失败（%s），打回重跑",
                unit_idx, attempt, "; ".join(last_errors),
            )

        if new_segs is None:
            # 重跑后仍不合规：先确认模型是否产出过任何有效段，否则保留原分割
            last_cleaned: list[TopicSegment] = []
            if isinstance(result, dict):
                last_cleaned = _clip_segments_to_scope(
                    _clean_segments(result.get("新话题列表")),
                    unit_row_ids[0], unit_row_ids[-1], f"3_2 u{unit_idx}",
                )
            if not last_cleaned:
                logger.warning(
                    "「3_2 u%d」重跑 %d 次后未产出有效段列表，保留原分割",
                    unit_idx, SEGMENT_MAX_ATTEMPTS,
                )
                return result, None, False
            fixed = _autofix_chunk_segments(unit_row_ids, last_cleaned)
            residual = _validate_chunk_segments(unit_row_ids, fixed)
            if residual:
                logger.warning(
                    "「3_2 u%d」重跑 %d 次并自动修复后仍不合规（%s），保留原分割",
                    unit_idx, SEGMENT_MAX_ATTEMPTS, "; ".join(residual[:3]),
                )
                return result, None, False
            new_segs = fixed
            logger.warning("「3_2 u%d」自动修复完成 → %d 段", unit_idx, len(fixed))

        revised = bool(result.get("是否需要修正")) if result is not None else False
        if not revised:
            logger.info(
                "「3_2 u%d」段 %d-%d：判定 1_1 分割无误（%d 段，困惑字段已更新）",
                unit_idx, idxs[0], idxs[-1], len(new_segs),
            )
        else:
            logger.info(
                "「3_2 u%d」段 %d-%d：判定需修正，%d 段 → %d 段；%s",
                unit_idx, idxs[0], idxs[-1], len(unit_segs), len(new_segs),
                str(result.get("修正说明", ""))[:80],
            )
        return result, new_segs, revised


def _apply_replacements(
    segments: list[TopicSegment],
    replacements: dict[tuple[int, int], list[TopicSegment]],
) -> list[TopicSegment]:
    """把各单元的新段列表替换回全局段列表（单元互不重叠，按下标区间替换）。"""
    if not replacements:
        return list(segments)
    out: list[TopicSegment] = []
    i = 0
    by_start = {lo: (hi, segs) for (lo, hi), segs in replacements.items()}
    while i < len(segments):
        if i in by_start:
            hi, new_segs = by_start[i]
            out.extend(s.model_copy() for s in new_segs)
            i = hi + 1
            continue
        out.append(segments[i].model_copy())
        i += 1

    # 全局不变量：替换后仍须互不重叠且升序（单元不重叠 + 单元内已校验 ⇒ 必然成立，
    # 写一道 assert 保险，与 1_2 合并后的同款检查一致）
    for prev, curr in zip(out, out[1:]):
        assert curr.开始行号 > prev.结束行号, (
            f"3_2 重置后段重叠：{prev} vs {curr}"
        )
    return out
