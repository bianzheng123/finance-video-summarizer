"""话题片段合并（流水线第 1.2 小步）。

1_1 分块并行把字幕切成完全覆盖的话题段。块内切分由 1_1 自己保证连贯
（1_1 的 prompt 已按「宁粗勿细」吸收合并判据）；本步只对**跨块边界的相邻
重要段对**调一次 LLM——块与块是两次独立调用，切口处最可能被错切（同一话题/
同一事件弧被分到两块的边界两侧）。被标 merge 的段对做桥接合并：「不重要」段
（观众互动/广告/寒暄）不参与判定，但夹在两重要段之间时不阻断配对——两侧判定
合并则中间段被吸收（行由组剪枝按合并前段列表标注「无关」）。

每对的 `[DATA]` 都是同一份完整字幕（供前缀缓存命中），两段的行号区间由
`[ROWID]` 给出，模型自己在全文里读衔接处。

输入：1_1 输出的 segments（已校验：互不重叠 + 全覆盖 + rowID 真实）+ 字幕。
输出：合并后的 segments（边界 rowID 仍来自原字幕；段间仍互不重叠且全覆盖）。
"""

import logging
from pathlib import Path

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
from .topic_segmenter import SEGMENT_CHUNK_SIZE, clean_subtitle_knowledge

logger = logging.getLogger(__name__)

# 桥接合并组的行数保护上限：超过则取消该组合并（防极端链式合并产生超大工作单元）
MAX_MERGED_SEGMENT_ROWS = 1000
# 桥接间隔上限：两重要段之间夹着的行数超过此值时视为"讨论已被打断转向"，
# 不判定、不合并（确定性 Python 门，防带货/吐槽长间隔把两个话题粘成巨型段）
MAX_BRIDGE_ROWS = 50


class SegmentMerger:
    """对相邻重要段对调 LLM 判定是否合并，按合并决策做桥接合并。"""

    def __init__(self) -> None:
        self._prompt = load_prompt("1_2_片段合并.txt", __file__)
        self._schema = _load_schema("1_2_片段合并", __file__)
        self._llm_client = LLMClient()
        _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("1_2_片段合并_user.jinja2")

    def merge(
        self,
        segments: list[TopicSegment],
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None = None,
    ) -> tuple[list[TopicSegment], dict[str, dict], list[dict]]:
        """片段合并，返回 `(合并后 segments, pair_outputs, subtitle_knowledge)`。

        `pair_outputs` 键为 `pair_{pair_idx:03d}`（与 log_name 后缀一致），
        值为该段对的 schema 校验后 result dict（供整合文件「输出」节落盘）。
        `subtitle_knowledge` 为各段对「字幕知识」聚合去重后的列表（补 1_1 分块
        漏网的跨块解释）。
        """
        with step_logger("第1_2步-片段合并"):
            if len(segments) <= 1:
                return list(segments), {}, []

            # 各对相邻段的 [DATA] 按「两段区间 + 桥接间隔」± 冗余行切片
            # （在 _judge_pair 内按对渲染）：判定只需要衔接处的语境。

            # 只判定「跨 chunk 边界的相邻重要段对」：1_1 分块并行分割，块内连贯性由
            # 1_1 自身保证（prompt 已按「宁粗勿细」吸收合并判据）；块与块是两次
            # 独立调用，切口两侧最可能错切。块内相邻对一律跳过。
            # 「不重要」段（观众互动/广告/寒暄）不参与判定，但夹在两重要段之间时
            # 不阻断配对——若两侧判定合并，中间段被桥接吸收（其行号区间并入合并段，
            # 行由组剪枝按合并前的段列表标注「无关」）。
            # 间隔超过 MAX_BRIDGE_ROWS 行的对视为讨论已被打断转向，不判定。
            important_idx = [
                idx for idx, seg in enumerate(segments)
                if seg.是否重要 is not False
            ]
            # 行号 → 字幕位置映射（chunk 按字幕行位置切分，行号不连续）
            rid_pos: dict[int, int] = {
                int(s.rowID): i for i, s in enumerate(subtitle_l)
                if s.rowID is not None
            }
            chunk_bounds = range(
                SEGMENT_CHUNK_SIZE, len(subtitle_l), SEGMENT_CHUNK_SIZE,
            )
            judged_pairs: list[tuple[int, int, dict, dict]] = []
            skipped_gap_pairs = 0
            skipped_inner_pairs = 0
            for idx_a, idx_b in zip(important_idx, important_idx[1:]):
                gap_rows = (
                    int(segments[idx_b].开始行号)
                    - int(segments[idx_a].结束行号)
                    - 1
                )
                if gap_rows > MAX_BRIDGE_ROWS:
                    skipped_gap_pairs += 1
                    continue
                a_end_pos = rid_pos[int(segments[idx_a].结束行号)]
                b_start_pos = rid_pos[int(segments[idx_b].开始行号)]
                if not any(a_end_pos < bound <= b_start_pos for bound in chunk_bounds):
                    skipped_inner_pairs += 1
                    continue
                judged_pairs.append((idx_a, idx_b, segments[idx_a], segments[idx_b]))
            # 一条日志说清全部账目：段数 → 段对数 → 各去向，避免把「段/段对」读成
            # 「chunk」（chunk 是 1_1 的输入切块单位，与段数无关，故一并打出块数）
            total_pairs = len(important_idx) - 1 if important_idx else 0
            logger.info(
                "输入 %d 行切 %d 块（每块 %d 行）；1_1 产出 %d 段（重要 %d 个）"
                "→ 相邻重要段对 %d 对：块内 %d 对跳过（1_1 保证块内连贯）、"
                "间隔>%d 行 %d 对跳过、跨块接缝 %d 对需 LLM 判定",
                len(subtitle_l), len(chunk_bounds) + 1, SEGMENT_CHUNK_SIZE,
                len(segments), len(important_idx), total_pairs,
                skipped_inner_pairs, MAX_BRIDGE_ROWS, skipped_gap_pairs,
                len(judged_pairs),
            )

            # 并行执行由全局共享线程池承载，并发上限由 ResourceManager 统一管控
            decisions: dict[tuple[int, int], dict] = {}  # (idx_a, idx_b) → pair result
            pair_outputs: dict[str, dict] = {}
            raw_knowledge: list[dict] = []
            futures = fanout("1_2", [
                (
                    self._judge_pair,
                    (
                        pair_idx, idx_a, idx_b, seg_a, seg_b,
                        subtitle_l, base_dir,
                    ),
                    {},
                )
                for pair_idx, (idx_a, idx_b, seg_a, seg_b) in enumerate(judged_pairs)
            ])
            for future, (pair_idx, (idx_a, idx_b, _seg_a, _seg_b)) in zip(
                futures, enumerate(judged_pairs)
            ):
                result, knowledge = future.result()
                if result is not None:
                    pair_outputs[f"pair_{pair_idx:03d}"] = result
                    decisions[(idx_a, idx_b)] = result
                raw_knowledge.extend(knowledge or [])

            merged = _apply_merge_decisions(segments, decisions)
            logger.info(
                "应用合并：%d 段 → %d 段（标 merge 的段对 %d）；困惑段 %d 个"
                "（合并后口径，将交 3_2 话题重置）",
                len(segments), len(merged),
                sum(1 for v in decisions.values() if v.get("是否合并")),
                sum(1 for s in merged if s.困惑),
            )

            # 合并后理论上仍互不重叠 + 全覆盖；写一道 assert 保险
            for prev, curr in zip(merged, merged[1:]):
                assert curr.开始行号 > prev.结束行号, (
                    f"合并后段重叠：{prev} vs {curr}"
                )

            full_text = "\n".join(
                sub.text for sub in subtitle_l if sub.text
            )
            subtitle_knowledge = clean_subtitle_knowledge(raw_knowledge, full_text)
            if subtitle_knowledge:
                logger.info("「1_2 字幕知识」聚合 %d 条", len(subtitle_knowledge))
            return merged, pair_outputs, subtitle_knowledge

    def _judge_pair(
        self,
        pair_idx: int,
        idx_a: int,
        idx_b: int,
        seg_a: TopicSegment,
        seg_b: TopicSegment,
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None,
    ) -> tuple[dict | None, list[dict]]:
        """对单个相邻重要段对调 LLM，返回 `(result, knowledge)`。

        `result` 是 schema 校验后的 result dict（失败为 None）；`knowledge` 是
        该 pair 输出的「字幕知识」原始列表（失败为空列表）。调用失败返回 None
        （该 pair 视为不合并）。`[DATA]` 是**两段区间 + 桥接间隔 ± 冗余行**的
        切片（判定只需衔接处语境，给整场是白发字幕）；两段各自的行号区间由
        `[ROWID]` / `[TASK]` 给出。间隔行整段纳入（`extra_span_ids`，不受冗余
        半径限制）——间隔上限 MAX_BRIDGE_ROWS=50 行 > 冗余行数，靠冗余给不全，
        而判定「中间夹的是不是打断」恰恰要读完整间隔。result 含 `是否合并` 及
        合并后段的 `新话题标签`/`事件`/`是否重要`/`困惑`/`困惑原因`（不合并时
        这些字段被忽略）。
        """
        a_start, a_end = int(seg_a.开始行号), int(seg_a.结束行号)
        b_start, b_end = int(seg_b.开始行号), int(seg_b.结束行号)
        a_scope = format_range_scope(a_start, a_end)
        b_scope = format_range_scope(b_start, b_end)

        # 窗口中心=两段自身的行；间隔行整段纳入（不受冗余半径限制）
        centers = list(range(a_start, a_end + 1)) + list(range(b_start, b_end + 1))
        gap_ids = list(range(a_end + 1, b_start))
        subtitle_text = render_windowed_subtitle(
            subtitle_l, centers, extra_span_ids=gap_ids,
        )
        # 本阶段尚无逐行注释，[ANNOTATION] 实际恒为空（整块不出现）
        annotation_text = render_annotations(
            subtitle_l,
            row_scope=window_row_ids(subtitle_l, centers, extra_span_ids=gap_ids),
        )

        user_prompt = self._user_template.render(
            task_rules=self._prompt,
            seg_a_label=seg_a.话题标签,
            seg_b_label=seg_b.话题标签,
            seg_a_event=str(seg_a.事件 or "").strip(),
            seg_b_event=str(seg_b.事件 or "").strip(),
            seg_a_scope=a_scope,
            seg_b_scope=b_scope,
            gap_count=idx_b - idx_a - 1,
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
            rowid_scope=a_scope + b_scope,
        )

        log_dir = get_step_output_dir(base_dir, 1) if base_dir is not None else None
        try:
            result = self._llm_client.call(
                user_prompt=user_prompt,
                temperature=0.0,
                schema=self._schema,
                log_dir=log_dir,
                log_name=f"1_2_片段合并_pair_{pair_idx:03d}",
                step_name=f"1_2_片段合并_pair_{pair_idx:03d}",
            )
        except Exception as e:
            logger.warning(
                "第 %d 对 (%s | %s) LLM 调用失败：%s",
                pair_idx, seg_a.话题标签, seg_b.话题标签, e,
            )
            return None, []

        if not isinstance(result, dict):
            return None, []
        return result, result.get("字幕知识", [])


def _apply_merge_decisions(
    segments: list[TopicSegment],
    decisions: dict[tuple[int, int], dict],
) -> list[TopicSegment]:
    """按段对的 merge 决策做合并（含跨「不重要」段桥接）。

    decisions[(idx_a, idx_b)] 是该 pair 的 schema 校验后 result dict（含 `是否合并`
    与合并后段的 `新话题标签`/`事件`/`是否重要`/`困惑`/`困惑原因`）。连续
    `是否合并=true` 的段对把整条链合并成一段：中间被桥接的不重要段被吸收（其行号
    区间并入合并段，行由组剪枝按合并前的段列表标注「无关」）。

    合并段的字段采信链上**首个** pair 的 LLM 输出（链式合并时多个 pair 各自只看到
    相邻两段，取首个作为合并起点，避免互相矛盾）；`新话题标签` 为空时回退旧标签
    拼接。`困惑` 对组内原段与 LLM 输出取 OR——组里只要有一处没读懂，整组就该进
    3_2 复核。合并组超过 MAX_MERGED_SEGMENT_ROWS 行时取消该组合并（防极端链式合并）。
    """
    if not segments:
        return []

    merged: list[TopicSegment] = []
    i = 0
    while i < len(segments):
        if segments[i].是否重要 is False:
            # 未被桥接吸收的不重要段原样保留（组剪枝会整段标注「无关」）
            merged.append(segments[i].model_copy())
            i += 1
            continue

        group = [segments[i]]
        pair_results: list[dict] = []
        j = i
        while True:
            nxt = j + 1
            while nxt < len(segments) and segments[nxt].是否重要 is False:
                nxt += 1
            if nxt >= len(segments):
                break
            res = decisions.get((j, nxt))
            if not res or not res.get("是否合并"):
                break
            pair_results.append(res)
            group.extend(segments[j + 1:nxt + 1])  # 含中间不重要段（桥接吸收）
            j = nxt

        if len(group) == 1:
            merged.append(group[0].model_copy())
        else:
            start_row = int(group[0].开始行号)
            end_row = int(group[-1].结束行号)
            if end_row - start_row > MAX_MERGED_SEGMENT_ROWS:
                logger.warning(
                    "合并组 %d-%d（%d 行）超过 %d 行保护上限，取消合并，原样保留",
                    start_row, end_row, end_row - start_row, MAX_MERGED_SEGMENT_ROWS,
                )
                merged.extend(s.model_copy() for s in group)
            else:
                first = pair_results[0]
                labels = [
                    str(s.话题标签).strip()
                    for s in group
                    if s.是否重要 is not False
                ]
                new_label = str(first.get("新话题标签", "") or "").strip()
                events = [
                    str(s.事件 or "").strip()
                    for s in group
                    if str(s.事件 or "").strip()
                ]
                llm_event = str(first.get("事件", "") or "").strip()
                merged_event = llm_event or (events[0] if events else "")
                if llm_event and events and llm_event != events[0]:
                    logger.warning(
                        "合并组 %d-%d LLM 事件名「%s」与组内原事件「%s」不一致，采信 LLM",
                        start_row, end_row, llm_event, events[0],
                    )
                # 困惑取 OR：组里只要有一段没读懂，或 LLM 自评没读懂，整组进 3_2
                reasons = [
                    str(s.困惑原因 or "").strip()
                    for s in group
                    if s.困惑 and str(s.困惑原因 or "").strip()
                ]
                llm_confused = bool(first.get("困惑"))
                llm_reason = str(first.get("困惑原因", "") or "").strip()
                if llm_confused and llm_reason:
                    reasons.append(llm_reason)
                merged.append(TopicSegment(
                    开始行号=start_row,
                    结束行号=end_row,
                    话题标签=new_label
                        or (" | ".join(lab for lab in labels if lab))
                        or "未命名",
                    事件=merged_event,
                    是否重要=bool(first.get("是否重要", True)),
                    困惑=any(s.困惑 for s in group) or llm_confused,
                    困惑原因="；".join(dict.fromkeys(reasons)),
                ))
        i = j + 1
    return merged
