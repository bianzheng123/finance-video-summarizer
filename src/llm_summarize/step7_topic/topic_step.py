"""步骤 7 主题分析编排器：7_1 主题生成 → 收尾 → 7_2 内联引用校验。

本步是管线的最后一环，只承担主题分析与跨 section 后处理（敏感信号判定/聚合已迁入
步骤 6）：

- **跨 section 后处理**：根级引用合并 + 引用总时长 + 黑话注入 + 空壳主题剔除。
- **收尾产物**：写 `7_analysis_result.json`（本步断点）。最终
  `llm_analysis.json` 由主流程在三步 join 后合并写，**本步不再写它**。

落盘：
- `7_analysis_result.json`：本步全部主题 section（校验定稿后），断点续跑用；
- `7_1_topic_analysis.json` / `7_2_引用校验.json`：各阶段 LLM 输出整合 dump；
- `7_validation_status.json`：主题引用校验状态台账。
"""

import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ...llm_infra.batch import batch_units
from ...llm_infra.io import (
    load_dump_outputs,
    read_json,
    write_integrated,
    write_json_atomic,
)
from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema
from ...llm_infra.config import get_batch_size
from ...llm_infra.resource import fanout
from ...log_config import step_logger
from ..analysis_common import PostProcessor, collect_used_jargons
from ..analysis_common.status_ledger import write_validation_status
from ..citation_check import CitationValidator
from ..clustering.event_topics import is_keyword_event, is_segment_event
from ..schemas import SubtitleRow, TopicData
from ..utils.llm_text_utils import (
    collect_non_analysis_row_ids,
    load_prompt,
    render_annotations,
    window_row_ids,
)
from ..utils.rowid_scope import filter_citations, scope_id_set
from .topic_analyzer import TopicAnalyzer

_section_anchor = __file__

_env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)

STEP_NUMBER = 7
VALIDATION_STAGE = "7_2"

_FIXED_SECTION_NAMES = {"宏观分析", "交易技巧", "精炼总结", "主讲人暗示重要话题"}


def _summary_topics(keyword_data: dict[str, TopicData]) -> set[str]:
    """返回需要生成 section 的主题集合：非叶子节点（有子项）+ 段落事件恒总结 + 孤立关键词事件。

    叶子节点（子项为空）不独立成 section，只作为父主题的子项呈现；
    关键词事件作为叶子由父主题「事件与资产」承载。但**未被任何父主题收编的
    孤立关键词事件**（候选父为空、聚类挂载失败）没有父去承载其事件叙事，
    若不升格就会整体丢失——故把这类孤立事件也升格为独立 section，走 7_1 的
    「事件类主题规则」把完整叙事写入「事件与资产」。
    """
    mounted_children: set[str] = set()
    for t, td in keyword_data.items():
        if isinstance(td, TopicData):
            for child in td.子项 or []:
                if isinstance(child, str) and child:
                    mounted_children.add(child)

    result: set[str] = set()
    orphan_events: list[str] = []
    for t, td in keyword_data.items():
        if not isinstance(td, TopicData):
            continue
        # 普通子项 = 剔除关键词事件子项（关键词事件由父主题「事件与资产」承载，
        # 不占子项位，避免把「只有关键词事件子项」的子主题误判为独立 section）。
        regular_children = [
            c for c in (td.子项 or [])
            if isinstance(c, str) and c in keyword_data
            and not is_keyword_event(keyword_data.get(c))
        ]
        if is_segment_event(td) or regular_children:
            result.add(t)
        elif is_keyword_event(td) and t not in mounted_children:
            result.add(t)
            orphan_events.append(t)
    if orphan_events:
        logger.info(
            "孤立关键词事件（无父）升格为独立 section，共 %d 个：%s",
            len(orphan_events), orphan_events,
        )
    return result


class TopicAnalysisStep:
    """步骤 7 编排器。`analyze()` 返回含全部主题 section 的 dict。"""

    def __init__(self) -> None:
        self._llm_client = LLMClient()
        self._topic = TopicAnalyzer(
            self._llm_client,
            load_prompt("7_1_主题分析.txt", _section_anchor),
            _load_schema("7_1_主题分析", _section_anchor),
        )
        self._validator = CitationValidator(self._llm_client)

    # ==================== 小工具 ====================

    # 两个断点文件，语义不同、不可合并：
    # - `7_generated_analysis.json`：7_1 生成阶段的增量断点（主题级，完成一个落一个），
    #   命中它说明生成完了但还没校验 → 直接进 7_2 的判定/校验流程；
    # - `7_analysis_result.json`：校验定稿后的断点 → 整步跳过。
    # 只用一个文件会让「生成到一半崩溃」被误判成「已定稿」，未校验的主题直接进最终产物。
    _GENERATED_CHECKPOINT = "7_generated_analysis.json"
    _RESULT_CHECKPOINT = "7_analysis_result.json"

    @staticmethod
    def _load_checkpoint(base_dir: Path, name: str) -> dict:
        data = read_json(base_dir / name)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, (dict, list))}
        return {}

    @staticmethod
    def _postprocess_topic(name: str, analysis: dict, subtitle_l: list[SubtitleRow]) -> None:
        """per-topic 后处理：行号→时间 → 原文填充 → 观点拆分与去重。

        只处理本轮新生成的 section：缓存加载进来的 section 在自己那一轮已经做过
        行号→时间转换，再转一次会因引用里只剩时间而崩（缺开始行号）。
        """
        section_data = analysis[name]
        if not isinstance(section_data, dict) or not section_data:
            return
        analysis[name] = PostProcessor.convert_row_ids_in_section(
            section_data, subtitle_l,
        )
        PostProcessor.populate_citation_texts_in_section(analysis[name], subtitle_l)
        opinions_before = analysis[name].get("观点")
        before = len(opinions_before) if isinstance(opinions_before, list) else 0
        TopicAnalyzer.split_opinion_entries(analysis[name])
        after_split = len(analysis[name].get("观点") or [])
        TopicAnalyzer.dedup_redundant_opinions(analysis[name])
        TopicAnalyzer.dedup_bare_conclusion_opinions(analysis[name])
        after = len(analysis[name].get("观点") or [])
        if before != after:
            logger.info(
                "主题「%s」观点整理：%d → %d 条（拆分 +%d，立场覆盖去重 -%d）",
                name, before, after, max(0, after_split - before), after_split - after,
            )

    @staticmethod
    def _reapply_opinion_invariants(analysis: dict) -> None:
        """7_2 重写可能重新引入观点冗余（分号合并/立场覆盖/裸结论重复），落盘前重跑一次（确定性、幂等）。"""
        for topic_name, topic_data in analysis.items():
            if topic_name in _FIXED_SECTION_NAMES or not isinstance(topic_data, dict):
                continue
            before = len(topic_data.get("观点") or [])
            TopicAnalyzer.split_opinion_entries(topic_data)
            TopicAnalyzer.dedup_redundant_opinions(topic_data)
            TopicAnalyzer.dedup_bare_conclusion_opinions(topic_data)
            after = len(topic_data.get("观点") or [])
            if before != after:
                logger.info(
                    "主题「%s」观点不变量兜底：%d → %d 条（7_2 重写残留整理）",
                    topic_name, before, after,
                )

    # ==================== 主入口 ====================

    def analyze(
        self,
        keyword_data: dict[str, TopicData],
        base_dir: Path,
        subtitle_l: list[SubtitleRow],
    ) -> dict:
        """生成并校验全部主题 section，写 `7_analysis_result.json` 并返回本步 section。

        最终 `llm_analysis.json` 由主流程 `llm_analysis.py` 在三步 join 后合并写，
        本步不碰它（避免三步并发写同一文件）。
        """
        with step_logger("第7步-主题分析"):
            blocked_row_ids = collect_non_analysis_row_ids(subtitle_l)
            sink_7_1: dict[str, dict] = load_dump_outputs(
                base_dir / "7_1_topic_analysis.json",
            )
            sink_7_2: dict[str, dict] = {}

            analysis = self._load_checkpoint(base_dir, self._RESULT_CHECKPOINT)
            if analysis:
                logger.info(
                    "断点恢复：%s 有 %d 个 section（已合并校验），跳过本步生成与校验",
                    self._RESULT_CHECKPOINT, len(analysis),
                )
            else:
                analysis = self._load_checkpoint(base_dir, self._GENERATED_CHECKPOINT)
                if analysis:
                    logger.info(
                        "断点恢复：%s 有 %d 个主题（已生成待校验），跳过 7_1 生成",
                        self._GENERATED_CHECKPOINT, len(analysis),
                    )
                else:
                    analysis = self._generate(
                        keyword_data, base_dir, subtitle_l, blocked_row_ids, sink_7_1,
                    )
                if analysis:
                    # 顺序敏感：收尾（引用总时长/黑话/空壳）→ 7_2 引用校验。
                    # 事件速览按「涉及资产」数组渲染、观点速览按个股独立展示，
                    # 无需跨名称搬运，故 7_2 的 extra_scope_rows 传空。
                    self._finalize_topics(analysis, keyword_data, subtitle_l, base_dir)
                    self._run_validation(
                        analysis, keyword_data, base_dir, subtitle_l,
                        {}, sink_7_2,
                    )
                    self._reapply_opinion_invariants(analysis)

            # ==================== 落盘 ====================
            write_json_atomic(base_dir / "7_analysis_result.json", analysis)
            write_integrated(base_dir / "7_1_topic_analysis.json", sink_7_1)
            if sink_7_2:
                write_integrated(base_dir / "7_2_引用校验.json", sink_7_2)
            logger.info(
                "步骤 7 完成，%d 个 section 已定稿，7_analysis_result.json 已写入",
                len(analysis),
            )
            return analysis

    # ==================== 7_1 生成 ====================

    def _generate(
        self,
        keyword_data: dict[str, TopicData],
        base_dir: Path,
        subtitle_l: list[SubtitleRow],
        blocked_row_ids: set[int],
        sink: dict[str, dict],
    ) -> dict:
        topic_segments = TopicAnalyzer.segment_subtitles(keyword_data, subtitle_l)
        logger.info("字幕分割完成，共 %d 个话题段", len(topic_segments))

        analysis: dict = {}
        topic_specs: list[dict] = []
        # 只对非叶子节点（有子项）与段落事件生成 section；叶子/关键词事件不独立成节
        for topic in sorted(_summary_topics(keyword_data)):
            segment_dicts = topic_segments.get(topic, [])
            if not segment_dicts:
                logger.warning("主题「%s」的字幕片段为空，跳过", topic)
                continue
            topic_row_ids = [s.rowID for s in segment_dicts if s.text]
            topic_scope = scope_id_set(topic_row_ids) - blocked_row_ids
            if not topic_scope:
                logger.warning("主题「%s」的字幕片段为空，跳过", topic)
                continue
            topic_specs.append({
                "主题名": topic,
                "行号": sorted(topic_scope),
                "行号集合": topic_scope,
                # 剪枝行号已由 [DATA] 第三列行号标识随行携带，本块只给关键词
                "注释": render_annotations(
                    subtitle_l, include_keywords=True, keywords_only_jargon=True,
                    row_scope=topic_scope,
                ),
            })

        batch_size = get_batch_size("7_1_主题分析")
        topic_batches = batch_units(
            topic_specs, batch_size, sort_key=lambda s: s["主题名"],
        )
        logger.info(
            "主题分析分批：%d 个主题 → %d 批（每批 ≤ %d 个）",
            len(topic_specs), len(topic_batches), batch_size,
        )

        specs: list[tuple] = []
        scopes: list[dict[str, set[int]]] = []
        for batch_idx, batch in enumerate(topic_batches):
            union_scope: set[int] = set()
            for s in batch:
                union_scope.update(s["行号集合"])
            scopes.append({s["主题名"]: s["行号集合"] for s in batch})
            batch_annotation = render_annotations(
                subtitle_l, include_keywords=True, keywords_only_jargon=True,
                row_scope=window_row_ids(subtitle_l, union_scope),
            )
            specs.append((
                self._run_batch_worker,
                (batch, subtitle_l, base_dir, keyword_data, batch_annotation,
                 batch_idx, sink),
                {},
            ))

        futures = fanout("7", specs)
        failures: list[tuple[str, BaseException]] = []
        for future, topic_scope_map in zip(futures, scopes):
            try:
                results, batch_failures = future.result()
            except Exception as exc:
                logger.error("主题批量执行失败：%s", exc, exc_info=True)
                failures.append(("?", exc))
                continue
            for name, exc in batch_failures:
                logger.error("主题「%s」执行失败：%s", name, exc, exc_info=True)
                failures.append((name, exc))
            for name, result in results:
                try:
                    filter_citations(result, topic_scope_map[name], f"章节「{name}」")
                    result = PostProcessor.filter_empty_fields(result)
                except Exception as exc:
                    logger.error("章节「%s」后处理失败：%s", name, exc, exc_info=True)
                    failures.append((name, exc))
                    continue
                analysis[name] = result
                self._postprocess_topic(name, analysis, subtitle_l)
                logger.info("主题完成（含引用收敛与后处理）: %s", name)
                # 生成阶段断点：完成一个落一个，崩溃后重跑只补未完成主题
                write_json_atomic(
                    base_dir / self._GENERATED_CHECKPOINT, analysis,
                )

        if failures:
            raise RuntimeError(
                f"步骤 7 有 {len(failures)} 个主题失败："
                f"{[n for n, _ in failures]}（已完成主题已落盘，重跑只补失败项）"
            )
        return analysis

    def _run_batch_worker(
        self,
        batch: list[dict],
        subtitle_l: list[SubtitleRow],
        base_dir: Path,
        keyword_data: dict[str, TopicData],
        batch_annotation: str,
        batch_idx: int,
        sink: dict[str, dict],
    ) -> tuple[list[tuple[str, dict]], list[tuple[str, BaseException]]]:
        """批量 worker：整批崩溃时把本批所有主题记为失败，不炸掉整个步骤。"""
        try:
            return self._topic.generate_batch(
                batch, subtitle_l, base_dir, keyword_data, batch_annotation,
                batch_idx, "deepseek-v4-flash", sink,
            )
        except Exception as exc:
            logger.error("主题分析批量调用整体失败：%s", exc, exc_info=True)
            return [], [(s["主题名"], exc) for s in batch]

    # ==================== 收尾 A：跨 section 后处理（校验前）====================

    def _finalize_topics(
        self,
        analysis: dict,
        keyword_data: dict[str, TopicData],
        subtitle_l: list[SubtitleRow],
        base_dir: Path,
    ) -> None:
        """根级引用合并 + 引用总时长 + 黑话注入 + 空壳剔除。

        必须在 7_2 校验**之前**完成：主题引用总时长与根级引用要在校验前定稿。
        """
        merged_topics = 0
        for topic_name, topic_data in analysis.items():
            if topic_name in _FIXED_SECTION_NAMES or not isinstance(topic_data, dict):
                continue
            keyword_topic = keyword_data.get(topic_name)
            if isinstance(keyword_topic, TopicData):
                TopicAnalyzer.merge_root_citations(topic_data, keyword_topic)
            TopicAnalyzer.add_citation_duration(topic_data)
            merged_topics += 1
        logger.info("主题根级引用合并 + 引用总时长完成（%d 个主题）", merged_topics)

        used_jargons = collect_used_jargons(subtitle_l, base_dir, keyword_data)
        if used_jargons:
            for topic_name, topic_data in analysis.items():
                if topic_name in _FIXED_SECTION_NAMES:
                    continue
                if isinstance(topic_data, dict):
                    TopicAnalyzer.inject_jargon(topic_data, used_jargons)
        logger.info("主题黑话注入完成（%d 组黑话）", len(used_jargons))

        analysis = TopicAnalyzer.drop_empty_topics(analysis)
        # drop_empty_topics 原地修改并返回同一个 dict 对象，这里原地生效

    # ==================== 7_2 内联引用校验 ====================

    def _run_validation(
        self,
        analysis: dict,
        keyword_data: dict[str, TopicData],
        base_dir: Path,
        subtitle_l: list[SubtitleRow],
        extra_scope_rows: dict[str, set[int]],
        sink_7_2: dict[str, dict],
    ) -> None:
        """对主题 section 做引用支撑性校验（主讲人暗示重要话题 已由步骤 6 校验）。"""
        topic_segments = TopicAnalyzer.segment_subtitles(keyword_data, subtitle_l)
        validation_sink: dict[str, dict] = {}
        analysis, statuses = self._validator.validate(
            analysis, subtitle_l, base_dir,
            stage_prefix=VALIDATION_STAGE, step_number=STEP_NUMBER,
            topic_segments=topic_segments, sections=set(analysis),
            result_sink=validation_sink, extra_scope_rows=extra_scope_rows,
        )
        for k, v in validation_sink.items():
            sink_7_2[k] = v
        write_validation_status(base_dir / "7_validation_status.json", statuses)
