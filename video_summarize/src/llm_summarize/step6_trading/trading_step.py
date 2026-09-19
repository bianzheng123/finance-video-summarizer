"""步骤 6 编排器：`6_1 交易技巧` ∥（`6_2 敏感信号判定` → 聚合 → `6_3 引用校验`）。

两个分支**并行**（主流程三路并行下，步骤 6 内部再并行两个分支，由全局线程池承载）：
- 分支 A：6_1 交易技巧。作用域是全场（技巧常是对某段行情的复盘），不依赖任何主题。
  **不做引用校验**（刻意的）：交易技巧是报告里权重最低的 section（操作方法论、
  不含标的多空判断），一条引用不精确不会误导仓位决策，而校验它要为整场条目各发
  一次 LLM。故只保留 `filter_citations` 的确定性越界过滤。
- 分支 B：6_2 敏感信号判定（正则召回 + LLM 语义判定，原地给 keyword_data 挂
  `敏感提示`）→ 聚合（`SensitivityAggregator`，产出顶层 `主讲人暗示重要话题`）→
  6_3 引用校验（`CitationValidator` 只校验 `主讲人暗示重要话题`）。判定与聚合只
  依赖 keyword_data，与主题分析（步骤 7）无依赖，故可整支并行。

落盘：
- `6_analysis_result.json`：本步的 `交易技巧` + `主讲人暗示重要话题` 两个 section，断点续跑用；
- `6_1_trading_skills.json`：交易技巧 LLM 输出整合 dump；
- `6_2_sensitive_signals.json`：敏感信号判定 + 6_3 校验 LLM 输出整合 dump；
- `6_validation_status.json`：敏感信号引用校验状态台账。
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

from ...llm_infra.io import (
    load_dump_outputs,
    read_json,
    write_integrated,
    write_json_atomic,
)
from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema
from ...llm_infra.resource import fanout
from ...log_config import step_logger
from ..analysis_common import PostProcessor
from ..analysis_common.status_ledger import write_validation_status
from ..schemas import SubtitleRow, TopicData
from ..citation_check import CitationValidator
from ..utils.llm_text_utils import (
    collect_non_analysis_row_ids,
    load_prompt,
    render_annotations,
    render_scoped_subtitle,
)
from ..utils.rowid_scope import filter_citations, scope_id_set
from .sensitivity_aggregator import SensitivityAggregator
from .sensitivity_analyzer import SensitivityAnalyzer
from .trading_skill_analyzer import TradingSkillAnalyzer

_section_anchor = __file__

_env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)

_SECTION_NAME = "交易技巧"


class TradingSensitivityStep:
    """步骤 6 编排器。`analyze()` 返回 `{"交易技巧": ..., "主讲人暗示重要话题": [...]}`。"""

    def __init__(self) -> None:
        self._llm_client = LLMClient()
        self._trading = TradingSkillAnalyzer(
            self._llm_client,
            load_prompt("6_1_交易技巧.txt", _section_anchor),
            _load_schema("6_1_交易技巧", _section_anchor),
        )
        self._sensitivity = SensitivityAnalyzer()
        self._validator = CitationValidator(self._llm_client)

    @staticmethod
    def _load_checkpoint(base_dir: Path) -> dict:
        data = read_json(base_dir / "6_analysis_result.json")
        if isinstance(data, dict) and data:
            return {k: v for k, v in data.items() if isinstance(v, (dict, list))}
        return {}

    def analyze(
        self,
        keyword_data: dict[str, TopicData],
        base_dir: Path,
        subtitle_l: list[SubtitleRow],
    ) -> dict:
        """双分支并行：6_1 交易技巧 ∥（6_2 判定 → 聚合 → 6_3 校验），返回本步 section。"""
        with step_logger("第6步-交易技巧与敏感信号"):
            cached = self._load_checkpoint(base_dir)
            if cached:
                logger.info(
                    "检测到 6_analysis_result.json，跳过本步（%d 个 section）",
                    len(cached),
                )
                return cached

            futures = fanout("6", [
                (self._run_trading_skill, (base_dir, subtitle_l), {}),
                (self._run_sensitivity, (keyword_data, subtitle_l, base_dir), {}),
            ])
            analysis: dict = {}
            for future in futures:
                # 任一分支失败整体上抛（另一分支产物已落盘，重跑只补失败分支）
                analysis.update(future.result())

            write_json_atomic(base_dir / "6_analysis_result.json", analysis)
            logger.info(
                "步骤 6 完成：%s",
                " / ".join(sorted(analysis)) or "无产出",
            )
            return analysis

    def _run_trading_skill(
        self, base_dir: Path, subtitle_l: list[SubtitleRow],
    ) -> dict:
        """分支 A：6_1 交易技巧（全场作用域，不做引用校验）。"""
        blocked_row_ids = collect_non_analysis_row_ids(subtitle_l)
        all_valid_row_ids = scope_id_set(
            s.rowID for s in subtitle_l if s.text
        ) - blocked_row_ids
        # 交易技巧作用域是全场：技巧常是对某段行情的复盘，切片会看不到被复盘的那段
        subtitle_text = render_scoped_subtitle(subtitle_l, with_type=True)
        annotation_text = render_annotations(subtitle_l)

        sink_6_1: dict[str, dict] = load_dump_outputs(
            base_dir / "6_1_trading_skills.json",
        )

        try:
            name, result = self._trading.analyze(
                subtitle_text, base_dir, annotation_text, result_sink=sink_6_1,
            )
        except Exception as exc:
            logger.error("「%s」生成失败：%s", _SECTION_NAME, exc, exc_info=True)
            raise RuntimeError(
                f"步骤 6 交易技巧生成失败：{exc}（已完成的已落盘，重跑只补失败项）"
            ) from exc
        filter_citations(result, all_valid_row_ids, f"章节「{name}」")
        analysis: dict = {name: PostProcessor.filter_empty_fields(result)}

        if _SECTION_NAME in analysis:
            analysis[_SECTION_NAME] = PostProcessor.convert_row_ids_in_section(
                analysis[_SECTION_NAME], subtitle_l,
            )
            PostProcessor.populate_citation_texts_in_section(
                analysis[_SECTION_NAME], subtitle_l,
            )
            logger.info(
                "6_1 交易技巧完成（刻意不做引用校验，仅 filter_citations 确定性越界过滤）"
            )

        write_integrated(base_dir / "6_1_trading_skills.json", sink_6_1)
        return analysis

    def _run_sensitivity(
        self,
        keyword_data: dict[str, TopicData],
        subtitle_l: list[SubtitleRow],
        base_dir: Path,
    ) -> dict:
        """分支 B：6_2 判定 → 聚合 → 6_3 校验（串行链）。"""
        sink_6_2: dict[str, dict] = load_dump_outputs(
            base_dir / "6_2_sensitive_signals.json",
        )

        # 6_2 敏感信号判定：原地给 keyword_data 挂 敏感提示
        self._sensitivity.analyze(
            keyword_data, subtitle_l, base_dir, result_sink=sink_6_2,
        )
        logger.info("敏感信号判定完成（6_2），产物挂在 keyword_data")

        # 聚合：从 keyword_data 各主题的 敏感提示 产出 主讲人暗示重要话题
        section = SensitivityAggregator.aggregate(keyword_data, subtitle_l)
        if not section:
            logger.info("无敏感信号，跳过聚合与 6_3 校验")
            write_integrated(base_dir / "6_2_sensitive_signals.json", sink_6_2)
            return {}

        # 6_3 引用校验：只校验 主讲人暗示重要话题
        validation_sink: dict[str, dict] = {}
        section, statuses = self._validator.validate(
            section, subtitle_l, base_dir,
            stage_prefix="6_3", step_number=6,
            topic_segments={}, sections={"主讲人暗示重要话题"},
            result_sink=validation_sink,
        )
        for k, v in validation_sink.items():
            sink_6_2[k] = v
        write_validation_status(base_dir / "6_validation_status.json", statuses)
        write_integrated(base_dir / "6_2_sensitive_signals.json", sink_6_2)
        logger.info(
            "敏感信号聚合与 6_3 校验完成：%d 条",
            len(section.get("主讲人暗示重要话题") or []),
        )
        return section
