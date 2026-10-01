"""步骤 2 板块分析编排器（经济版）：e2_1 分字段组分析 → e2_2 内联引用校验。

经济版专用独立步骤，与宏观分析（经济版 step1）完全独立——独立 step 号、独立落盘、
独立引用校验 stage：

- 无选段：经济版无 keyword_data/话题段，直接对整场字幕按三个字段组分析；
- 校验内联：引用校验（e2_2）紧跟生成执行，不支撑当场重跑该字段组重写**单条**条目。

落盘：
- `2_analysis_result.json`：本步的 `板块分析` section（校验定稿后），断点续跑用；
- `e2_1_板块分析.json`：e2_1 分析 + e2_1 修复 LLM 输出整合 dump；
- `e2_2_引用校验.json`：e2_2 引用校验 LLM 输出整合 dump；
- `2_validation_status.json`：引用校验状态台账（通过/丢弃/异常保留/未处理）。
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
from ...utils.analysis_cleanup import _DROP_KEY
from ...log_config import step_logger
from ...llm_summarize.analysis_common import PostProcessor
from ...llm_summarize.analysis_common.status_ledger import write_validation_status
from ...llm_summarize.citation_check import CitationValidator
from ...llm_summarize.schemas import SubtitleRow
from ...llm_summarize.utils.llm_text_utils import (
    collect_non_analysis_row_ids,
    load_prompt,
    render_scoped_subtitle,
    render_windowed_subtitle,
)
from ...llm_summarize.utils.rowid_scope import (
    collect_cited_row_ids,
    filter_citations,
    scope_id_set,
)
from .sector_analyzer import (
    SECTOR_FIELD_GROUPS,
    SectorAnalyzer,
    SectorFieldGroup,
)

_section_anchor = __file__

_env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)

STEP_NUMBER = 2
VALIDATION_STAGE = "e2_2"

_SECTION_NAME = "板块分析"

def _dedup_sector_block_lists(sector: dict) -> None:
    """校验收尾兜底：板块各字段组条目列表内同标识条目只保留首个。

    字段组的结构描述（条目路径、标识键）来自 SECTOR_FIELD_GROUPS 单一来源。
    三个字段组各自独立生成后再合并，组内重复由模型自检、组间重复（以及修复产物
    带回来的重复）只能在此兜底。在全部校验完成、不再有在途按索引写回后执行，
    删元素不会引发索引错位。
    """
    for group in SECTOR_FIELD_GROUPS:
        path = group.条目路径
        key = group.标识键
        node: object = sector
        broken = False
        for part in path[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                broken = True
                break
        if broken:
            continue
        items = node.get(path[-1]) if isinstance(node, dict) else None
        if not isinstance(items, list):
            continue
        seen: set[str] = set()
        kept: list = []
        for item in items:
            if not isinstance(item, dict):
                kept.append(item)
                continue
            ident = str(item.get(key) or "").strip()
            if ident and ident in seen:
                logger.warning(
                    "e2_2 收尾去重：「%s」出现同%s条目，删除重复的「%s」",
                    ".".join(path), key, ident,
                )
                continue
            if ident:
                seen.add(ident)
            kept.append(item)
        if len(kept) != len(items):
            node[path[-1]] = kept


class SectorAnalysisStep:
    """步骤 2 编排器（经济版）。`analyze()` 是唯一对外入口，返回 `{"板块分析": ...}`。"""

    def __init__(self) -> None:
        self._llm_client = LLMClient()
        self._sector = SectorAnalyzer(
            self._llm_client,
            load_prompt("e2_1_板块分析.txt", _section_anchor),
            _load_schema("e2_1_板块分析", _section_anchor),
        )
        self._validator = CitationValidator(self._llm_client, with_type=False)

    @staticmethod
    def _load_checkpoint(base_dir: Path) -> dict:
        """断点：已定稿的板块分析直接复用（含校验，不重跑）。"""
        data = read_json(base_dir / "2_analysis_result.json")
        if isinstance(data, dict) and isinstance(data.get(_SECTION_NAME), dict):
            return {_SECTION_NAME: data[_SECTION_NAME]}
        return {}

    def analyze(self, base_dir: Path, subtitle_l: list[SubtitleRow]) -> dict:
        """生成并校验 `板块分析` section，返回 `{"板块分析": {...}}`。

        经济版无 keyword_data/话题段，直接对整场字幕按三个字段组分析。
        """
        with step_logger("第2步-板块分析"):
            cached = self._load_checkpoint(base_dir)
            if cached:
                logger.info("检测到 2_analysis_result.json，跳过本步（已完成校验）")
                return cached

            groups = list(SECTOR_FIELD_GROUPS)
            blocked_row_ids = collect_non_analysis_row_ids(subtitle_l)
            all_valid_row_ids = scope_id_set(
                s.rowID for s in subtitle_l if s.text
            ) - blocked_row_ids
            logger.info(
                "板块分析范围：整场字幕 %d 行；字段组 %d 个",
                len(all_valid_row_ids), len(groups),
            )

            # [DATA] = 整场字幕（经济版无选段，两列 rowID\t文本，无行号标识、无 ANNOTATION）
            subtitle_text = render_scoped_subtitle(
                subtitle_l, row_scope=all_valid_row_ids,
            )

            sink_2_1: dict[str, dict] = load_dump_outputs(
                base_dir / "e2_1_板块分析.json",
            )
            sink_2_2: dict[str, dict] = load_dump_outputs(
                base_dir / "e2_2_引用校验.json",
            )

            def _run_group(group: SectorFieldGroup) -> tuple[str, dict]:
                return self._sector.analyze_group(
                    group, subtitle_text, base_dir,
                    result_sink=sink_2_1,
                )

            futures = fanout(
                "2", [(_run_group, (g,), {}) for g in groups],
            )
            sector: dict = {}
            failures: list[tuple[str, BaseException]] = []
            for group, future in zip(groups, futures):
                try:
                    name, result = future.result()
                except Exception as exc:
                    logger.error(
                        "板块字段组「%s」生成失败：%s", group.名称, exc, exc_info=True,
                    )
                    failures.append((group.名称, exc))
                    continue
                filter_citations(result, all_valid_row_ids, f"板块字段组「{name}」")
                sector.update(PostProcessor.filter_empty_fields(result))

            if failures:
                raise RuntimeError(
                    f"步骤 2 有 {len(failures)} 个字段组失败："
                    f"{[n for n, _ in failures]}（已完成组已落盘，重跑只补失败项）"
                )
            if not sector:
                logger.warning("板块分析三个字段组全部无产出，跳过本 section")
                return {}

            analysis = {_SECTION_NAME: sector}

            # 行号 → 时间 + 引用原文填充（内联校验在**时间形态**引用上进行）
            analysis[_SECTION_NAME] = PostProcessor.convert_row_ids_in_section(
                sector, subtitle_l,
            )
            PostProcessor.populate_citation_texts_in_section(
                analysis[_SECTION_NAME], subtitle_l,
            )

            # ==================== e2_2 内联引用校验 ====================
            def _repair_fn(task: dict, current_item: dict, failure_reason: str) -> dict | None:
                """重跑失败条目所属字段组，只重写这一条。

                [DATA] 按失败条目的引用行窗口化（±30 行上下文），
                修复只重写一条，结论句往往就在原引用行附近，整场重读是浪费。
                """
                rows = all_valid_row_ids  # 越界校验基准保持全场不变
                centers = collect_cited_row_ids(current_item)
                subtitle_text = render_windowed_subtitle(
                    subtitle_l, centers, pad=30,
                )
                repaired = self._sector.repair_item(
                    task, current_item, failure_reason, subtitle_text, base_dir,
                    rows,
                    result_sink=sink_2_1,
                )
                if repaired is None:
                    return None
                if isinstance(repaired, dict) and _DROP_KEY in repaired:
                    # 丢弃 sentinel：板块观点修复判定「合并/删除」，跳过引用转换，交由上层丢弃
                    return repaired
                repaired = PostProcessor.convert_row_ids_in_section(repaired, subtitle_l)
                PostProcessor.populate_citation_texts_in_section(repaired, subtitle_l)
                return repaired

            validation_sink: dict[str, dict] = {}
            analysis, statuses = self._validator.validate(
                analysis, subtitle_l, base_dir,
                stage_prefix=VALIDATION_STAGE, step_number=STEP_NUMBER,
                sections={_SECTION_NAME}, repair_fn=_repair_fn,
                result_sink=validation_sink,
            )
            for k, v in validation_sink.items():
                sink_2_2[k] = v

            # 同标识去重（三个字段组合并 + 修复产物可能带入重复）
            _dedup_sector_block_lists(analysis[_SECTION_NAME])

            # ==================== 落盘 ====================
            write_json_atomic(base_dir / "2_analysis_result.json", analysis)
            write_integrated(base_dir / "e2_1_板块分析.json", sink_2_1)
            write_integrated(base_dir / "e2_2_引用校验.json", sink_2_2)
            write_validation_status(base_dir / "2_validation_status.json", statuses)
            logger.info(
                "步骤 2 完成：板块分析 %d 个字段，%s",
                len(analysis[_SECTION_NAME]),
                {f: _size(v) for f, v in analysis[_SECTION_NAME].items()},
            )
            return analysis


def _size(value: object) -> str:
    if isinstance(value, list):
        return f"{len(value)} 条"
    if isinstance(value, dict):
        return "✓"
    return "—"
