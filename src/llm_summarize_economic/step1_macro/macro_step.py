"""步骤 1 宏观分析编排器（经济版）：e1_1 分字段组分析 → e1_2 内联引用校验。

经济版专用独立步骤，与完整版 ``step5_macro`` 完全独立——独立 step 号、独立落盘、
独立引用校验 stage：

- 无选段：经济版无 keyword_data/话题段，直接对整场字幕按四个字段组分析；
- 校验内联：引用校验（e1_2）紧跟生成执行，不支撑当场重跑该字段组重写**单条**条目。

落盘：
- `1_analysis_result.json`：本步的 `宏观分析` section（校验定稿后），断点续跑用；
- `e1_1_宏观分析.json`：e1_1 分析 + e1_1 修复 LLM 输出整合 dump；
- `e1_2_引用校验.json`：e1_2 引用校验 LLM 输出整合 dump；
- `1_validation_status.json`：引用校验状态台账（通过/丢弃/异常保留/未处理）。
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
from .macro_analyzer import (
    MACRO_FIELD_GROUPS,
    MacroAnalyzer,
    MacroFieldGroup,
)

_section_anchor = __file__

_env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)

STEP_NUMBER = 1
VALIDATION_STAGE = "e1_2"

_SECTION_NAME = "宏观分析"

# 宏观分析中需做同标识去重的列表字段：(字段路径, 标识键)
_MACRO_BLOCK_LISTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("大盘情绪",), "标题"),
    (("宏观事件",), "标题"),
    (("宏观观点",), "标题"),
    (("仓位建议",), "标题"),
)


def _dedup_macro_block_lists(macro: dict) -> None:
    """校验收尾兜底：宏观各分块列表内同标识（标题）条目只保留首个。

    四个字段组各自独立生成后再合并，组内重复由模型自检、组间重复（以及修复产物
    带回来的重复）只能在此兜底。在全部校验完成、不再有在途按索引写回后执行，
    删元素不会引发索引错位。
    """
    for path, key in _MACRO_BLOCK_LISTS:
        node: object = macro
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
                    "e1_2 收尾去重：「%s」出现同%s条目，删除重复的「%s」",
                    ".".join(path), key, ident,
                )
                continue
            if ident:
                seen.add(ident)
            kept.append(item)
        if len(kept) != len(items):
            node[path[-1]] = kept


class MacroAnalysisStep:
    """步骤 1 编排器（经济版）。`analyze()` 是唯一对外入口，返回 `{"宏观分析": ...}`。"""

    def __init__(self) -> None:
        self._llm_client = LLMClient()
        self._macro = MacroAnalyzer(
            self._llm_client,
            _load_schema("e1_1_宏观分析", _section_anchor),
            load_prompt("_macro_data_econ.txt", _section_anchor),
        )
        self._validator = CitationValidator(self._llm_client, with_type=False)

    @staticmethod
    def _load_checkpoint(base_dir: Path) -> dict:
        """断点：已定稿的宏观分析直接复用（含校验，不重跑）。"""
        data = read_json(base_dir / "1_analysis_result.json")
        if isinstance(data, dict) and isinstance(data.get(_SECTION_NAME), dict):
            return {_SECTION_NAME: data[_SECTION_NAME]}
        return {}

    def analyze(self, base_dir: Path, subtitle_l: list[SubtitleRow]) -> dict:
        """生成并校验 `宏观分析` section，返回 `{"宏观分析": {...}}`。

        经济版无 keyword_data/话题段，直接对整场字幕按四个字段组分析。
        """
        with step_logger("第1步-宏观分析"):
            cached = self._load_checkpoint(base_dir)
            if cached:
                logger.info("检测到 1_analysis_result.json，跳过本步（已完成校验）")
                return cached

            groups = list(MACRO_FIELD_GROUPS)
            blocked_row_ids = collect_non_analysis_row_ids(subtitle_l)
            all_valid_row_ids = scope_id_set(
                s.rowID for s in subtitle_l if s.text
            ) - blocked_row_ids
            logger.info(
                "宏观分析范围：整场字幕 %d 行；字段组 %d 个",
                len(all_valid_row_ids), len(groups),
            )

            # [DATA] = 整场字幕（经济版无选段，两列 rowID\t文本，无行号标识、无 ANNOTATION）
            subtitle_text = render_scoped_subtitle(
                subtitle_l, row_scope=all_valid_row_ids,
            )

            sink_1_1: dict[str, dict] = load_dump_outputs(
                base_dir / "e1_1_宏观分析.json",
            )
            sink_1_2: dict[str, dict] = load_dump_outputs(
                base_dir / "e1_2_引用校验.json",
            )

            def _run_group(group: MacroFieldGroup) -> tuple[str, dict]:
                return self._macro.analyze_group(
                    group, subtitle_text, base_dir,
                    result_sink=sink_1_1,
                )

            futures = fanout(
                "1", [(_run_group, (g,), {}) for g in groups],
            )
            macro: dict = {}
            failures: list[tuple[str, BaseException]] = []
            for group, future in zip(groups, futures):
                try:
                    name, result = future.result()
                except Exception as exc:
                    logger.error(
                        "宏观字段组「%s」生成失败：%s", group.名称, exc, exc_info=True,
                    )
                    failures.append((group.名称, exc))
                    continue
                filter_citations(result, all_valid_row_ids, f"宏观字段组「{name}」")
                macro.update(PostProcessor.filter_empty_fields(result))

            if failures:
                raise RuntimeError(
                    f"步骤 1 有 {len(failures)} 个字段组失败："
                    f"{[n for n, _ in failures]}（已完成组已落盘，重跑只补失败项）"
                )
            if not macro:
                logger.warning("宏观分析四个字段组全部无产出，跳过本 section")
                return {}

            analysis = {_SECTION_NAME: macro}

            # 行号 → 时间 + 引用原文填充（内联校验在**时间形态**引用上进行）
            analysis[_SECTION_NAME] = PostProcessor.convert_row_ids_in_section(
                macro, subtitle_l,
            )
            PostProcessor.populate_citation_texts_in_section(
                analysis[_SECTION_NAME], subtitle_l,
            )

            # ==================== e1_2 内联引用校验 ====================
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
                repaired = self._macro.repair_item(
                    task, current_item, failure_reason, subtitle_text, base_dir,
                    rows,
                    result_sink=sink_1_1,
                )
                if repaired is None:
                    return None
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
                sink_1_2[k] = v

            # 同标识去重（四个字段组合并 + 修复产物可能带入重复）
            _dedup_macro_block_lists(analysis[_SECTION_NAME])

            # ==================== 落盘 ====================
            write_json_atomic(base_dir / "1_analysis_result.json", analysis)
            write_integrated(base_dir / "e1_1_宏观分析.json", sink_1_1)
            write_integrated(base_dir / "e1_2_引用校验.json", sink_1_2)
            write_validation_status(base_dir / "1_validation_status.json", statuses)
            logger.info(
                "步骤 1 完成：宏观分析 %d 个字段，%s",
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
