"""步骤 5 宏观分析编排器：5_1 选段 → 5_2 分字段组分析 → 5_3 内联引用校验。

**与旧实现的三处结构性差异**（对应「目录爆炸 / 修复不彻底 / 事后校验」三个问题）：

1. **渐进式披露**：不再把整场字幕一次性塞给模型生成六个字段，而是先按段索引选段
   （5_1，只吃索引），再把选中段全文按字段组分别分析（5_2）。
2. **校验内联**：引用校验（5_3）在同一个步骤内紧跟生成执行，不支撑当场重跑该
   字段组重写**单条**条目并写回定稿，不跨步骤回滚。
3. **修复粒度**：修复目标从"整个字段"缩到"失败的那一条"，log_name 用
   「字段组 + 条目标识」的可读名，不再出现 `5_1_宏观分析_repair_{字段}_{uuid6}` 目录。

落盘：
- `5_analysis_result.json`：本步的 `宏观分析` section（校验定稿后），断点续跑用；
- `5_1_macro_analysis.json`：5_1 选段 / 5_2 分析 / 5_2 修复 全部 LLM 输出整合 dump；
- `5_validation_status.json`：引用校验状态台账（通过/丢弃/异常保留/未处理）。
"""

import json
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
from ..analysis_common import (
    PostProcessor,
    collect_used_jargons,
)
from ..analysis_common.status_ledger import write_validation_status
from ..schemas import SubtitleRow, TopicData
from ..citation_check import CitationValidator
from ..utils.llm_text_utils import (
    collect_non_analysis_row_ids,
    load_prompt,
    render_annotations,
    render_scoped_subtitle,
    render_windowed_subtitle,
    window_row_ids,
)
from ..utils.rowid_scope import (
    collect_cited_row_ids,
    filter_citations,
    scope_id_set,
)
from .macro_analyzer import (
    MACRO_FIELD_GROUPS,
    MacroAnalyzer,
    MacroFieldGroup,
)
from .macro_segment_selector import MacroSegmentSelector, build_segment_index

_section_anchor = __file__

_env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)

STEP_NUMBER = 5
VALIDATION_STAGE = "5_3"

# 与步骤 6/7 产出可被本步校验的固定章节名区分：本步只产出 宏观分析
_SECTION_NAME = "宏观分析"

# 宏观分析中需做同标识去重的列表字段：(字段路径, 标识键)
_MACRO_BLOCK_LISTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("大盘情绪与状态判断",), "标题"),
    (("宏观经济与事件影响分析",), "标题"),
    (("后市观点",), "标题"),
    (("仓位建议",), "标题"),
    (("板块分析", "板块涨跌"), "板块名称"),
    (("热点板块预测", "板块清单"), "板块名称"),
)


def _dedup_macro_block_lists(macro: dict) -> None:
    """校验收尾兜底：宏观各分块列表内同标识（标题/板块名称）条目只保留首个。

    五个字段组各自独立生成后再合并，组内重复由模型自检、组间重复（以及修复产物
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
                    "5_3 收尾去重：「%s」出现同%s条目，删除重复的「%s」",
                    ".".join(path), key, ident,
                )
                continue
            if ident:
                seen.add(ident)
            kept.append(item)
        if len(kept) != len(items):
            node[path[-1]] = kept


def _repair_sectors_line(sub_key: str | None, current_item: dict, all_sectors_text: str) -> str:
    """repair 的 [DATABASE] 收窄：板块类字段只留当前板块那一行，其余字段组传空。

    修复只重写一条，板块名称已在原条目里，模型只需知道该板块在字幕里的引用行号
    范围即可重新挑引用，无需整份 140 个板块的清单；匹配不到返回空（安全退化）。
    """
    if sub_key not in ("板块分析", "热点板块预测") or not all_sectors_text:
        return ""
    name = str((current_item or {}).get("板块名称") or "").strip()
    if not name:
        return ""
    prefix = f"- {name}："
    for line in all_sectors_text.splitlines():
        if line.startswith(prefix):
            return line
    return ""


class MacroAnalysisStep:
    """步骤 5 编排器。`analyze()` 是唯一对外入口，返回 `{"宏观分析": ...}`。"""

    def __init__(self) -> None:
        self._llm_client = LLMClient()
        self._selector = MacroSegmentSelector(
            self._llm_client, load_prompt("5_1_宏观分析选段.txt", _section_anchor),
        )
        self._macro = MacroAnalyzer(
            self._llm_client,
            load_prompt("5_2_宏观分析.txt", _section_anchor),
            _load_schema("5_2_宏观分析", _section_anchor),
        )
        # 引用校验是组件：prompt / schema 与它本模块共置，此处不拼路径
        self._validator = CitationValidator(self._llm_client)

    # ==================== 小工具 ====================

    @staticmethod
    def _format_sectors_with_ranges_for_macro(
        keyword_data: dict[str, TopicData],
    ) -> tuple[str, str, str]:
        """按类型分流生成 (板块列表, 指数列表, 风格因子列表)，均带引用行号区间。

        板块列表供「板块分析/热点板块预测」定位；指数/风格因子列表供
        「大盘情绪/后市观点」定位指数表现与市场风格。空 keyword_data（如经济版）
        安全返回三空串，由模板 `{% if %}` 决定是否渲染。
        """
        skip_keys = {
            'result', 'system_prompt', 'user_prompt',
            '步骤', '输入', '思考', '输出', '引用时间', '子项',
        }

        def _line(key: str, value: TopicData) -> str:
            range_strs = []
            for tr in value.引用 or []:
                if not isinstance(tr, dict):
                    continue
                start, end = tr.get("开始行号"), tr.get("结束行号")
                if start is None or end is None:
                    continue
                range_strs.append(f"{start}-{end}" if start != end else f"{start}")
            return f"- {key}：行号 {', '.join(range_strs)}" if range_strs else f"- {key}：无引用"

        sector_lines: list[str] = []
        index_lines: list[str] = []
        style_lines: list[str] = []
        for key, value in keyword_data.items():
            if key in skip_keys or not isinstance(value, TopicData):
                continue
            text = _line(key, value)
            if value.类型 == "指数":
                index_lines.append(text)
            elif value.类型 == "风格因子":
                style_lines.append(text)
            else:
                sector_lines.append(text)
        return (
            "\n".join(sector_lines),
            "\n".join(index_lines),
            "\n".join(style_lines),
        )

    @staticmethod
    def _load_checkpoint(base_dir: Path) -> dict:
        """断点：已定稿的宏观分析直接复用（含校验，不重跑）。"""
        data = read_json(base_dir / "5_analysis_result.json")
        if isinstance(data, dict) and isinstance(data.get(_SECTION_NAME), dict):
            return {_SECTION_NAME: data[_SECTION_NAME]}
        return {}

    @staticmethod
    def _selected_rows(
        seg_rows: dict[int, set[int]], picked: list[int], fallback: set[int],
    ) -> set[int]:
        """选中的段号 → rowID 集合；未选/选空时退回全部可分析行（= 整场字幕）。"""
        rows: set[int] = set()
        for n in picked:
            rows |= seg_rows.get(n, set())
        if not rows:
            return set(fallback)
        return rows & fallback if fallback else rows

    # ==================== 主入口 ====================

    def analyze(
        self,
        keyword_data: dict[str, TopicData],
        base_dir: Path,
        subtitle_l: list[SubtitleRow],
        *,
        economic: bool = False,
    ) -> dict:
        """生成并校验 `宏观分析` section，返回 `{"宏观分析": {...}}`。

        经济版（economic=True）：只跑四个非板块字段组（跳过 `名称=="板块"` 的
        「热点板块预测/板块分析」，其依赖聚类的板块清单），并跳过黑话注入
        （无词级产物与聚类，黑话注入无意义）。
        """
        with step_logger("第5步-宏观分析"):
            cached = self._load_checkpoint(base_dir)
            if cached:
                logger.info("检测到 5_analysis_result.json，跳过本步（已完成校验）")
                return cached

            groups = [g for g in MACRO_FIELD_GROUPS if not (economic and g.名称 == "板块")]
            blocked_row_ids = collect_non_analysis_row_ids(subtitle_l)
            all_valid_row_ids = scope_id_set(
                s.rowID for s in subtitle_l if s.text
            ) - blocked_row_ids
            all_sectors_text, indices_text, styles_text = self._format_sectors_with_ranges_for_macro(keyword_data)

            segment_index, seg_rows = build_segment_index(
                base_dir, keyword_data, subtitle_l,
            )
            logger.info(
                "话题段索引构建完成：%d 段；字段组 %d 个",
                len(seg_rows), len(groups),
            )

            sink: dict[str, dict] = load_dump_outputs(
                base_dir / "5_1_macro_analysis.json",
            )

            # ==================== 5_1 选段 + 5_2 分析 ====================
            # 每个字段组串行做「选段 → 分析」（后者依赖前者选出的段），
            # 组与组之间并行（全局线程池 + ResourceManager 并发闸）。
            def _run_group(group: MacroFieldGroup) -> tuple[str, dict]:
                picked: list[int] = []
                if segment_index:
                    picked = self._selector.select(group, segment_index, base_dir, sink)
                else:
                    logger.warning(
                        "话题段索引为空，字段组「%s」跳过选段、按整场字幕分析",
                        group.名称,
                    )
                rows = self._selected_rows(seg_rows, picked, all_valid_row_ids)
                logger.info(
                    "字段组「%s」分析范围：选中 %d 段 → %d 行",
                    group.名称, len(picked), len(rows),
                )
                # [DATA] = 选中段全文（新的 [DATA] 例外，见模块 docstring）
                subtitle_text = render_scoped_subtitle(subtitle_l, row_scope=rows, with_type=True)
                annotation_text = render_annotations(
                    subtitle_l, include_keywords=True, keywords_only_jargon=True,
                    row_scope=rows,
                )
                return self._macro.analyze_group(
                    group, subtitle_text, base_dir,
                    all_sectors_text=all_sectors_text,
                    indices_text=indices_text,
                    styles_text=styles_text,
                    annotation_text=annotation_text,
                    result_sink=sink,
                )

            futures = fanout(
                "5", [(_run_group, (g,), {}) for g in groups],
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
                    f"步骤 5 有 {len(failures)} 个字段组失败："
                    f"{[n for n, _ in failures]}（已完成组已落盘，重跑只补失败项）"
                )
            if not macro:
                logger.warning("宏观分析五个字段组全部无产出，跳过本 section")
                return {}

            analysis = {_SECTION_NAME: macro}

            # 行号 → 时间 + 引用原文填充（内联校验在**时间形态**引用上进行）
            analysis[_SECTION_NAME] = PostProcessor.convert_row_ids_in_section(
                macro, subtitle_l,
            )
            PostProcessor.populate_citation_texts_in_section(
                analysis[_SECTION_NAME], subtitle_l,
            )

            # ==================== 5_3 内联引用校验 ====================
            def _repair_fn(task: dict, current_item: dict, failure_reason: str) -> dict | None:
                """重跑失败条目所属字段组，只重写这一条。

                [DATA]/[ANNOTATION] 按失败条目的引用行窗口化（±30 行上下文），
                [DATABASE] 收窄到当前板块那一行：修复只重写一条，结论句往往就在
                原引用行附近，整场重读是浪费（16 次 repair 占 5_2 一半思考）。
                """
                rows = all_valid_row_ids  # 越界校验基准保持全场不变
                centers = collect_cited_row_ids(current_item)
                subtitle_text = render_windowed_subtitle(
                    subtitle_l, centers, with_type=True, pad=30,
                )
                visible = window_row_ids(subtitle_l, centers, pad=30)
                annotation_text = render_annotations(
                    subtitle_l, include_keywords=True, keywords_only_jargon=True,
                    row_scope=visible,
                )
                sectors_text = _repair_sectors_line(
                    task.get("sub_key"), current_item, all_sectors_text,
                )
                repaired = self._macro.repair_item(
                    task, current_item, failure_reason, subtitle_text, base_dir,
                    rows,
                    all_sectors_text=sectors_text,
                    annotation_text=annotation_text,
                    result_sink=sink,
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
                sink[k] = v

            # 同标识去重（五个字段组合并 + 修复产物可能带入重复）
            _dedup_macro_block_lists(analysis[_SECTION_NAME])

            # 黑话注入（字幕里实际出现过的黑话挂到板块名）。
            # 经济版跳过：无词级产物与聚类，黑话注入既无依据也无目标（板块字段组不跑）。
            if not economic:
                used_jargons = collect_used_jargons(subtitle_l, base_dir, keyword_data)
                if used_jargons:
                    MacroAnalyzer.inject_jargon(analysis[_SECTION_NAME], used_jargons)
                logger.info("宏观分析黑话注入完成（%d 组黑话）", len(used_jargons))

            # ==================== 落盘 ====================
            write_json_atomic(base_dir / "5_analysis_result.json", analysis)
            write_integrated(base_dir / "5_1_macro_analysis.json", sink)
            write_validation_status(base_dir / "5_validation_status.json", statuses)
            logger.info(
                "步骤 5 完成：宏观分析 %d 个字段，%s",
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
