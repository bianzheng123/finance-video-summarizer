"""引用内容支撑性校验组件（跨步骤内联复用，本身不是一个步骤）。

**组件而非步骤**：校验必须紧跟生成——「生成 → 校验 → 不支撑当场重写/修复 → 定稿」
闭环在同一步骤内完成，不跨步骤回滚（旧实现把校验单独放步骤 6，修复回调又去
重跑步骤 5 的 5_1，产物回写路径长且不复核）。因此本模块不带自己的步骤号：
`stage_prefix` / `step_number` 由调用方（步骤 5 与 7）传入，决定 `log_name`、
`step_name`（阶段参数匹配）与落盘的 `llm_output_N` 目录。

覆盖四类章节：投资机会 / 观点 / 宏观分析 / 主讲人暗示重要话题。**`交易技巧` 刻意不在
其内**——它是权重最低的 section（操作方法论、不含标的多空判断），不值得为它按条目
各发一次校验调用；步骤 6 只做确定性越界过滤（详见 `step6_trading/trading_step.py`）。
同一 section 的待校验条目合成一批，**每个条目只做一次校验调用**（组内条目共享
[DATA]/[ANNOTATION]/[ROWID]/模型，只有 [TASK] 的条目清单不同）；返回不支撑时
当轮定案，不做二轮复验：
- 提供了 `repair_fn` 的章节（现为步骤 5 宏观分析）：调它重跑该字段组（把校验的
  失败原因作为修复反馈），**直接采用修复产物**（每条目最多修 1 次）——不接受
  字幕照抄式重写是既定要求；修复不可用/失败时退回重写路径。
- 其余条目：直接采用 LLM 重写后的条目，重新做行号→时间转换 + 原文填充后定案。
- 重写缺失 / 重写为空（观点类判定无重写价值）→ 丢弃 sentinel。

**单轮定案的取舍**：旧实现最多 3 轮，批内任一条目不支撑就整批重跑下一轮，
reasoning token 被放大（实测批量化后 156K→426K，净成本反升）。改为单轮后调用数
从「条目批数 × 实际轮数」降为「条目批数」，代价是重写/修复产物不再经第二次复核
——可接受的原因：重写引用先过作用域过滤，再由 `_refill_citations` 用真实字幕
确定性回填 时间 + 原文，引用本身不可能凭空捏造。

丢弃 sentinel 由 validate() 收尾 clean_analysis 从 analysis 中整条移除
（不进 llm_analysis.json）；validate() 返回 (analysis, statuses)，statuses 记录
全部待校验条目的最终状态（通过/丢弃/异常保留/未处理），供状态台账落盘。
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ...log_config import step_logger
from ...llm_infra.batch import batch_units
from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ...llm_infra.config import get_batch_size
from ...llm_infra.resource import RESOURCE_MANAGER, fanout
from ..schemas import SubtitleRow
from ..utils.llm_text_utils import (
    collect_non_analysis_row_ids,
    render_annotations,
    render_windowed_subtitle,
    window_row_ids,
)
from ..utils.row_id_2_time_range import (
    convert_subtitle_l2rawID2subtitle_m,
    convert_row_range_to_time,
    merge_adjacent_citation_ranges,
)
from ..utils.rowid_scope import (
    collect_cited_row_ids,
    filter_range_citations,
    format_scope,
    scope_id_set,
)
from ..analysis_common.post_processor import PostProcessor
from ...utils.analysis_cleanup import _DROP_KEY, clean_analysis

logger = logging.getLogger(__name__)

_FIXED_SECTION_NAMES = {"宏观分析", "交易技巧", "精炼总结", "主讲人暗示重要话题"}

# 投资机会条目的立场子字段键，及「未表态」取值（须与 7_1_主题分析.json 的 观点 枚举逐字一致）
_INVESTMENT_STANCE_KEYS = ("短线", "中长线", "当前价格")
_NO_STANCE = "未表态 🤐"


def _sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "_").replace("\\", "_").replace(":", "_")
        .replace("*", "_").replace("?", "_").replace('"', "_")
        .replace("<", "_").replace(">", "_").replace("|", "_")
    )


def _has_citations(value: Any) -> bool:
    """递归检测 value 是否含非空 引用[] 字段。"""
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "引用" and isinstance(v, list) and v:
                return True
            if isinstance(v, (dict, list)) and _has_citations(v):
                return True
    elif isinstance(value, list):
        for item in value:
            if _has_citations(item):
                return True
    return False


def _investment_is_no_claim(item: Any) -> bool:
    """投资机会条目是否纯「未表态」（无需校验的否定性条目）。

    引用校验 prompt 已明确：`未表态` 的立场子字段允许空引用、也允许指向提及位置，
    且引用非空但原文与该资产无关也不判不支撑——这类条目校验结果必然是「支撑」，
    跳过与校验等价。但「看多/看空/中性/贵/便宜/不贵也不便宜」即使引用为空也必须
    照常校验（校验会把它改回未表态），故跳过条件只要求「所有立场子字段均为未表态」，
    不看引用是否为空。
    """
    if not isinstance(item, dict):
        return False
    for key in _INVESTMENT_STANCE_KEYS:
        sub = item.get(key)
        if not isinstance(sub, dict):
            continue
        if str(sub.get("观点") or "") != _NO_STANCE:
            return False
    return True


class CitationValidator:
    """引用内容支撑性校验组件（被步骤 5 与 7 内联调用）。

    # 每条目只做一次校验调用（不做二轮复验，见模块 docstring 的取舍说明）；
    # 宏观条目不支撑时修复一次、修复产物直接采用——故无需修复轮数计数器。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        evidence_prompt: str | None = None,
        evidence_schema: dict | None = None,
    ) -> None:
        """prompt / schema 缺省时从本模块同目录自加载（`引用校验.txt` / `引用校验_batch.json`）。

        组件资产不与任何步骤共置，故各调用步骤直接 `CitationValidator(client)` 即可，
        无需各自拼路径；显式传参仅用于测试覆盖。
        """
        self._llm_client = llm_client
        self._evidence_prompt = (
            evidence_prompt
            if evidence_prompt is not None
            else (Path(__file__).parent / "引用校验.txt").read_text(encoding="utf-8").strip()
        )
        self._evidence_schema = (
            evidence_schema
            if evidence_schema is not None
            else _load_schema("引用校验_batch", __file__)
        )
        _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("引用校验_batch_user.jinja2")

    def validate(
        self,
        analysis: dict,
        subtitle_l: list[SubtitleRow],
        base_dir: Path,
        stage_prefix: str,
        step_number: int,
        topic_segments: dict[str, list[SubtitleRow]] | None = None,
        sections: set[str] | None = None,
        repair_fn: Callable[[dict, dict, str], dict | None] | None = None,
        result_sink: dict[str, dict] | None = None,
        extra_scope_rows: dict[str, set[int]] | None = None,
    ) -> tuple[dict, list[dict]]:
        """对本步骤产出的 section 做引用支撑性校验并原地定稿。

        Args:
            analysis: 完整 analysis（只校验 `sections` 白名单里的 section，其余不动）
            stage_prefix: 调用方给的本步阶段号，如 "5_3" / "7_2"（用于
                log_name / step_name → llm_profiles 阶段参数匹配）
            step_number: 调用方的步骤号 5 / 7（决定落盘 `llm_output_N`）
            topic_segments: 主题 → 字幕行列表（主题类条目的作用域；非主题步骤传 None）
            sections: 只校验这些 section；None = 本步骤全部可校验 section
        """
        with step_logger(f"第{stage_prefix}步-引用校验"):
            rowID2subtitle_m = (
                convert_subtitle_l2rawID2subtitle_m(subtitle_l) if subtitle_l else {}
            )

            worker_tasks = self._build_worker_tasks(
                analysis, topic_segments or {}, subtitle_l,
                extra_scope_rows, sections,
            )

            if not worker_tasks:
                logger.info("没有待校验条目，跳过")
                return analysis, []

            logger.info("待校验条目 %d 个，并行启动", len(worker_tasks))

            # 同 section 条目合成一批：组内共享语义作用域与模型（topic 组共享
            # topic_for_model，宏观/敏感组均为 None），一次调用校验一整批。
            # 组内条目数超过批大小上限（parameter_config）时按上限切子批。
            # [DATA]/[ANNOTATION] 不再进 key——它们按子批的窗口中心并集渲染
            # （见 _render_batch_context），组内条目本就同作用域同注释口径。
            groups: dict[tuple, list[dict]] = {}
            for task in worker_tasks:
                key = (
                    task["section"], tuple(task["scope_row_ids"]),
                    task["keywords_only_jargon"],
                )
                groups.setdefault(key, []).append(task)
            batch_size = get_batch_size("引用校验")
            sub_batches: list[tuple[int, list[dict]]] = []
            for group in groups.values():
                for sub_idx, ts in enumerate(batch_units(group, batch_size)):
                    sub_batches.append((sub_idx, ts))

            # 按 section 分组，统计每组通过情况
            section_total: dict[str, int] = {}
            section_passed: dict[str, int] = {}
            for task in worker_tasks:
                section = task["section"]
                section_total[section] = section_total.get(section, 0) + 1
                section_passed[section] = section_passed.get(section, 0)

            # 并行执行由全局共享线程池承载，并发上限由 ResourceManager 统一管控
            futures = fanout(stage_prefix, [
                (
                    self._validate_batch_with_retry,
                    (
                        ts, subtitle_l, base_dir, rowID2subtitle_m, stage_prefix,
                        step_number, repair_fn, result_sink, sub_idx,
                    ),
                    {},
                )
                for sub_idx, ts in sub_batches
            ])
            # 状态台账：每个 task 一条最终状态记录（通过/丢弃/异常保留/未处理），
            # 供调用方步骤落盘 {step_number}_validation_status.json
            statuses: list[dict] = []
            for future, (_sub_idx, tasks) in zip(futures, sub_batches):
                try:
                    finals = future.result()  # dict[int, dict | None]：组内索引 → 最终条目
                except Exception as e:
                    logger.warning(
                        "批量校验「%s」异常：%s（%d 条全部保留原条目）",
                        tasks[0]["section"], e, len(tasks),
                    )
                    for task in tasks:
                        statuses.append({
                            "section": task["section"],
                            "路径": task["路径"],
                            "状态": "异常保留",
                            "失败原因": str(e),
                        })
                    continue
                for idx, task in enumerate(tasks):
                    final_item = finals.get(idx, None)
                    if final_item is None:
                        statuses.append({
                            "section": task["section"],
                            "路径": task["路径"],
                            "状态": "未处理（非字典条目）",
                        })
                        continue
                    task["set_item"](final_item)
                    if isinstance(final_item, dict) and _DROP_KEY in final_item:
                        logger.warning(
                            "%s 校验不支撑且无法修复/重写，已标记丢弃：%s",
                            task["label"], final_item.get(_DROP_KEY),
                        )
                        statuses.append({
                            "section": task["section"],
                            "路径": task["路径"],
                            "状态": "丢弃",
                            "失败原因": str(final_item.get(_DROP_KEY) or ""),
                        })
                        continue
                    statuses.append({
                        "section": task["section"],
                        "路径": task["路径"],
                        "状态": "通过",
                    })
                    section_passed[task["section"]] = section_passed.get(task["section"], 0) + 1
                    if section_passed[task["section"]] == section_total[task["section"]]:
                        logger.info(
                            "「%s」全部 %d 条通过验证",
                            task["section"], section_total[task["section"]],
                        )

            # 收尾：物理移除丢弃 sentinel 与历史残留的 内容支撑=false 条目
            # （无幻觉优先，宁缺勿错），失败条目绝不进入 llm_analysis.json
            for rec in clean_analysis(analysis):
                logger.warning(
                    "丢弃校验失败条目：%s（%s）",
                    rec["路径"], rec["失败原因"] or "无失败原因",
                )

            return analysis, statuses

    # ==================== 任务构建 ====================

    def _build_worker_tasks(
        self,
        analysis: dict,
        topic_segments: dict[str, list[SubtitleRow]],
        subtitle_l: list[SubtitleRow],
        extra_scope_rows: dict[str, set[int]] | None = None,
        sections: set[str] | None = None,
    ) -> list[dict]:
        """建每个条目一条 task。

        `[DATA]` 走窗口化（`render_windowed_subtitle`），故 task 只携带**窗口中心**
        `window_center_ids` 与语义作用域 `scope_row_ids`，实际文本在批组装时按批渲染：
        - `scope_row_ids`：**语义作用域**，重写引用的越界过滤基准（不含 ±pad 上下文），
          与迁移前逐字一致——主题类=本主题段行，宏观/敏感类=全部可分析行；
        - `window_center_ids`：**窗口中心**，决定模型看得到哪些行。主题类=本主题段行
          （窗口=段行 ±pad）；非主题类=**本条目自己引用到的行**——它们的语义作用域是
          全场，直接拿去开窗等于不切片；校验一条宏观条目只需要它引用处的上下文。

        `sections` 白名单：只建这些 section 的 task——校验已内联进各生成步骤，
        一步只校验自己产出的 section，绝不越界动别的步骤的产物。
        """
        worker_tasks: list[dict] = []
        skipped_no_claim = 0
        blocked = collect_non_analysis_row_ids(subtitle_l)
        allowed = sections  # None = 全部

        def _want(section: str) -> bool:
            return allowed is None or section in allowed

        def _visible(row_ids: list) -> list:
            out: list = []
            for rid in row_ids:
                try:
                    if int(rid) not in blocked:
                        out.append(rid)
                except (TypeError, ValueError):
                    out.append(rid)
            return out

        all_row_ids = _visible([s.rowID for s in subtitle_l if s.text])

        # ---- 1) 主题条目（投资机会 / 观点，每个条目一条 task）----
        for topic_name, topic_data in analysis.items():
            if topic_name in _FIXED_SECTION_NAMES:
                continue
            if not isinstance(topic_data, dict):
                continue
            if not _want(topic_name):
                continue
            segs = topic_segments.get(topic_name, [])
            topic_row_ids = _visible([s.rowID for s in segs if s.text])
            # 6_1 跨主题搬运补丁：被搬进本主题的单元，其引用行不在本主题段行里，
            # 不补进作用域会被重写过滤判越界 → 条目整条丢弃（已实测：搬 双创 进
            # 科技 时 4 个引用行 0 个落在科技段行内）
            extra = (extra_scope_rows or {}).get(topic_name)
            if extra:
                known = {
                    int(r) for r in topic_row_ids
                    if isinstance(r, (int, str)) and str(r).isdigit()
                }
                topic_row_ids = list(topic_row_ids) + [
                    r for r in sorted(extra) if r not in known and r not in blocked
                ]
            for kind, container in (
                ("投资机会", topic_data.get("投资机会")),
                ("观点", topic_data.get("观点")),
            ):
                if not isinstance(container, list):
                    continue
                for idx in range(len(container)):
                    # 纯「未表态」的投资机会条目无需校验（引用校验 prompt 已定义其为
                    # 必然通过），跳过以省一次 LLM 调用；其余条目照常建 task。
                    if kind == "投资机会" and _investment_is_no_claim(container[idx]):
                        skipped_no_claim += 1
                        continue
                    worker_tasks.append({
                        "kind": kind,
                        "section": topic_name,
                        "label": f"{topic_name} / {kind}[{idx}]",
                        "路径": f"{topic_name}.{kind}[{idx}]",
                        "scope_row_ids": topic_row_ids,
                        # 主题类：窗口中心=本主题段行（窗口即段行 ±pad）
                        "window_center_ids": topic_row_ids,
                        # 主题/宏观/敏感类统一只带无法内联的黑话子集（已内联的进 [DATA] 原文）
                        "keywords_only_jargon": True,
                        "get_item": (lambda d=container, i=idx: d[i]),
                        "set_item": (lambda new, d=container, i=idx: d.__setitem__(i, new)),
                        "topic_for_model": topic_name,
                    })

        # ---- 2) 宏观分析子段（板块涨跌 + 热点板块预测 + 顶层 子段）----
        macro = analysis.get("宏观分析")
        if isinstance(macro, dict) and _want("宏观分析"):
            for sub_key, list_key, label_prefix in (
                ("板块分析", "板块涨跌", "宏观分析 / 板块涨跌"),
                ("热点板块预测", "板块清单", "宏观分析 / 热点板块预测.板块清单"),
            ):
                parent = macro.get(sub_key)
                if not isinstance(parent, dict):
                    continue
                arr = parent.get(list_key)
                if not isinstance(arr, list):
                    continue
                path_prefix = f"宏观分析.{sub_key}.{list_key}"
                for idx in range(len(arr)):
                    item = arr[idx]
                    worker_tasks.append({
                        "kind": "宏观分析",
                        "section": "宏观分析",
                        "label": f"{label_prefix}[{idx}]",
                        "路径": f"{path_prefix}[{idx}]",
                        "scope_row_ids": all_row_ids,
                        # 宏观类：窗口中心=本条目引用到的行（作用域是全场，
                        # 直接开窗等于不切片）
                        "window_center_ids": sorted(
                            collect_cited_row_ids(item) - blocked,
                        ),
                        "keywords_only_jargon": True,
                        "get_item": (lambda d=arr, i=idx: d[i]),
                        "set_item": (lambda new, d=arr, i=idx: d.__setitem__(i, new)),
                        "topic_for_model": None,
                        "sub_key": sub_key,
                        "list_key": list_key,
                        "item_index": idx,
                    })
            # 其它顶层子段（除已枚举的 板块分析 / 热点板块预测 之外）：
            # dict（如资金流向）整段一条 task；分块字段是 list，按块各建一条 task
            for sub_key, sub_val in macro.items():
                if sub_key in ("板块分析", "热点板块预测"):
                    continue
                if isinstance(sub_val, dict):
                    if not _has_citations(sub_val):
                        continue
                    worker_tasks.append({
                        "kind": "宏观分析",
                        "section": "宏观分析",
                        "label": f"宏观分析 / {sub_key}",
                        "路径": f"宏观分析.{sub_key}",
                        "scope_row_ids": all_row_ids,
                        "window_center_ids": sorted(
                            collect_cited_row_ids(sub_val) - blocked,
                        ),
                        "keywords_only_jargon": True,
                        "get_item": (lambda m=macro, k=sub_key: m[k]),
                        "set_item": (lambda new, m=macro, k=sub_key: m.__setitem__(k, new)),
                        "topic_for_model": None,
                        "sub_key": sub_key,
                        "list_key": None,
                        "item_index": None,
                    })
                elif isinstance(sub_val, list):
                    for idx in range(len(sub_val)):
                        item = sub_val[idx]
                        if not isinstance(item, dict) or not _has_citations(item):
                            continue
                        worker_tasks.append({
                            "kind": "宏观分析",
                            "section": "宏观分析",
                            "label": f"宏观分析 / {sub_key}[{idx}]",
                            "路径": f"宏观分析.{sub_key}[{idx}]",
                            "scope_row_ids": all_row_ids,
                            "window_center_ids": sorted(
                                collect_cited_row_ids(item) - blocked,
                            ),
                            "keywords_only_jargon": True,
                            "get_item": (lambda l=sub_val, i=idx: l[i]),
                            "set_item": (lambda new, l=sub_val, i=idx: l.__setitem__(i, new)),
                            "topic_for_model": None,
                            "sub_key": sub_key,
                            "list_key": None,  # 语义：字段本身即列表，修复按 item_index 就地写回
                            "item_index": idx,
                        })

        # ---- 3) 主讲人暗示重要话题（每个 entry 一条 task）----
        sens = analysis.get("主讲人暗示重要话题")
        if isinstance(sens, list) and _want("主讲人暗示重要话题"):
            for idx in range(len(sens)):
                worker_tasks.append({
                    "kind": "主讲人暗示重要话题",
                    "section": "主讲人暗示重要话题",
                    "label": f"主讲人暗示重要话题[{idx}]",
                    "路径": f"主讲人暗示重要话题[{idx}]",
                    "scope_row_ids": all_row_ids,
                    "window_center_ids": sorted(
                        collect_cited_row_ids(sens[idx]) - blocked,
                    ),
                    "keywords_only_jargon": True,
                    "get_item": (lambda d=sens, i=idx: d[i]),
                    "set_item": (lambda new, d=sens, i=idx: d.__setitem__(i, new)),
                    "topic_for_model": None,
                })

        if skipped_no_claim:
            logger.info(
                "7_2 跳过 %d 条纯未表态的投资机会条目，无需校验",
                skipped_no_claim,
            )
        return worker_tasks

    # ==================== 批量重试 ====================

    def _validate_batch_with_retry(
        self,
        tasks: list[dict],
        subtitle_l: list[SubtitleRow],
        base_dir: Path,
        rowID2subtitle_m: dict,
        stage_prefix: str,
        step_number: int,
        repair_fn: Callable[[dict, dict, str], dict | None] | None = None,
        result_sink: dict[str, dict] | None = None,
        sub_idx: int = 0,
    ) -> dict[int, dict | None]:
        """对同 section 的一个子批条目做引用校验（一次 LLM 批量调用，单轮定案）。

        返回 `{组内索引: 最终条目}`：通过 → 原条目；不支撑 → 修复/重写产物；
        无法修复或重写 → `{_DROP_KEY: 原因}` sentinel；非字典条目 → None
        （调用方记「未处理」）。批内条目共享 [DATA]/[ANNOTATION]/[ROWID]/模型，
        只有 [TASK] 的条目清单不同。

        单轮定案：不再有「整批重跑下一轮」——批内一个条目不支撑不会拖着整批
        重新推理（旧实现的 reasoning 放大源）。仅两种情况会额外发起调用，且都
        只针对具体条目、不重复整批：
        - 整批调用失败（retry 重试耗尽）→ 该批条目按 batch_size=1 各单跑一次；
        - `结果` 数组元素缺失/非 dict（模型少给了条目）→ 该条目单跑一次。
        """
        # 每条目独立状态：当前条目 / 已定案结果
        current: dict[int, dict] = {}
        for idx, task in enumerate(tasks):
            item = task["get_item"]()
            if isinstance(item, dict):
                current[idx] = item
        final: dict[int, dict | None] = {}
        for idx, task in enumerate(tasks):
            if idx not in current:
                final[idx] = None  # 非字典条目：未处理

        section_safe = _sanitize_filename(tasks[0]["section"])
        log_dir = get_step_output_dir(base_dir, step_number) if base_dir is not None else None

        # 恒用 flash：本阶段是 reasoning_effort=low 的机械逐条核对（判断引用原文
        # 是否支撑条目结论），不需要更强的推理深度。
        model = "deepseek-v4-flash"

        pending = list(current)
        try:
            result = self._call_validation_batch(
                tasks, current, pending, model, log_dir,
                log_name=f"{stage_prefix}_引用校验_batch_{section_safe}_b{sub_idx}",
                step_name=f"{stage_prefix}_引用校验_batch_{tasks[0]['section']}_b{sub_idx}",
                subtitle_l=subtitle_l,
            )
        except Exception as e:
            # 整批调用失败（retry 重试耗尽）：拆回单条目各跑一次
            logger.warning(
                "批量校验「%s」LLM 调用失败（%d 条）：%s——拆回单条目重跑",
                tasks[0]["section"], len(pending), e,
            )
            for idx in pending:
                self._validate_single(
                    idx, tasks, current, final, model, log_dir, section_safe,
                    sub_idx, rowID2subtitle_m, subtitle_l, stage_prefix,
                    repair_fn, result_sink,
                )
            return final

        if not isinstance(result, dict) or not isinstance(result.get("结果"), list):
            logger.warning(
                "批量校验「%s」返回非 dict 或缺结果数组，%d 条全部标记丢弃",
                tasks[0]["section"], len(pending),
            )
            for idx in pending:
                final[idx] = {_DROP_KEY: "返回非 dict 或缺结果数组"}
            return final

        if result_sink is not None:
            result_sink[f"{stage_prefix}_引用校验_batch_{section_safe}_b{sub_idx}"] = result

        results = result["结果"]
        for pos, idx in enumerate(pending):
            r = results[pos] if pos < len(results) else None
            if not isinstance(r, dict):
                # 模型少给了本条目：单跑一次补齐（只补这一条，不重跑整批）
                self._validate_single(
                    idx, tasks, current, final, model, log_dir, section_safe,
                    sub_idx, rowID2subtitle_m, subtitle_l, stage_prefix,
                    repair_fn, result_sink,
                )
                continue
            self._apply_result(
                idx, r, tasks, current, final,
                rowID2subtitle_m, subtitle_l, stage_prefix, repair_fn,
            )
        return final

    @staticmethod
    def _render_batch_context(
        tasks: list[dict],
        idxs: list[int],
        current: dict[int, dict],
        subtitle_l: list[SubtitleRow],
    ) -> tuple[str, str]:
        """按本批条目的窗口中心并集渲染 (subtitle_text, annotation_text)。

        窗口中心取**当前条目**（`current`，可能已是重写产物）的引用行——宏观类
        条目引用变了，窗口也该跟着变。主题类的窗口中心是本主题段行，与条目内容
        无关，直接用 task 里的静态值。

        [ANNOTATION] 用 `window_row_ids` 切到与 [DATA] 完全相同的行集：注释块
        只应描述 [DATA] 里出现过的行，否则模型会为看不见的行读到注释。
        """
        centers: set[int] = set()
        for idx in idxs:
            static_center = scope_id_set(tasks[idx].get("window_center_ids") or [])
            # 宏观/敏感类按当前条目实际引用行开窗（重写后引用会变）；主题类用
            # 静态段行。任一侧为空时用另一侧兜底——主题段行可能为空（4_x 未给
            # 该主题分到段），此时退回条目自己的引用行，仍是有效窗口。
            if tasks[idx]["topic_for_model"]:
                centers.update(
                    static_center or collect_cited_row_ids(current.get(idx) or {}),
                )
            else:
                centers.update(
                    collect_cited_row_ids(current.get(idx) or {}) or static_center,
                )

        subtitle_text = render_windowed_subtitle(
            subtitle_l, centers, with_type=True,
        )
        # 注释块切到与 [DATA] 完全相同的行集。centers 全空时 render_windowed_subtitle
        # 已兜底为完整字幕，window_row_ids 同步返回全部行——两者始终一致。
        visible = window_row_ids(subtitle_l, centers)
        annotation_text = render_annotations(
            subtitle_l,
            include_keywords=True,
            keywords_only_jargon=tasks[idxs[0]]["keywords_only_jargon"],
            row_scope=visible,
        )
        return subtitle_text, annotation_text

    def _call_validation_batch(
        self,
        tasks: list[dict],
        current: dict[int, dict],
        idxs: list[int],
        model: str,
        log_dir: Path | None,
        log_name: str,
        step_name: str,
        subtitle_l: list[SubtitleRow],
    ) -> dict:
        """渲染 idxs 对应条目清单并调校验 LLM（批量与单条目回退共用同一渲染路径）。"""
        items = [
            {"类型": tasks[idx]["kind"], "条目": current[idx]}
            for idx in idxs
        ]
        subtitle_text, annotation_text = self._render_batch_context(
            tasks, idxs, current, subtitle_l,
        )
        # [ROWID] = 本批条目的语义作用域并集。冗余上下文行使 [DATA] ⊃ 作用域，
        # 必须显式声明重写引用能落在哪些行——否则模型可能引用冗余行，
        # 而 `filter_range_citations` 会把这类引用丢掉，重写反而失去支撑。
        scope: set[int] = set()
        for idx in idxs:
            scope |= scope_id_set(tasks[idx]["scope_row_ids"])
        user_prompt = self._user_template.render(
            task_rules=self._evidence_prompt,
            items=items,
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
            rowid_scope=format_scope(scope),
        )
        return self._llm_client.call(
            user_prompt=user_prompt,
            model=model,
            temperature=0.0,
            schema=self._evidence_schema,
            log_dir=log_dir,
            log_name=log_name,
            step_name=step_name,
        )

    def _validate_single(
        self,
        idx: int,
        tasks: list[dict],
        current: dict[int, dict],
        final: dict[int, dict | None],
        model: str,
        log_dir: Path | None,
        section_safe: str,
        sub_idx: int,
        rowID2subtitle_m: dict,
        subtitle_l: list[SubtitleRow],
        stage_prefix: str,
        repair_fn: Callable[[dict, dict, str], dict | None] | None,
        result_sink: dict[str, dict] | None,
    ) -> None:
        """拆批回退：单条目一次校验调用（batch_size=1 走同一渲染路径）并定案。

        本方法只在整批调用失败或模型漏给该条目时被调用，且**只跑一次**；
        跑完仍拿不到可用结果 → 丢弃 sentinel（不再有下一轮兜底）。
        """
        task = tasks[idx]
        try:
            result = self._call_validation_batch(
                tasks, current, [idx], model, log_dir,
                log_name=f"{stage_prefix}_引用校验_batch_{section_safe}_b{sub_idx}_s{idx}",
                step_name=f"{stage_prefix}_引用校验_single_{task['label']}",
                subtitle_l=subtitle_l,
            )
        except Exception as e:
            logger.warning(
                "%s 单条目校验 LLM 调用失败：%s，标记丢弃", task["label"], e,
            )
            final[idx] = {_DROP_KEY: f"LLM 调用失败：{e}"}
            return
        if not isinstance(result, dict) or not isinstance(result.get("结果"), list):
            logger.warning(
                "%s 单条目校验返回非 dict 或缺结果数组，标记丢弃", task["label"],
            )
            final[idx] = {_DROP_KEY: "返回非 dict 或缺结果数组"}
            return
        if result_sink is not None:
            result_sink[
                f"{stage_prefix}_引用校验_batch_{section_safe}_b{sub_idx}_s{idx}"
            ] = result
        results = result["结果"]
        r = results[0] if results else None
        if not isinstance(r, dict):
            logger.warning(
                "%s 单条目校验结果数组为空或元素非 dict，标记丢弃", task["label"],
            )
            final[idx] = {_DROP_KEY: "结果数组长度不足或元素非 dict"}
            return
        self._apply_result(
            idx, r, tasks, current, final,
            rowID2subtitle_m, subtitle_l, stage_prefix, repair_fn,
        )

    def _apply_result(
        self,
        idx: int,
        r: dict,
        tasks: list[dict],
        current: dict[int, dict],
        final: dict[int, dict | None],
        rowID2subtitle_m: dict,
        subtitle_l: list[SubtitleRow],
        stage_prefix: str,
        repair_fn: Callable[[dict, dict, str], dict | None] | None,
    ) -> None:
        """处理单条目校验结果并定案（支撑/重跑修复/重写/丢弃），批量与单跑共用。

        单轮语义：本方法返回时 `final[idx]` 必已写入——要么原条目、要么修复/重写
        产物、要么丢弃 sentinel，不存在「留到下一轮」的中间态。
        """
        task = tasks[idx]
        if r.get("是否支撑") is True:
            final[idx] = current[idx]
            return
        failure = r.get("失败原因") or "未提供失败原因"

        # 调用方给了 repair_fn（现为步骤 5 宏观分析：重跑该字段组）时优先走它——
        # 重新生成比接受「字幕照抄式重写」更贴合质量要求（既定要求）。
        # 修复产物在 repair_fn 内已做 行号过滤 → 行号→时间 → 原文填充。
        if repair_fn is not None:
            try:
                repaired = repair_fn(task, current[idx], failure)
            except Exception as e:
                logger.warning(
                    "%s 重跑修复异常：%s，退回重写路径", task["label"], e,
                )
                repaired = None
            if isinstance(repaired, dict):
                logger.info(
                    "%s 不支撑（%s），已重跑生成阶段并采用修复产物",
                    task["label"], failure,
                )
                final[idx] = repaired
                return

        rewritten = self._unwrap_rewritten(r.get("重写后条目"), task)
        if not isinstance(rewritten, dict) or not rewritten:
            # 缺重写 / 空重写（观点类即「无支撑且无重写价值」）→ 丢弃，
            # 由 validate() 收尾 clean_analysis 移除
            logger.warning(
                "%s 不支撑（%s）且无可用重写后条目，标记丢弃",
                task["label"], failure,
            )
            final[idx] = {_DROP_KEY: failure}
            return
        if not self._rewritten_shape_ok(rewritten, task):
            # 结构与输入条目不同构（如缺 标题/标的名称）→ 宁可丢弃也不要写进
            # analysis：无名条目会在渲染层用空串跟所有观点做子串匹配，把整个
            # section 的观点全吸到一行上（已发生过，见 summarize_test_example）
            logger.warning(
                "%s 不支撑（%s），但重写后条目结构非法（缺名称字段），标记丢弃",
                task["label"], failure,
            )
            final[idx] = {_DROP_KEY: f"{failure}（重写后条目结构非法）"}
            return

        # [ROWID] 校验：重写时模型看得到完整字幕，引用必须仍落在本条目作用域内。
        # 该阶段的引用是 {开始行号, 结束行号} 区间形态，走 range 版过滤。
        filter_range_citations(
            rewritten,
            scope_id_set(task["scope_row_ids"]),
            f"{stage_prefix} {task['label']} 重写",
        )
        self._refill_citations(rewritten, rowID2subtitle_m, subtitle_l)
        logger.info(
            "%s 不支撑（%s），已采用 %s 重写后条目",
            task["label"], failure, stage_prefix,
        )
        final[idx] = rewritten

    # ==================== 重写产物的形状归一与校验 ====================

    # 条目名称字段：投资机会/宏观分块用 标题，观点用 标的名称
    _NAME_KEY_BY_KIND = {
        "投资机会": "标题",
        "观点": "标的名称",
    }

    @staticmethod
    def _unwrap_rewritten(rewritten: Any, task: dict) -> Any:
        """剥掉模型误加的 `{类型, 条目}` 外壳。

        校验的输入清单每项是 `{"类型": ..., "条目": {...}}`（批量协议的包裹），
        模型重写时偶发把这层外壳一起复述回来。若不剥，写进 analysis 的条目就成了
        `{类型, 条目}`——没有 `标题`，渲染层拿空串去跟所有观点做双向子串匹配，
        整个 section 的观点会被全部吸到这一行（实测 17 条观点挤进一行、名称显示"无"）。
        """
        if (
            isinstance(rewritten, dict)
            and set(rewritten) == {"类型", "条目"}
            and isinstance(rewritten.get("条目"), dict)
        ):
            logger.warning(
                "%s 重写后条目带了 {类型, 条目} 外壳，已剥离取内层条目", task["label"],
            )
            return rewritten["条目"]
        return rewritten

    @classmethod
    def _rewritten_shape_ok(cls, rewritten: dict, task: dict) -> bool:
        """重写产物必须带非空名称字段（与输入条目同构）。

        无名条目一旦进入 analysis，渲染层的观点匹配会退化成"空串匹配一切"，
        危害远大于丢掉这一条，故宁缺勿错。宏观/敏感类条目名称字段各异，
        只对已知的两类（投资机会/观点）强校验。
        """
        name_key = cls._NAME_KEY_BY_KIND.get(task.get("kind"))
        if name_key is None:
            return True
        return bool(str(rewritten.get(name_key) or "").strip())

    # ==================== 引用回填 ====================

    @staticmethod
    def _refill_citations(
        item: dict,
        rowID2subtitle_m: dict,
        subtitle_l: list[SubtitleRow],
    ) -> None:
        """递归把 item 内所有 引用[] 重新做 行号→时间 转换 + 原文填充（in-place）。"""

        def _walk(value: Any) -> None:
            if isinstance(value, dict):
                for k, v in list(value.items()):
                    if k == "引用" and isinstance(v, list):
                        # 先把 LLM 重写的碎片区间（间隙 ≤ 2 行）合并为整段，再转时间
                        ranges = merge_adjacent_citation_ranges(v)
                        new_list: list = []
                        for cit in ranges:
                            converted = convert_row_range_to_time(cit, rowID2subtitle_m)
                            if not converted:
                                continue
                            for c in converted:
                                start_t = c.get("开始时间", "")
                                end_t = c.get("结束时间", "")
                                if start_t and end_t:
                                    c["原文"] = PostProcessor._extract_subtitle_text(
                                        subtitle_l, start_t, end_t,
                                    )
                                new_list.append(c)
                        value[k] = new_list
                    elif isinstance(v, (dict, list)):
                        _walk(v)
            elif isinstance(value, list):
                for it in value:
                    _walk(it)

        _walk(item)
