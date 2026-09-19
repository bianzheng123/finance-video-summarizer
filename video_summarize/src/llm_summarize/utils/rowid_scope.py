"""行号作用域工具：区间渲染 + 越界校验。

作用域**由代码侧过滤 `[DATA]` 表达**（`render_windowed_subtitle` 只渲染工作
范围 ± pad 行），不再有 `[ROWID]` 块告诉模型"本次只负责哪些行"。块序为
`[TASK]→[DATA]→[ANNOTATION]→[DATABASE]`，静态的 `[TASK]` 置于
最前以最大化前缀缓存命中。

两类职责：

- **区间渲染**（`format_scope` / `merge_to_ranges`）：把离散 rowID 合并成
  `[{"start": int, "end": int}, ...]` 闭区间数组。`[ROWID]` 块已删除，但
  `[TASK]` 内的清单仍需要区间形态（如 5_3 逐主题的「行号范围」/「兄弟主题
  区间」、1_2 的两段区间），故保留。
- **越界过滤**（`filter_*`）：**一条都不能省**。窗口化后模型仍看得到 ±pad
  的上下文行，仍可能对它们作答；批量调用拆回时还要靠 `union` 参数区分
  "兄弟单元的行"（静默跳过）与"真越界"（告警丢弃）。
"""

import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


def merge_to_ranges(row_ids: Iterable[Any]) -> list[tuple[int, int]]:
    """把离散 rowID 合并为连续闭区间列表：[3,4,5,8,10,11] → [(3,5),(8,8),(10,11)]。"""
    ids: set[int] = set()
    for r in row_ids or []:
        try:
            ids.add(int(r))
        except (TypeError, ValueError):
            continue
    if not ids:
        return []
    sorted_ids = sorted(ids)
    ranges: list[tuple[int, int]] = []
    start = end = sorted_ids[0]
    for r in sorted_ids[1:]:
        if r == end + 1:
            end = r
        else:
            ranges.append((start, end))
            start = end = r
    ranges.append((start, end))
    return ranges


def format_scope(row_ids: Iterable[Any]) -> list[dict]:
    """把 rowID 集合渲染成区间数组：`[{"start": 12, "end": 30}, ...]`。

    供 `[TASK]` 内的行号清单使用（5_3 逐主题「行号范围」/「兄弟主题区间」、
    1_2 的两段区间等）。统一用 JSON 区间数组而非 `12-30, 45` 这类紧凑串：
    模型对 JSON 的边界解析比对自定义分隔符稳定得多，且与各阶段 schema 里的
    结构化输出保持同一形态。单行也照样写成 `{"start": N, "end": N}`，不做
    特例——形态一致比省字节重要。
    """
    return [{"start": s, "end": e} for s, e in merge_to_ranges(row_ids)]


def format_range_scope(row_id_start: int, row_id_end: int) -> list[dict]:
    """连续区间的快捷渲染（分块阶段用块首尾 rowID 直接给出）。

    仅 1_1 / 1_2 尚在使用（它们的 `[ROWID]` 块还未拆除）；两阶段完成窗口化
    改造后本函数即可删除。
    """
    return [{"start": row_id_start, "end": row_id_end}]


def scope_id_set(row_ids: Iterable[Any]) -> set[int]:
    """把任意 rowID 可迭代对象规范成 int 集合，供越界判定使用。"""
    out: set[int] = set()
    for r in row_ids or []:
        try:
            out.add(int(r))
        except (TypeError, ValueError):
            continue
    return out


def collect_cited_row_ids(value: Any) -> set[int]:
    """递归收集任意结构里所有 `引用[]` 覆盖到的 rowID（闭区间展开）。

    步骤 5 之后的条目（投资机会 / 观点 / 宏观子段 / 敏感信号）都把引用挂在
    `引用[]` 下、形态为 `{开始行号, 结束行号}`（尚未转时间）。步骤 5/7 的引用校验
    用它算窗口中心：判定「这些条目该不该合并 / 引用是否支撑」只需要条目引用处的
    字幕，不需要全场。缺行号或已转成时间形态的引用自然跳过（取不到行号）。
    """
    out: set[int] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "引用" and isinstance(v, list):
                    for cit in v:
                        if not isinstance(cit, dict):
                            continue
                        try:
                            start = int(cit["开始行号"])
                            end = int(cit["结束行号"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if start > end:
                            start, end = end, start
                        out.update(range(start, end + 1))
                elif isinstance(v, (dict, list)):
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(value)
    return out


# ==================== 越界过滤 ====================


def filter_row_ids(
    row_ids: Iterable[Any],
    allowed: set[int],
    label: str,
    union: set[int] | None = None,
) -> list[int]:
    """过滤裸 rowID 数组，丢弃不在 allowed 里的行号并告警。

    `union` 为本次 LLM 调用 [ROWID] 的并集（批量调用时批内全部单元的行号）：
    落在 union 内但不属于本单元的行是兄弟单元的行，静默跳过不告警；
    只有 union 外的行才是真越界。union=None 时维持旧行为（全部告警）。
    """
    kept: list[int] = []
    dropped: list = []
    for r in row_ids or []:
        try:
            rid = int(r)
        except (TypeError, ValueError):
            dropped.append(r)
            continue
        if rid in allowed:
            kept.append(rid)
        elif union is not None and rid in union:
            continue
        else:
            dropped.append(rid)
    if dropped:
        logger.warning(
            "%s：丢弃 %d 个越界行号 %s",
            label, len(dropped), dropped[:10],
        )
    return kept


def filter_items_by_rowid(
    items: Iterable[Any],
    allowed: set[int],
    label: str,
    key: str = "rowID",
    union: set[int] | None = None,
) -> list[dict]:
    """过滤 `[{rowID: int, ...}]` 形态的条目列表，丢弃 rowID 越界的条目并告警。

    `union` 语义同 `filter_row_ids`：union 内但不属本单元（allowed）的条目
    是兄弟单元的，静默跳过；只有 union 外的条目才是真越界。
    """
    kept: list[dict] = []
    dropped: list = []
    for item in items or []:
        if not isinstance(item, dict):
            dropped.append(item)
            continue
        try:
            rid = int(item.get(key))
        except (TypeError, ValueError):
            dropped.append(item.get(key))
            continue
        if rid in allowed:
            kept.append(item)
        elif union is not None and rid in union:
            continue
        else:
            dropped.append(rid)
    if dropped:
        logger.warning(
            "%s：丢弃 %d 条越界条目（%s=%s）",
            label, len(dropped), key, dropped[:10],
        )
    return kept


def filter_range_citations(obj: Any, allowed: set[int], label: str) -> int:
    """递归就地过滤 `引用` 里的 `{开始行号, 结束行号}` 区间，丢弃与 allowed 无交集的区间。

    引用校验的重写阶段模型看到的是已转过一次的区间形态引用（而非整数数组），所以需要这条
    独立路径。与 allowed 有交集的区间会被收缩到交集的首尾行号；完全无交集的整条丢弃。
    """
    if not allowed:
        return 0
    dropped_total = 0
    dropped_sample: list[tuple] = []

    def _clip(cit: dict) -> dict | None:
        try:
            s = int(cit["开始行号"])
            e = int(cit["结束行号"])
        except (KeyError, TypeError, ValueError):
            return cit  # 非区间形态（例如已转成时间），交给别处处理
        if s > e:
            s, e = e, s
        inside = [r for r in allowed if s <= r <= e]
        if not inside:
            return None
        return {**cit, "开始行号": min(inside), "结束行号": max(inside)}

    def _walk(node: Any) -> None:
        nonlocal dropped_total
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if k == "引用" and isinstance(v, list):
                    kept: list = []
                    for cit in v:
                        if not isinstance(cit, dict):
                            kept.append(cit)
                            continue
                        clipped = _clip(cit)
                        if clipped is None:
                            dropped_total += 1
                            if len(dropped_sample) < 10:
                                dropped_sample.append(
                                    (cit.get("开始行号"), cit.get("结束行号")),
                                )
                        else:
                            kept.append(clipped)
                    node[k] = kept
                elif isinstance(v, (dict, list)):
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(obj)
    if dropped_total:
        logger.warning(
            "%s：丢弃 %d 个越界引用区间 %s",
            label, dropped_total, dropped_sample,
        )
    return dropped_total


def filter_citations(
    obj: Any,
    allowed: set[int],
    label: str,
    union: set[int] | None = None,
) -> int:
    """递归就地过滤 obj 里所有 `引用` 整数数组，丢弃越界 rowID。返回丢弃总数。

    只处理元素为整数的 `引用` 数组（即 LLM 刚输出、尚未转成时间区间的形态）；
    已经是 `{开始时间, 结束时间}` 字典的引用不受影响。
    `union` 语义同 `filter_row_ids`：union 内但不属本主题（allowed）的引用
    是兄弟主题的，静默跳过；只有 union 外的才是真越界。
    """
    dropped_total = 0
    dropped_sample: list[int] = []

    def _walk(node: Any) -> None:
        nonlocal dropped_total
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if k == "引用" and isinstance(v, list):
                    kept: list = []
                    for r in v:
                        if isinstance(r, bool) or not isinstance(r, int):
                            kept.append(r)
                            continue
                        if r in allowed:
                            kept.append(r)
                        elif union is not None and r in union:
                            continue
                        else:
                            dropped_total += 1
                            if len(dropped_sample) < 10:
                                dropped_sample.append(r)
                    node[k] = kept
                elif isinstance(v, (dict, list)):
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(obj)
    if dropped_total:
        logger.warning(
            "%s：引用数组丢弃 %d 个越界行号 %s",
            label, dropped_total, dropped_sample,
        )
    return dropped_total
