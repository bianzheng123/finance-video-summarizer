"""批量 LLM 调用的共享工具：分组、结果拆分与整批失败回退编排。

各步骤「一次调用处理多个工作单元」的公共底座（纯函数 + 一个编排器）：
  - batch_units             稳定排序 + 连续切块（每块 ≤ batch_size）
  - round_robin_groups      round-robin 分组（4_1/4_2/4_3 共用的 G 组并行语义）
  - split_batch_result      「数组 + 键字段」形态的批量输出拆回 per-key 字典
  - run_batch_with_fallback 整批失败 → 逐单元并行单跑；批内个别单元缺失/校验失败 → 单跑一次

设计约束（CLAUDE.md 设计哲学）：
  - 分组必须确定性（稳定排序），批 log_name 由单元序列枚举，LLM_REPLAY 回放才可复现；
  - 失败回退保证「一个坏单元不牵连同批其余单元」，断点粒度仍是单元级；
  - 「业务对象 + 结果概要」的状态日志由各步骤的 run_batch/run_unit 内部打，
    编排器只打回退触发与失败收口的 WARNING。
"""

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import as_completed
from typing import Any, TypeVar

from .resource import RESOURCE_MANAGER

logger = logging.getLogger(__name__)

T = TypeVar("T")


def batch_units(
    units: list[T],
    batch_size: int,
    sort_key: Callable[[T], Any] | None = None,
) -> list[list[T]]:
    """稳定排序 + 连续切块：每块 ≤ batch_size。

    与 round_robin_groups 的语义差异：连续切块把相邻单元**聚拢**（同类单元
    同批利于模型局部聚焦），round-robin 把相似单元**拆开**（组间负载平衡）。
    批量化的目的是「每次调用的单元数上限」，用本函数；4_x 的 G 组并行用后者。
    sort_key 为 None 时保持传入顺序（调用方需保证传入序已确定）。
    """
    batch_size = max(1, int(batch_size))
    if not units:
        return []
    ordered = sorted(units, key=sort_key) if sort_key is not None else list(units)
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def round_robin_groups(
    items: list[T],
    group_count: int,
    sort_key: Callable[[T], Any] | None = None,
) -> list[list[T]]:
    """排序后 round-robin 分组（4_1/4_2/4_3 共用：G 组并行、组内批量）。

    排序 + 轮转让相似单元尽量落入不同组（组间负载/命中率平衡），
    组数 = 并行度，每组一次 LLM 调用。
    """
    group_count = max(1, int(group_count))
    ordered = sorted(items, key=sort_key) if sort_key is not None else list(items)
    groups: list[list[T]] = [[] for _ in range(group_count)]
    for i, item in enumerate(ordered):
        groups[i % group_count].append(item)
    return groups


def split_batch_result(
    result: Any,
    array_key: str,
    key_field: str,
    expected_keys: Iterable[str],
) -> tuple[dict[str, Any], list[str]]:
    """拆「{array_key: [{key_field: ..., ...}]}」形态的批量输出为 per-key 字典。

    - result 非 dict / array_key 缺失或非 list → 整批无效，返回 ({}, 全部 expected)；
    - 数组元素非 dict 或缺 key_field → 丢弃该元素 + WARNING（不阻塞其他元素）；
    - key 重复 → 取首个 + WARNING；key 不在 expected_keys 内 → 丢弃 + WARNING（防编造）。
    返回 (命中映射, 缺失 keys)——缺失 keys 由调用方按回退语义单跑。
    """
    expected = list(expected_keys)
    if not isinstance(result, dict):
        logger.warning(
            "批量输出不是 JSON 对象（%s），%d 个单元全部按缺失处理",
            type(result).__name__, len(expected),
        )
        return {}, expected
    array = result.get(array_key)
    if not isinstance(array, list):
        logger.warning(
            "批量输出缺「%s」数组，%d 个单元全部按缺失处理", array_key, len(expected),
        )
        return {}, expected
    mapping: dict[str, Any] = {}
    for item in array:
        if not isinstance(item, dict):
            logger.warning("批量输出元素非 JSON 对象：%r，丢弃", item)
            continue
        key = item.get(key_field)
        if key is None:
            logger.warning("批量输出元素缺键「%s」，丢弃", key_field)
            continue
        if key in mapping:
            logger.warning("批量输出键「%s」重复，保留首个", key)
            continue
        if key not in expected:
            logger.warning("批量输出键「%s」不在本批范围内，丢弃", key)
            continue
        mapping[key] = item
    missing = [k for k in expected if k not in mapping]
    return mapping, missing


def _run_units_parallel(
    units: list[T],
    unit_key: Callable[[T], str],
    run_unit: Callable[[T], Any],
    validate_unit: Callable[[T, Any], bool],
    batch_label: str,
    results: dict[str, Any],
    failures: dict[str, Exception],
) -> None:
    """逐单元并行单跑（回退路径），结果并入 results / failures。"""
    if len(units) == 1:
        unit = units[0]
        key = unit_key(unit)
        try:
            payload = run_unit(unit)
        except Exception as exc:
            logger.warning("%s 单元「%s」单跑失败：%s", batch_label, key, exc)
            failures[key] = exc
            return
        if validate_unit(unit, payload):
            results[key] = payload
        else:
            # 调用方（2_1/3_1）对校验失败按"无产出"降级接受，属预期路径，不告警
            logger.info("%s 单元「%s」单跑结果未通过语义校验，按无产出接受", batch_label, key)
            failures[key] = ValueError("单跑结果未通过语义校验")
        return
    futures = {RESOURCE_MANAGER.submit(run_unit, unit): unit for unit in units}
    for future in as_completed(futures):
        unit = futures[future]
        key = unit_key(unit)
        try:
            payload = future.result()
        except Exception as exc:
            logger.warning("%s 单元「%s」单跑失败：%s", batch_label, key, exc)
            failures[key] = exc
            continue
        if validate_unit(unit, payload):
            results[key] = payload
        else:
            # 调用方（2_1/3_1）对校验失败按"无产出"降级接受，属预期路径，不告警
            logger.info("%s 单元「%s」单跑结果未通过语义校验，按无产出接受", batch_label, key)
            failures[key] = ValueError("单跑结果未通过语义校验")


def run_batch_with_fallback(
    units: list[T],
    batch_size: int,
    unit_key: Callable[[T], str],
    run_batch: Callable[[list[T]], dict[str, Any]],
    run_unit: Callable[[T], Any],
    validate_unit: Callable[[T, Any], bool] | None = None,
    batch_label: str = "",
) -> tuple[dict[str, Any], dict[str, Exception]]:
    """批量执行 + 拆批回退编排。

    - 整批 run_batch raise（重试耗尽）→ 该批全部单元转 run_unit 并行单跑；
    - 批成功但某单元 key 缺失或 validate_unit 不过 → 该单元单跑一次；
    - 单跑仍失败/仍不过校验 → 记入 failures（含异常），由调用方按各步现有
      失败语义处理（步骤 5 收集后统一 raise；2/3/6 步按 None/空列表降级）。
    run_batch 闭包内部负责渲染 + call/call_conversation + split_batch_result，
    返回 {unit_key(unit): payload}；validate_unit(unit, payload) 为各步自定的
    业务语义校验，None 表示只查 key 缺失。
    返回 (results: key → 结果, failures: key → Exception)。
    """
    results: dict[str, Any] = {}
    failures: dict[str, Exception] = {}
    check = validate_unit if validate_unit is not None else (lambda unit, payload: True)
    label = batch_label or "批量"
    for batch in batch_units(units, batch_size, sort_key=unit_key):
        fallback_units: list[T] = []
        try:
            batch_result = run_batch(batch)
        except Exception as exc:
            logger.warning(
                "%s 整批调用失败（%d 单元）：%s——拆回逐单元单跑",
                label, len(batch), exc,
            )
            fallback_units = batch
            batch_result = {}
        if not fallback_units:
            for unit in batch:
                key = unit_key(unit)
                payload = batch_result.get(key)
                if payload is not None and check(unit, payload):
                    results[key] = payload
                    continue
                if payload is None:
                    logger.warning("%s 单元「%s」在批量输出中缺失，单跑一次", label, key)
                else:
                    logger.warning("%s 单元「%s」语义校验失败，单跑一次", label, key)
                fallback_units.append(unit)
        if fallback_units:
            _run_units_parallel(fallback_units, unit_key, run_unit, check, label, results, failures)
    return results, failures
