"""LLM 回放客户端：从 llm_output_N/{log_name}/ 的历史日志读取已校验结果，零 token 重放。

用途：修改管线代码后，用 LLM_REPLAY=1 重跑总结管线，LLM 调用不再触网，
直接从历史日志（概览.json + round{r}__attempt{n}.json）回放结果——
LLM 与 WSA 网络一并绕开，回放路径完全不写盘，不会污染作为"录制"的历史日志。

匹配规则（按优先级）：
  1. 精确命中 {log_dir}/{log_name}/概览.json → 直接使用（绝不剥后缀）；
  2. 未命中且 log_name 形如 "<base>_<6位hex>" → 剥后缀归一化后建候选队列，
     优先选 user_prompt 与本次完全一致的候选，否则按目录名消费队首。
     该形态是**历史遗留**：旧宏观修复用 `_repair_{字段}_{uuid6}` 带随机 hex 后缀
     （现已改为可读 log_name、不再带 hex），此路径仅为回放旧 golden 保留；
  3. 其余未命中 → ReplayMissError（信息里列出相近的已记录目录名）。

注意：LLM_REPLAY 不要写进 .env——client.py 的 load_dotenv(override=True)
会在 src 导入时执行，覆盖脚本预先注入的值，导致开关静默失效。
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 历史遗留：旧 5_1 修复调用 log_name 的随机 hex 后缀
# （"5_1_宏观分析_repair_仓位建议_08cb98"）。新实现已用可读名、不带 hex。
_REPAIR_HEX_RE = re.compile(r"^(?P<base>.+_repair_.+)_[0-9a-f]{6}$")
_ROUND_FILE_RE = re.compile(r"^round(\d+)__attempt(\d+)\.json$")
_OVERVIEW_NAME = "概览.json"

# 模糊匹配候选消费队列：(str(log_dir), 归一化名) -> 候选列表。
# 进程级生命周期，支持同一 log_name 的多次逻辑调用：每次命中消费一个元素，
# 保证每次调用对应用一个录制文件夹。
_QUEUE_REGISTRY: dict[tuple[str, str], list[dict[str, Any]]] = {}


def replay_enabled() -> bool:
    """回放开关：环境变量 LLM_REPLAY=1/true/yes 时开启。

    每次调用时读取（而非 import 时快照）：client.py 的 load_dotenv 在其
    import 之后才执行，快照会漏掉 .env 里的配置。
    """
    return os.getenv("LLM_REPLAY", "").strip().lower() in ("1", "true", "yes")


class ReplayMissError(RuntimeError):
    """回放未命中或记录不可用。信息中带 log_dir/log_name 与相近候选目录名。"""


def replay_call(
    user_prompt: str, log_dir: Path | None, log_name: str | None,
) -> dict:
    """单轮 call 回放，返回概览.结果（与 live 返回值等价：已过 schema 校验）。

    状态非「成功」→ ReplayMissError（live 语义是最终 raise）。
    """
    folder = _resolve_folder(user_prompt, log_dir, log_name)
    overview = _read_overview(folder)
    if overview.get("状态") != "成功":
        raise ReplayMissError(
            f"回放记录状态为「{overview.get('状态')}」而非成功："
            f"{_fmt_path(log_dir, log_name)}"
            f"（失败原因：{overview.get('失败原因', '无')}）",
        )
    return overview.get("结果")


def replay_call_conversation(
    user_prompt: str,
    schema: dict | None,
    log_dir: Path | None,
    log_name: str | None,
) -> tuple[dict | str | None, list[dict]]:
    """多轮 call_conversation 回放，返回 (结果, trace)。

    状态=成功 → (结果, trace)；schema 为 None 时结果取「输出」字符串
    （live 在该形态下返回正文字符串）。
    状态=失败 → (None, trace)——live 语义是轮数用尽/schema 未过时返回 None
    而管线继续，回放不在此处抛出。
    """
    folder = _resolve_folder(user_prompt, log_dir, log_name)
    overview = _read_overview(folder)
    trace = _build_trace(folder)
    if overview.get("状态") != "成功":
        return None, trace
    result = overview.get("结果")
    if result is None and schema is None:
        result = overview.get("输出", "")
    return result, trace


def _resolve_folder(
    user_prompt: str, log_dir: Path | None, log_name: str | None,
) -> Path:
    """按 精确 → repair 模糊 → 报错 的顺序定位回放文件夹。"""
    if log_dir is None or not log_name:
        raise ReplayMissError(
            "回放模式需要 log_dir 与 log_name（live 未落盘的调用没有记录可回放）",
        )
    exact = log_dir / log_name / _OVERVIEW_NAME
    if exact.exists():
        _warn_if_prompt_changed(log_dir, log_name, user_prompt)
        return exact.parent

    m = _REPAIR_HEX_RE.match(log_name)
    if m is None:
        raise ReplayMissError(
            f"无回放记录：{_fmt_path(log_dir, log_name)}；"
            f"相近目录：{_nearest_names(log_dir, log_name)}",
        )
    base = m.group("base")
    key = (str(log_dir), base)
    queue = _QUEUE_REGISTRY.get(key)
    if queue is None:
        queue = _build_candidate_queue(log_dir, base)
        _QUEUE_REGISTRY[key] = queue
    if not queue:
        raise ReplayMissError(
            f"回放记录已耗尽（同 key 调用次数超过录制数）：{_fmt_path(log_dir, log_name)}",
        )
    for i, entry in enumerate(queue):
        if entry["prompt"] == user_prompt:
            return queue.pop(i)["folder"]
    logger.warning(
        "回放模糊匹配无 prompt 完全一致者，按序消费队首：%s",
        _fmt_path(log_dir, log_name),
    )
    return queue.pop(0)["folder"]


def _build_candidate_queue(log_dir: Path, base: str) -> list[dict[str, Any]]:
    """收集目录名等于 base 或归一化后等于 base 的候选文件夹，按目录名排序。

    每个候选缓存其 概览.输入.user_prompt（修复各轮 repair_targets 不同，
    prompt 是消费队列内的排他匹配依据）。
    """
    candidates: list[dict[str, Any]] = []
    for child in sorted(log_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or not (child / _OVERVIEW_NAME).is_file():
            continue
        m = _REPAIR_HEX_RE.match(child.name)
        if not (child.name == base or (m and m.group("base") == base)):
            continue
        overview = _read_overview(child)
        candidates.append({
            "folder": child,
            "prompt": (overview.get("输入") or {}).get("user_prompt", ""),
        })
    return candidates


def _build_trace(folder: Path) -> list[dict]:
    """按 round 文件重建 trace（与 live 的 call_conversation trace 同构）。

    只取状态=成功且带工具调用的尝试；多工具并行执行时结果按原顺序回灌，
    日志的 工具调用 与 工具结果 一一对应，直接 zip。
    自然排序：round2 < round10。
    """
    round_files = [
        p for p in folder.iterdir() if _ROUND_FILE_RE.match(p.name)
    ]

    def sort_key(p: Path) -> tuple[int, int]:
        m = _ROUND_FILE_RE.match(p.name)
        assert m is not None
        return (int(m.group(1)), int(m.group(2)))

    trace: list[dict] = []
    for rf in sorted(round_files, key=sort_key):
        with open(rf, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("状态") != "成功":
            continue
        tool_calls = data.get("工具调用") or []
        tool_results = data.get("工具结果") or []
        for tc, tr in zip(tool_calls, tool_results):
            result_text = tr.get("结果", "")
            trace.append({
                "round": data.get("轮次"),
                "name": tc.get("名称"),
                "args": tc.get("参数") or {},
                "result_chars": tr.get("字符数", len(str(result_text))),
            })
    return trace


def _nearest_names(log_dir: Path, log_name: str, limit: int = 8) -> list[str]:
    """同目录下与 log_name 公共前缀最长的目录名（报错信息用）。"""
    names = [p.name for p in log_dir.iterdir() if p.is_dir()]

    def common_prefix_len(name: str) -> int:
        n = 0
        for a, b in zip(name, log_name):
            if a != b:
                break
            n += 1
        return n

    return sorted(names, key=common_prefix_len, reverse=True)[:limit]


def _read_overview(folder: Path) -> dict:
    with open(folder / _OVERVIEW_NAME, encoding="utf-8") as f:
        return json.load(f)


def _warn_if_prompt_changed(
    log_dir: Path, log_name: str, user_prompt: str,
) -> None:
    """精确命中但 prompt 与录制时不一致 → 提示 golden 可能过期（只警告不拦截）。"""
    overview = _read_overview(log_dir / log_name)
    recorded = (overview.get("输入") or {}).get("user_prompt", "")
    if recorded and recorded != user_prompt:
        logger.warning(
            "回放精确命中但 user_prompt 与录制时不一致，输出差异可能来自模板/数据变更：%s",
            _fmt_path(log_dir, log_name),
        )


def _fmt_path(log_dir: Path, log_name: str | None) -> str:
    return f"{log_dir}/{log_name}"
