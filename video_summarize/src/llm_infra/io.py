"""JSON 原子读写与 LLM 对话日志落盘，两块合并在本模块。

1. **阶段整合检查点 IO**（原 checkpoint_io）：x_y_z.json 的原子写、统一包封与读取。

   整合文件结构：`{"输出": {<子键>: <LLM 校验后 result dict>}, "结果": <阶段后处理产物>}`
   ——「输出」是该阶段全部 LLM 调用的 result dict 整合，「结果」仅在断点续跑 /
   下游需要后处理产物时存在。

   全部为原子写（临时文件 + os.replace）。写盘失败不在此吞异常——缓存写不
   成功即视为该阶段未完成，下次续跑重算。

2. **对话日志写入器**（原 conversation_log_writer）：一次逻辑 LLM 调用落一个文件夹。

   布局（`llm_output_N/{log_name}/`）：
   - `概览.json`：调用级聚合（步骤/状态/输入/输出/结果/调用次数/轮数），
     每轮结束后原子重写；中途崩溃残留状态为「进行中」。
   - `round{r}__attempt{n}.json`：每轮每次尝试一个文件，写完即定稿不再重写；
     每个文件记录该次尝试的完整输入（实际发送的消息序列）、思考、正文、
     工具调用/结果、用量与状态。

   同样全部为原子写（临时文件 + os.replace）。写盘失败不在此吞异常——由调用方
   （client.py）降级为 warning，避免日志盘故障触发 tenacity 重新调用 LLM。
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 一、阶段整合检查点 IO
# ═══════════════════════════════════════════════════════════════════════════

def write_json_atomic(path: Path, data: Any) -> None:
    """原子写 JSON（tmp + os.replace），失败让异常自然抛出。"""
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_integrated(
    path: Path,
    llm_outputs: dict[str, Any],
    result: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """写整合文件 `{"输出": ..., "结果": ...}`；result 为 None 时不写「结果」。

    `extra` 的键原样并入顶层，供需要在「结果」之外再带一份结构的检查点使用
    （第 2 步的 `2_corrected_subtitle.json` 用 `结论集` 带 verdicts/指代/初判剪枝，
    「结果」则是字幕行列表）。
    """
    payload: dict[str, Any] = {"输出": llm_outputs}
    if result is not None:
        payload["结果"] = result
    if extra:
        payload.update(extra)
    write_json_atomic(path, payload)


def read_json(path: Path) -> Any:
    """读 JSON 文件，不存在/损坏返回 None（缓存是外部状态，损坏视为不存在）。

    文件不存在是首轮正常态（断点尚未生成），打 info 提示「将自行生成」而非 warning；
    仅文件存在但解析失败（真正损坏）才打 warning。
    """
    if not path.exists():
        logger.info("找不到 %s，将自行生成", path.name)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 —— 损坏缓存自愈，由调用方重算
        logger.warning("读取 %s 失败：%s", path.name, e)
        return None


def load_dump_outputs(path: Path) -> dict[str, Any]:
    """加载 dump 文件的「输出」节作基底（增量续跑时保留被跳过阶段的旧输出）。

    不存在/损坏/结构不符返回 `{}`。文件不存在是首轮正常态，不打日志。
    """
    if not path.exists():
        return {}
    data = read_json(path)
    if isinstance(data, dict):
        outputs = data.get("输出")
        if isinstance(outputs, dict):
            return outputs
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# 二、对话日志落盘
# ═══════════════════════════════════════════════════════════════════════════

def write_attempt_log(
    log_dir: Path,
    log_name: str,
    round_i: int,
    attempt_n: int,
    payload: dict[str, Any],
) -> None:
    """原子写 {log_dir}/{log_name}/round{round_i}__attempt{attempt_n}.json。"""
    call_dir = log_dir / log_name
    call_dir.mkdir(parents=True, exist_ok=True)
    target = call_dir / f"round{round_i}__attempt{attempt_n}.json"
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)


def write_overview_log(
    log_dir: Path,
    log_name: str,
    payload: dict[str, Any],
) -> None:
    """原子写 {log_dir}/{log_name}/概览.json（调用级聚合，增量重写）。"""
    call_dir = log_dir / log_name
    call_dir.mkdir(parents=True, exist_ok=True)
    target = call_dir / "概览.json"
    tmp = target.with_name("概览.json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
