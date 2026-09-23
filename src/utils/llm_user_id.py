"""DeepSeek user_id 作用域 — 每个视频一个随机 user_id，实现并发总结的 KVCache 隔离。

背景（见 https://api-docs.deepseek.com/zh-cn/quick_start/rate_limit）：
  user_id 是 Chat Completions 请求体的顶层字段，DeepSeek 按它做 KVCache / 内容安全 /
  调度隔离。不传时所有请求归入空 user_id 命名空间，多个视频并发总结会共享同一
  缓存命名空间，产生混淆。本模块为每个视频生成独立 user_id：

  - 生成后写入该视频目录的 metadata.json（断点续跑时复用，缓存连续性不中断）；
  - 用 ContextVar 在当前上下文传播；嵌套线程池 / run_in_executor 经 log_config
    的 submit 补丁自动继承。client 在此基础上派生 per-stage id
    `{base_uid}_{stage}` 做阶段级 KVCache 隔离（只缓存 system_prompt + TASK）；
  - client 在每次请求的 extra_body 里携带（顶层字段）。

格式约束：`[a-zA-Z0-9-_]+`、≤512 字符、不得含隐私信息——"vid_" + uuid4 hex 满足。
"""

import contextvars
import json
import logging
import uuid
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

METADATA_FILENAME = "metadata.json"
USER_ID_KEY = "user_id"
_ID_PREFIX = "vid_"

_VIDEO_USER_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_user_id", default=None
)


def generate_user_id() -> str:
    """生成一个合规且不含隐私信息的随机 user_id。"""
    return f"{_ID_PREFIX}{uuid.uuid4().hex}"


def get_current_user_id() -> str | None:
    """当前上下文所属视频的 user_id；不在 user_id_scope 内时为 None。"""
    return _VIDEO_USER_ID.get()


def read_user_id(save_path: Path) -> str | None:
    """从该视频目录的 metadata.json 读既有 user_id；不存在或未设置返回 None。

    metadata.json 损坏时 JSONDecodeError 会自然抛出，阻断处理以暴露问题。
    """
    meta_path = Path(save_path) / METADATA_FILENAME
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    user_id = meta.get(USER_ID_KEY)
    return user_id if isinstance(user_id, str) and user_id else None


def _write_user_id(save_path: Path, user_id: str) -> None:
    """把 user_id merge 进该视频目录的 metadata.json（保留已有字段）。

    写入失败让异常自然抛出——user_id 落不了盘意味着断点续跑会换新 id、
    缓存连续性断裂，不如直接阻断本次分析暴露问题。
    """
    meta_path = Path(save_path) / METADATA_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path} 内容不是 JSON 对象: {type(meta).__name__}")
    meta[USER_ID_KEY] = user_id
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )


def load_or_create_user_id(save_path: Path) -> str:
    """读该视频既有 user_id；没有则生成新 id 并落盘 metadata.json。"""
    existing = read_user_id(save_path)
    if existing:
        return existing
    user_id = generate_user_id()
    _write_user_id(save_path, user_id)
    logger.info("为 %s 生成新 user_id=%s（已写入 metadata.json）", save_path, user_id)
    return user_id


@contextmanager
def user_id_scope(save_path: Path) -> Iterator[str]:
    """在 with 块内把当前上下文绑定到该视频的 user_id（首次自动生成并落盘）。

    LLMAnalyzer.summarize 入口处包裹本作用域；块内（含嵌套线程池的并行阶段）
    所有 LLM 调用都会携带同一 user_id，退出后自动复位。
    """
    user_id = load_or_create_user_id(save_path)
    token = _VIDEO_USER_ID.set(user_id)
    try:
        yield user_id
    finally:
        _VIDEO_USER_ID.reset(token)
        # 清理本视频全部 done 态钉门（含 {base_uid}_{stage} 派生 id），防长跑泄漏。
        from ..llm_infra.gate import clear_user_id

        clear_user_id(user_id)
