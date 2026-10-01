"""System prompt 作用域 — 各版本绑定自己的 system prompt。

背景：system prompt 已从基础设施层（``llm_infra/prompts/``）下放到各版本的总结目录
（完整版 ``llm_summarize/shared_system.txt``、经济版 ``llm_summarize_economic/shared_system.txt``），
因此 ``LLMClient`` 不再硬编码加载全局 system prompt，而是从本模块的 ContextVar 读当前
版本绑定的 prompt。

- 完整版入口 ``LLMAnalyzer.summarize``、经济版入口 ``EconomicAnalyzer.summarize`` 各自在
  最外层用 ``system_prompt_scope`` 绑定自己的 system prompt；
- 用 ContextVar 传播；嵌套线程池 / run_in_executor 经 ``log_config`` 的 submit 补丁自动继承
  （与 ``llm_user_id`` 的 ``user_id_scope`` 同模式）；
- ``LLMClient.call`` 的 ``system_prompt`` 参数为 None 时读 ``get_current_system_prompt()``；
  显式传空串仍表示「不带 system message」（如主题去重传空串）。
"""

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SYSTEM_PROMPT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_system_prompt", default=None,
)


def get_current_system_prompt() -> str | None:
    """当前上下文绑定的 system prompt；不在 system_prompt_scope 内时为 None。"""
    return _SYSTEM_PROMPT.get()


def load_system_prompt_file(path: Path) -> str:
    """从文件加载 system prompt 文本（strip 首尾空白）。"""
    return path.read_text(encoding="utf-8").strip()


@contextmanager
def system_prompt_scope(prompt: str) -> Iterator[None]:
    """在 with 块内把当前上下文绑定到指定 system prompt，退出后复位。"""
    token = _SYSTEM_PROMPT.set(prompt)
    try:
        yield
    finally:
        _SYSTEM_PROMPT.reset(token)
