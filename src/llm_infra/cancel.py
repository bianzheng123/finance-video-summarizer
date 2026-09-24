"""用户主动取消任务的控制流机制（供桌面 GUI「停止」按钮使用）。

核心思路：任务取消是一个**跨线程的共享信号**，用一个 ``threading.Event`` 承载；
当前任务身份用 ``contextvars.ContextVar`` 传播——依赖 ``log_config._install_context_propagation``
已给 ``ThreadPoolExecutor.submit`` 打的补丁（每次 submit 都 ``copy_context()``），
LLM 线程池里的 worker 线程会继承提交点的 ContextVar 快照，从而能在任意深度读到
「我属于哪个任务」。

取消粒度是一个 LLM 调用：``client.call`` / ``client.call_conversation`` 在每次
``bound.invoke`` 前调 :func:`check_cancelled`，命中即抛 :class:`PipelineCancelled`。
已发出的 HTTP 请求无法中断，等当前调用结束后、下一次检查点生效——对「提前终止」
足够及时，也不会把已发出的请求半途掐断留下坏状态。
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager

# 当前线程正在处理的任务 id（随线程池 submit 的 copy_context 传播到 LLM worker 线程）。
_current_task_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "cancel_task_id", default="",
)

# task_id → 取消信号。Event 是跨线程共享的可变对象：GUI 线程 set、LLM worker 线程
# is_set，二者通过同一对象引用互通，无需额外同步消息。
_cancel_events: dict[str, threading.Event] = {}
_cancel_lock = threading.Lock()


class PipelineCancelled(Exception):
    """用户主动取消任务时抛出的控制流异常（区别于真正的失败）。"""


@contextmanager
def cancel_scope(task_id: str) -> Iterator[None]:
    """声明「当前线程及其派生线程正在处理 task_id」，退出时清理取消信号。

    与 ``log_stream.task_scope`` 同构：进入时把 task_id 写入 ContextVar 并注册一个
    Event，yield 后 reset ContextVar、移除 Event。放在 worker 线程调用
    ``summarize_url`` 之前，使该任务的全部 LLM 调用都能被 :func:`check_cancelled`
    感知到取消请求。
    """
    event = threading.Event()
    with _cancel_lock:
        _cancel_events[task_id] = event
    token = _current_task_id.set(task_id)
    try:
        yield
    finally:
        _current_task_id.reset(token)
        with _cancel_lock:
            _cancel_events.pop(task_id, None)


def request_cancel(task_id: str) -> None:
    """请求取消 task_id：置其 Event。若任务未注册（已结束/未开始）则安全忽略。"""
    with _cancel_lock:
        event = _cancel_events.get(task_id)
    if event is not None:
        event.set()


def check_cancelled() -> None:
    """若当前任务已被请求取消则抛 :class:`PipelineCancelled`，否则无操作。

    放在每次 LLM 调用前；非取消路径零副作用、开销可忽略（一次 ContextVar 读 +
    一次加锁查字典 + 一次 Event 判读）。
    """
    task_id = _current_task_id.get()
    if not task_id:
        return
    with _cancel_lock:
        event = _cancel_events.get(task_id)
    if event is not None and event.is_set():
        raise PipelineCancelled("任务已被用户取消")
