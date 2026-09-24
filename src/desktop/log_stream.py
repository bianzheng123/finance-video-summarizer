"""日志 → 队列的流式桥接：把管线 logging 记录投递到线程安全队列，供 GUI 消费。

管线（``src/log_config.py``）已给每条 ``LogRecord`` 注入 ``record.step``（当前步骤标签，
如「第1步-话题分割」「第4_2步-关键词合并」）与 ``record.session``（会话名）。本模块只做
「格式化一行日志 → 投递队列」，不改管线。纯标准库实现，不依赖 PySide6，便于独立测试。
"""

from __future__ import annotations

import contextvars
import logging
import queue
from collections.abc import Iterator
from contextlib import contextmanager


# 当前日志所属的任务 ID（多 URL 并行时，每个 URL 一个 task_id）。
# 用 ContextVar 而非线程局部变量：worker 线程设置后，内部线程池经
# ``log_config._install_context_propagation`` 的 submit 补丁会把上下文（含 task_id）
# 带进 LLM worker 线程，从而让该 URL 在任意线程产出的日志都能正确路由回对应面板。
_task_id: contextvars.ContextVar[str] = contextvars.ContextVar("task_id", default="")


@contextmanager
def task_scope(task_id: str) -> Iterator[None]:
    """把当前线程（及其派生子线程）的日志打上 ``task_id`` 标签，退出后自动复位。

    与 ``log_config.step_logger`` 同构：设置 ContextVar，yield 后 reset。供
    ``SummarizeWorker.run`` 在调用 ``summarize_url`` 前包裹，使该 URL 的**全部**日志
    （含进入 ``session_logger`` 之前的「处理URL」「视频信息」阶段）都能按 task_id 路由。
    """
    token = _task_id.set(task_id)
    try:
        yield
    finally:
        _task_id.reset(token)


class LogLine:
    """一条投递给 UI 的日志：已格式化文本 + 当前步骤标签 + 所属任务 ID（可能为空串）。"""

    __slots__ = ("text", "step", "task_id")

    def __init__(self, text: str, step: str, task_id: str) -> None:
        self.text = text
        self.step = step
        self.task_id = task_id


class _PipelineFormatter(logging.Formatter):
    """与 ``src/log_config._PipelineFormatter`` 同款行格式：时间 [会话] [级别] [步骤] 消息。"""

    def __init__(self) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        session = getattr(record, "session", "") or ""
        step = getattr(record, "step", "") or ""
        parts = [self.formatTime(record, self.datefmt)]
        if session:
            parts.append(f"[{session}]")
        parts.append(f"[{record.levelname}]")
        if step:
            parts.append(f"[{step}]")
        parts.append(record.getMessage())
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))
        return " ".join(parts)


class QueueLogHandler(logging.Handler):
    """把每条日志格式化为一行 :class:`LogLine` 投递到有界队列，供 GUI 线程轮询消费。

    有界队列防内存堆积；``emit`` 内绝不抛异常、绝不阻塞管线——队列满时丢弃最旧一条，
    避免消费端卡死时把管线带崩或让日志堆积耗尽内存。
    """

    def __init__(self, maxsize: int = 4096) -> None:
        super().__init__()
        self.queue: "queue.Queue[LogLine]" = queue.Queue(maxsize=maxsize)
        self.setFormatter(_PipelineFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = LogLine(
                text=self.format(record),
                step=getattr(record, "step", "") or "",
                task_id=_task_id.get(),
            )
            try:
                self.queue.put_nowait(line)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.queue.put_nowait(line)
                except queue.Full:
                    pass
        except Exception:
            # 日志桥接绝不影响主流程
            pass

    def drain(self, limit: int = 1000) -> list[LogLine]:
        """非阻塞地批量取走当前队列里的所有日志行（最多 ``limit`` 条）。"""
        lines: list[LogLine] = []
        for _ in range(limit):
            try:
                lines.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return lines
