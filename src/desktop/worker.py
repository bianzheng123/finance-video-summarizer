"""后台总结线程：在 QThread 中运行 ``summarize_url``，完成后经信号把结果/错误带回主线程。

日志的实时展示不走本线程的信号，而是由主线程的 QTimer 周期性从 ``QueueLogHandler``
队列里 drain（队列线程安全，worker 及其派生子线程在 push 端、主线程在 pull 端），
本线程只负责跑总结并回报终态，职责单一、无跨线程 UI 调用。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal


class SummarizeWorker(QThread):
    """后台运行一次总结任务。

    Signals:
        succeeded(object): 传回 :class:`~src.summarize_entry.SummarizeResult`。
        failed(str): 传回失败原因（已格式化为可读文本）。
    """

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        url: str,
        *,
        economic: bool,
        output_root: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._url = url
        self._economic = economic
        self._output_root = output_root

    def run(self) -> None:
        # 延迟 import：确保 desktop_app.py 已在 import src.* 前调用 load_app_env()，
        # 让 summarize_entry 及其依赖在 env 就位后再加载。
        from src.summarize_entry import summarize_url

        try:
            result = summarize_url(
                self._url,
                economic=self._economic,
                output_root=self._output_root,
            )
        except Exception as exc:  # noqa: BLE001 — 兜底把未预期异常带回主线程展示
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return

        if result is None:
            self.failed.emit("总结失败：无法获取视频信息或字幕识别失败（详见日志）")
        else:
            self.succeeded.emit(result)
