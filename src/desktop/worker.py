"""后台总结线程：在 QThread 中运行 ``summarize_url``，完成后经信号把结果/错误带回主线程。

多 URL 并行时每个 URL 一个 worker，用 ``task_scope`` 包裹 ``summarize_url`` 调用，
让该 URL 在任意线程（含内部 LLM 线程池）产出的日志都打上 ``task_id`` 标签，供主线程
按 URL 路由到对应面板。日志的实时展示不走本线程的信号，而是由主线程的 QTimer 周期性
从 ``QueueLogHandler`` 队列里 drain（队列线程安全，worker 及其派生子线程在 push 端、
主线程在 pull 端），本线程只负责跑总结并回报终态，职责单一、无跨线程 UI 调用。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from src.llm_infra.cancel import PipelineCancelled, cancel_scope, request_cancel

from .log_stream import task_scope


def friendly_error(exc: BaseException) -> str:
    """把异常映射为面向用户的可读失败文案；余额不足 / 密钥无效给明确提示，其余保持原样。

    只做「识别已知失败原因」的文案翻译，不吞异常、不改控制流——调用方仍按失败处理。
    """
    status = getattr(exc, "status_code", None)
    text = str(exc)
    text_lower = text.lower()
    if status == 402 or "insufficient balance" in text_lower or "余额不足" in text:
        return "余额不足，请充值后重试"
    if status == 401 or "invalid api key" in text_lower or "authentication" in text_lower:
        return "API Key 无效或未配置好，请到「设置」页检查后重试"
    return f"{type(exc).__name__}: {exc}"


class SummarizeWorker(QThread):
    """后台运行一次总结任务。

    Signals:
        succeeded(str, object): 首参数为 task_id，次参数为
            :class:`~src.summarize_entry.SummarizeResult`。
        failed(str, str): 首参数为 task_id，次参数为失败原因（已格式化为可读文本）。
        cancelled(str): 首参数为 task_id，任务被用户主动取消。
        info_ready(str, float): 首参数为 task_id，次参数为视频时长（秒），
            供 ETA 估算；获取失败则不发。
    """

    succeeded = Signal(str, object)
    failed = Signal(str, str)
    cancelled = Signal(str)
    info_ready = Signal(str, float)

    def __init__(
        self,
        url: str,
        *,
        task_id: str,
        economic: bool,
        output_root: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._url = url
        self._task_id = task_id
        self._economic = economic
        self._output_root = output_root

    def cancel(self) -> None:
        """请求取消本任务：置取消信号，在下次 LLM 调用检查点生效。"""
        request_cancel(self._task_id)

    def run(self) -> None:
        # 延迟 import：确保 desktop_app.py 已在 import src.* 前调用 load_app_env()，
        # 让 summarize_entry 及其依赖在 env 就位后再加载。
        from src.client.video_info_adapter import video_info_adapter
        from src.summarize_entry import summarize_url

        with task_scope(self._task_id), cancel_scope(self._task_id):
            # 预取视频时长供 GUI 估算 ETA（轻量 API；summarize_url 内部会再取一次，
            # 重复开销可忽略）。获取失败仅跳过 ETA 的时长系数，不影响主流程。
            try:
                info = video_info_adapter.get_video_info(self._url)
            except Exception:  # noqa: BLE001 — 时长只是 ETA 增强，任何失败都不阻断任务
                info = None
            if info and info.get("duration"):
                self.info_ready.emit(self._task_id, float(info["duration"]))

            try:
                result = summarize_url(
                    self._url,
                    economic=self._economic,
                    output_root=self._output_root,
                )
            except PipelineCancelled:
                self.cancelled.emit(self._task_id)
                return
            except Exception as exc:  # noqa: BLE001 — 兜底把未预期异常带回主线程展示
                self.failed.emit(self._task_id, friendly_error(exc))
                return

        if result is None:
            self.failed.emit(
                self._task_id, "总结失败：无法获取视频信息或字幕识别失败（详见日志）",
            )
        else:
            self.succeeded.emit(self._task_id, result)
