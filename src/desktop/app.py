"""桌面主窗口：左侧边栏（总结 / 日志 / 设置）+ 三页切换。

外壳只负责三件事：导航、任务调度、日志双路由。

- **导航**：``QListWidget`` 侧边栏 + ``QStackedWidget`` 三页（总结页 / 日志页 / 设置页）。
- **任务调度**：总结页通过 ``start_requested`` 信号把 ``(task_id, url, economic)`` 列表交给
  外壳，外壳按并发上限（``read_concurrency()``）用 ``SummarizeWorker``（QThread）调度。
- **日志双路由**：主线程 ``QTimer`` 周期性从 ``QueueLogHandler`` 队列 drain，按 ``task_id``
  把每条 ``LogLine`` 同时路由到日志页（追加文本）与总结页（按 ``step`` 推进度）；无
  ``task_id`` 的日志只进日志页全局面板。
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..summary_cost import compute_cost
from .env_loader import app_root, is_configured, load_app_env
from .log_page import LogPage
from .log_stream import QueueLogHandler
from .settings import read_concurrency
from .settings_page import SettingsPage
from .summary_page import SummaryPage
from .theme import apply_theme
from .worker import SummarizeWorker

if TYPE_CHECKING:
    from ..summarize_entry import SummarizeResult

_POLL_INTERVAL_MS = 200
# 全局基础字号固定 12pt（不开放给用户调整，避免设置项膨胀）。
_FONT_SIZE = 12

# 侧边栏三页索引（与 QListWidget 顺序一致）。
_PAGE_SUMMARY = 0
_PAGE_LOG = 1
_PAGE_SETTINGS = 2


def _apply_app_font(app: QApplication) -> None:
    """全局放大基础字体，解决默认字体过小看不清的问题。"""
    font = app.font()
    font.setPointSize(_FONT_SIZE)
    app.setFont(font)


def _mono_font() -> QFont:
    """日志区等宽字体（比基础字号小 1pt，保证对齐与信息密度）。"""
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(max(9, _FONT_SIZE - 1))
    return font


class MainWindow(QMainWindow):
    """主窗口外壳：侧边栏 + 三页，负责任务调度与日志双路由。"""

    def __init__(self, log_handler: QueueLogHandler) -> None:
        super().__init__()
        self.setWindowTitle("财经视频总结")
        self.resize(1080, 780)

        self._log_handler = log_handler
        self._pending: deque[tuple[str, str, bool]] = deque()
        self._running = 0
        self._concurrency = read_concurrency()
        self._task_status: dict[str, str] = {}
        self._task_economic: dict[str, bool] = {}
        self._task_results: dict[str, "SummarizeResult"] = {}
        self._task_costs: dict[str, float] = {}
        self._workers: dict[str, SummarizeWorker] = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- 侧边栏（三个栏目均分高度填满） ----
        self._sidebar = QWidget()
        self._sidebar.setObjectName("Sidebar")
        self._sidebar.setFixedWidth(160)
        side_layout = QVBoxLayout(self._sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        for index, name in enumerate(["总结", "日志", "设置"]):
            btn = QPushButton(name)
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            side_layout.addWidget(btn, 1)
            self._nav_group.addButton(btn, index)
        self._nav_group.idClicked.connect(self._on_nav)
        root.addWidget(self._sidebar)

        # ---- 三页 ----
        self._stack = QStackedWidget()
        self._summary_page = SummaryPage()
        self._log_page = LogPage(_mono_font())
        self._settings_page = SettingsPage()
        self._stack.addWidget(self._summary_page)
        self._stack.addWidget(self._log_page)
        self._stack.addWidget(self._settings_page)
        root.addWidget(self._stack, 1)

        self._summary_page.start_requested.connect(self._on_start)
        self._summary_page.stop_requested.connect(self._on_stop)
        self._summary_page.settings_requested.connect(self.navigate_to_settings)

        self._nav_group.button(_PAGE_SUMMARY).setChecked(True)

        # ---- 日志轮询 ----
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll_logs)
        self._timer.start()

    # ---- 导航 ----
    def _on_nav(self, row: int) -> None:
        self._stack.setCurrentIndex(row)

    def navigate_to_settings(self) -> None:
        self._nav_group.button(_PAGE_SETTINGS).setChecked(True)
        self._on_nav(_PAGE_SETTINGS)

    # ---- 批次调度 ----
    def _on_start(self, tasks: list[tuple[str, str, bool]]) -> None:
        if self._running:
            return
        self._concurrency = read_concurrency()
        self._pending = deque(tasks)
        self._running = 0
        self._task_status = {task_id: "等待" for task_id, _url, _eco in tasks}
        self._task_economic = {task_id: eco for task_id, _url, eco in tasks}
        self._task_results = {}
        self._task_costs = {}
        self._workers = {}
        self._log_page.reset_tasks(tasks)
        self._update_summary_bar()
        self._dispatch()

    def _dispatch(self) -> None:
        while self._running < self._concurrency and self._pending:
            self._start_worker(self._pending.popleft())

    def _start_worker(self, task: tuple[str, str, bool]) -> None:
        task_id, url, economic = task
        self._task_status[task_id] = "运行中"
        self._summary_page.set_task_status(task_id, "运行中")
        worker = SummarizeWorker(
            url,
            task_id=task_id,
            economic=economic,
            output_root=self._summary_page.output_root(),
            parent=self,
        )
        worker.succeeded.connect(self._on_succeeded)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.info_ready.connect(self._summary_page.set_task_duration)
        worker.finished.connect(lambda tid=task_id: self._on_finished(tid))
        self._workers[task_id] = worker
        self._running += 1
        self._update_summary_bar()
        worker.start()

    def _on_succeeded(self, task_id: str, result: "SummarizeResult") -> None:
        self._task_status[task_id] = "成功"
        self._task_results[task_id] = result
        self._summary_page.set_task_status(task_id, "成功")
        self._summary_page.set_task_result(task_id, result)
        # 真实 token 用量费用（落盘日志口径）；tokenizer=False 跳过加载、按官方换算降级，
        # 避免 GUI/打包环境触发 tokenizer 下载。仅 LLM 费用，ASR/embedding 需看腾讯云账单。
        cost = compute_cost(
            result.save_path,
            tokenizer=False,
            economic=self._task_economic.get(task_id, False),
        )
        if cost is not None:
            cost_yuan = round(cost["cost_yuan"], 2)
            self._task_costs[task_id] = cost_yuan
            self._summary_page.set_task_cost(task_id, cost_yuan)
        self._update_summary_bar()

    def _on_failed(self, task_id: str, message: str) -> None:
        self._task_status[task_id] = "失败"
        self._summary_page.set_task_status(task_id, "失败")
        self._log_page.append_task_log(task_id, f"[失败] {message}")
        self._update_summary_bar()

    def _on_stop(self, task_id: str) -> None:
        """用户点击「停止」：请求取消对应 worker（在下次 LLM 调用检查点生效）。"""
        worker = self._workers.get(task_id)
        if worker is not None and self._task_status.get(task_id) == "运行中":
            self._log_page.append_task_log(task_id, "[停止] 已请求取消，将在当前步骤结束后生效…")
            worker.cancel()
        else:
            self._log_page.append_task_log(task_id, "[停止] 该任务不在运行中，无法取消")

    def _on_cancelled(self, task_id: str) -> None:
        self._task_status[task_id] = "已取消"
        self._summary_page.set_task_status(task_id, "已取消")
        self._log_page.append_task_log(task_id, "[已取消] 任务已被用户取消")
        self._update_summary_bar()

    def _on_finished(self, task_id: str) -> None:
        self._running -= 1
        self._dispatch()
        self._update_summary_bar()
        if self._running == 0 and not self._pending:
            ok = sum(1 for s in self._task_status.values() if s == "成功")
            fail = sum(1 for s in self._task_status.values() if s == "失败")
            cancelled = sum(1 for s in self._task_status.values() if s == "已取消")
            total_cost = sum(self._task_costs.values()) if self._task_costs else None
            self._summary_page.on_batch_finished(ok, fail, cancelled, total_cost)

    def _update_summary_bar(self) -> None:
        total = len(self._task_status)
        ok = sum(1 for s in self._task_status.values() if s == "成功")
        fail = sum(1 for s in self._task_status.values() if s == "失败")
        cancelled = sum(1 for s in self._task_status.values() if s == "已取消")
        parts = [f"运行中 {self._running}", f"成功 {ok}", f"失败 {fail}"]
        if cancelled:
            parts.append(f"已取消 {cancelled}")
        parts.append(f"共 {total}")
        self._summary_page.set_summary_label(" · ".join(parts))

    # ---- 日志双路由 ----
    def _poll_logs(self) -> None:
        for line in self._log_handler.drain():
            if line.task_id:
                self._log_page.append_task_log(line.task_id, line.text)
                if line.step:
                    self._summary_page.advance_task(line.task_id, line.step)
            else:
                self._log_page.append_global_log(line.text)


def main() -> None:
    """GUI 入口：加载 .env → 配置日志桥接 → 放大字体 → 启动主窗口。"""
    load_app_env()

    from ..log_config import install_logging_hooks

    install_logging_hooks()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    log_handler = QueueLogHandler()
    root.addHandler(log_handler)

    # 应用级日志持久化到用户可写目录（frozen 下避免写 _MEIPASS）
    log_dir = app_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "desktop_app.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root.addHandler(file_handler)

    app = QApplication(sys.argv)
    app.setApplicationName("财经视频总结")
    _apply_app_font(app)
    apply_theme(app)

    window = MainWindow(log_handler)
    window.show()

    # 首次运行未配置 → 切到设置页引导填写
    if not is_configured():
        window.navigate_to_settings()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
