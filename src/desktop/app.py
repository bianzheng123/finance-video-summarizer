"""桌面主窗口：URL 输入 → 后台总结 → 实时日志 → 完成通知与打开产物。

日志实时展示：主线程 QTimer 周期性从 ``QueueLogHandler`` 队列 drain（队列线程安全，
worker 及其派生子线程在 push 端、主线程在 pull 端），故 worker 无需也不应跨线程直接
调用 UI，只通过 ``succeeded`` / ``failed`` 信号回报终态。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .env_loader import app_root, is_configured, load_app_env
from .log_stream import QueueLogHandler
from .settings import SettingsDialog
from .worker import SummarizeWorker

if TYPE_CHECKING:
    from ..summarize_entry import SummarizeResult

_POLL_INTERVAL_MS = 200
_MAX_LOG_LINES = 2000


class MainWindow(QMainWindow):
    """主窗口：输入区（URL / 完整版经济版 / 输出目录）+ 只读日志面板 + 状态栏。"""

    def __init__(self, log_handler: QueueLogHandler) -> None:
        super().__init__()
        self.setWindowTitle("财经视频总结")
        self.resize(920, 660)

        self._log_handler = log_handler
        self._worker: SummarizeWorker | None = None
        self._last_step = ""

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 输入区
        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("视频 URL："))
        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("粘贴 B 站 / YouTube / 抖音 视频链接后回车")
        self._url_edit.returnPressed.connect(self._on_start)
        url_row.addWidget(self._url_edit, 1)
        root.addLayout(url_row)

        opt_row = QHBoxLayout()
        self._economic_check = QCheckBox("经济版（只做宏观分析，更快更省 token）")
        opt_row.addWidget(self._economic_check)
        opt_row.addWidget(QLabel("输出目录："))
        self._output_edit = QLineEdit(str(app_root() / "summary_data"))
        opt_row.addWidget(self._output_edit, 1)
        browse_btn = QPushButton("浏览…")
        browse_btn.clicked.connect(self._on_browse)
        opt_row.addWidget(browse_btn)
        root.addLayout(opt_row)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("开始总结")
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)
        settings_btn = QPushButton("设置")
        settings_btn.clicked.connect(self.open_settings)
        btn_row.addWidget(settings_btn)
        open_dir_btn = QPushButton("打开输出目录")
        open_dir_btn.clicked.connect(self._open_output_dir)
        btn_row.addWidget(open_dir_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # 日志区
        root.addWidget(QLabel("运行日志："))
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(_MAX_LOG_LINES)
        root.addWidget(self._log_view, 1)

        # 状态栏：当前阶段
        self._step_label = QLabel("就绪")
        self.statusBar().addWidget(self._step_label)

        # 日志轮询
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll_logs)
        self._timer.start()

    # ---- 事件处理 ----

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        url = self._url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "提示", "请先粘贴视频 URL")
            return
        if not is_configured():
            self.open_settings()
            if not is_configured():
                return

        output_root = Path(self._output_edit.text().strip() or str(app_root() / "summary_data"))
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        self._log_view.clear()
        self._last_step = ""
        self._step_label.setText("运行中…")
        self._start_btn.setEnabled(False)

        self._worker = SummarizeWorker(
            url,
            economic=self._economic_check.isChecked(),
            output_root=output_root,
            parent=self,
        )
        self._worker.succeeded.connect(self._on_succeeded)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_browse(self) -> None:
        current = self._output_edit.text().strip() or str(app_root() / "summary_data")
        chosen = QFileDialog.getExistingDirectory(self, "选择输出目录", current)
        if chosen:
            self._output_edit.setText(chosen)

    def _open_output_dir(self) -> None:
        path = Path(self._output_edit.text().strip() or str(app_root() / "summary_data"))
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._open_path(path)

    def open_settings(self) -> None:
        SettingsDialog(self).exec()

    def _poll_logs(self) -> None:
        for line in self._log_handler.drain():
            self._log_view.appendPlainText(line.text)
            if line.step and line.step != self._last_step:
                self._last_step = line.step
                self._step_label.setText(f"当前阶段：{line.step}")

    def _on_succeeded(self, result: "SummarizeResult") -> None:
        self._start_btn.setEnabled(True)
        self._poll_logs()
        self._step_label.setText("完成")
        self._show_result_dialog(result)

    def _on_failed(self, message: str) -> None:
        self._start_btn.setEnabled(True)
        self._poll_logs()
        self._step_label.setText("失败")
        QMessageBox.critical(self, "总结失败", message)

    def _on_worker_finished(self) -> None:
        # 用 identity 守卫避免「新任务已启动后，旧线程的 finished 误清空新 worker 引用」
        if self.sender() is self._worker:
            self._worker = None

    # ---- 结果展示 ----

    def _show_result_dialog(self, result: "SummarizeResult") -> None:
        box = QMessageBox(self)
        box.setWindowTitle("总结完成")
        box.setIcon(QMessageBox.Icon.Information)
        lines = [
            "报告已生成：",
            f"完整版 HTML：{result.summary_html}",
            f"折叠式 HTML：{result.summary_structured_html}",
        ]
        if result.summary_pdf:
            lines.append(f"PDF：{result.summary_pdf}")
        else:
            lines.append("PDF：未生成（需安装 weasyprint）")
        box.setText("\n".join(lines))

        open_folder_btn = box.addButton("打开输出文件夹", QMessageBox.ButtonRole.AcceptRole)
        open_full_btn = box.addButton("打开完整版 HTML", QMessageBox.ButtonRole.ActionRole)
        open_struct_btn = box.addButton("打开折叠式 HTML", QMessageBox.ButtonRole.ActionRole)
        open_pdf_btn = box.addButton("打开 PDF", QMessageBox.ButtonRole.ActionRole) if result.summary_pdf else None
        box.addButton("关闭", QMessageBox.ButtonRole.RejectRole)

        box.exec()

        clicked = box.clickedButton()
        if clicked is open_folder_btn:
            self._open_path(result.save_path)
        elif clicked is open_full_btn:
            self._open_path(result.summary_html)
        elif clicked is open_struct_btn:
            self._open_path(result.summary_structured_html)
        elif open_pdf_btn is not None and clicked is open_pdf_btn:
            self._open_path(result.summary_pdf)

    @staticmethod
    def _open_path(path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def main() -> None:
    """GUI 入口：加载 .env → 配置日志桥接 → 启动主窗口。"""
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

    window = MainWindow(log_handler)
    window.show()

    # 首次运行未配置 → 引导进设置页
    if not is_configured():
        window.open_settings()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
