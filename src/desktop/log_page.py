"""日志页：全局日志 + 每个 URL 一个可折叠日志面板。

顶部「全局日志」收无 ``task_id`` 的应用级消息；下方按任务创建可折叠
``QGroupBox``（勾选折叠），内部是该 URL 的只读滚动日志。日志行由
``MainWindow`` 按 ``task_id`` 路由后调用 :meth:`append_task_log` 追加。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

_MAX_LOG_LINES = 2000


class LogPage(QWidget):
    """日志页：全局日志 + 每 URL 可折叠面板。"""

    def __init__(self, log_font: QFont, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log_font = log_font
        self._task_views: dict[str, QPlainTextEdit] = {}
        self._task_groups: dict[str, QGroupBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # ---- 全局日志（纯标题，不可折叠） ----
        self._global_group = QGroupBox("全局日志")
        global_layout = QVBoxLayout(self._global_group)
        self._global_view = self._make_view()
        self._global_view.setFixedHeight(120)
        global_layout.addWidget(self._global_view)
        layout.addWidget(self._global_group)

        # ---- 任务日志（每 URL 一个可折叠面板） ----
        task_box = QGroupBox("任务日志")
        task_box_layout = QVBoxLayout(task_box)
        self._task_container = QWidget()
        self._task_layout = QVBoxLayout(self._task_container)
        self._task_layout.setContentsMargins(0, 0, 0, 0)
        self._task_layout.setSpacing(8)
        # 顶部对齐：面板少时贴顶，多余空间留底部（与总结页行为一致）
        self._task_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._empty_label = QLabel("暂无任务日志，开始总结后这里会按视频显示日志面板。")
        self._empty_label.setStyleSheet("color: #9CA3AF;")
        self._task_layout.addWidget(self._empty_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._task_container)
        task_box_layout.addWidget(scroll)
        layout.addWidget(task_box, 1)

    def _make_view(self) -> QPlainTextEdit:
        view = QPlainTextEdit()
        view.setReadOnly(True)
        # 显式声明只允许选中复制、禁止编辑，作为 readOnly 的兜底
        view.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard,
        )
        view.setMaximumBlockCount(_MAX_LOG_LINES)
        view.setFont(self._log_font)
        return view

    # ---- 供 MainWindow 调用 ----
    def reset_tasks(self, tasks: list[tuple[str, str, bool]]) -> None:
        """重建任务面板：清空旧面板，按当前批次任务（task_id, url, economic）新建。"""
        for task_id in list(self._task_groups):
            self._remove_task_group(task_id)
        self._empty_label.setVisible(not tasks)
        for task_id, url, _economic in tasks:
            self._add_task_group(task_id, url)

    def append_global_log(self, text: str) -> None:
        self._global_view.appendPlainText(text)

    def append_task_log(self, task_id: str, text: str) -> None:
        view = self._task_views.get(task_id)
        if view is not None:
            view.appendPlainText(text)
        else:
            self._global_view.appendPlainText(text)

    # ---- 内部 ----
    def _add_task_group(self, task_id: str, url: str) -> None:
        group = QGroupBox(f"任务 #{task_id}")
        group.setCheckable(True)
        group.setChecked(True)
        group.setToolTip(url)
        group_layout = QVBoxLayout(group)
        view = self._make_view()
        group_layout.addWidget(view)
        self._task_layout.addWidget(group)
        self._task_groups[task_id] = group
        self._task_views[task_id] = view

    def _remove_task_group(self, task_id: str) -> None:
        group = self._task_groups.pop(task_id, None)
        if group is not None:
            self._task_layout.removeWidget(group)
            group.deleteLater()
        self._task_views.pop(task_id, None)
