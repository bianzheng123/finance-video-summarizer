"""总结页：视频 URL 输入 + 逐条进度展示。

- 每个 URL 一个 :class:`UrlCard`（URL 输入框 + 「经济版」开关 + 删除按钮）；
  底部「添加 URL」按钮动态增删，运行期禁用编辑。
- 点击「开始总结」后，每个非空 URL 卡片进入运行态：输入框锁定、显示该 URL 专属的
  :class:`TaskProgressBar`（百分比进度条 + 当前步骤 + 预计剩余时间）+ 状态徽标 +
  「打开产物」按钮；任务成功后卡片显示本次 LLM 费用。
- 本页不跑任务，只收集任务并通过 ``start_requested`` 信号交给 ``MainWindow`` 调度；
  进度/状态/产物/费用由 ``MainWindow`` 按 ``task_id`` 回写本页。

对外契约（供 ``MainWindow`` 调用）：
- ``start_requested(object)``：携带 ``list[tuple[task_id, url, economic]]``。
- ``output_root()`` / ``set_task_status()`` / ``advance_task()`` / ``set_task_result()`` /
  ``set_task_duration()`` / ``set_task_cost()`` / ``set_summary_label()`` /
  ``on_batch_finished()`` / ``open_output_dir()``。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .env_loader import app_root, missing_required_keys
from .progress import ECONOMIC_STEPS, FULL_STEPS, TaskProgressBar, step_key_of

if TYPE_CHECKING:
    from ..summarize_entry import SummarizeResult

# 状态 → 徽标颜色（与 progress.py 的三态配色一致）。
_STATUS_COLORS: dict[str, str] = {
    "等待": "#D97706",
    "运行中": "#2563EB",
    "成功": "#16A34A",
    "失败": "#DC2626",
    "已取消": "#6B7280",
}

# URL 卡片固定高度：宽度填满滚动区（与「输出目录」栏右对齐），高度随状态切换
# （运行态多出进度条 + 当前步骤 + 预计剩余时间）。
_CARD_HEIGHT_IDLE = 120
_CARD_HEIGHT_RUNNING = 200


class UrlCard(QFrame):
    """一个 URL 的卡片：输入态（填写/删除）与运行态（进度条/状态/打开产物）。"""

    def __init__(
        self,
        index: int,
        on_remove: object,
        on_stop: object,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("UrlCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedHeight(_CARD_HEIGHT_IDLE)
        self.index = index
        self.task_id = ""
        self.status = "等待"
        self.result: "SummarizeResult | None" = None
        self.progress: TaskProgressBar | None = None
        self._on_remove = on_remove
        self._on_stop = on_stop
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._index_label = QLabel(f"#{self.index}")
        self._index_label.setStyleSheet("font-weight: 600; color: #374151;")
        header.addWidget(self._index_label)
        self._status_label = QLabel()
        self._apply_status()
        header.addWidget(self._status_label)
        header.addStretch(1)
        self._remove_btn = QPushButton("删除")
        self._remove_btn.setFixedWidth(56)
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        header.addWidget(self._remove_btn)
        root.addLayout(header)

        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("粘贴 B 站 / YouTube / 抖音 视频链接")
        root.addWidget(self._url_edit)

        opt_row = QHBoxLayout()
        opt_row.setSpacing(8)
        self._economic_check = QCheckBox("经济版（只做宏观分析，更快更省 token）")
        opt_row.addWidget(self._economic_check)
        opt_row.addStretch(1)
        self._cost_label = QLabel()
        self._cost_label.setStyleSheet("color: #6B7280; font-weight: 600;")
        self._cost_label.hide()
        opt_row.addWidget(self._cost_label)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setStyleSheet("color: #DC2626; font-weight: 600;")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._stop_btn.hide()
        opt_row.addWidget(self._stop_btn)
        self._open_btn = QPushButton("打开产物")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_result)
        opt_row.addWidget(self._open_btn)
        root.addLayout(opt_row)

        # 进度条占位（运行态才显示）
        self._progress_holder = QWidget()
        self._progress_layout = QVBoxLayout(self._progress_holder)
        self._progress_layout.setContentsMargins(0, 0, 0, 0)
        self._progress_holder.hide()
        root.addWidget(self._progress_holder)

    # ---- 对外访问 ----
    def url(self) -> str:
        return self._url_edit.text().strip()

    def economic(self) -> bool:
        return self._economic_check.isChecked()

    def set_index(self, index: int) -> None:
        self.index = index
        self._index_label.setText(f"#{index}")

    # ---- 状态与进度 ----
    def _apply_status(self) -> None:
        color = _STATUS_COLORS.get(self.status, "#5B5F66")
        self._status_label.setText(f"● {self.status}")
        self._status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def set_status(self, status: str) -> None:
        self.status = status
        self._apply_status()
        if self.progress is None:
            return
        if status == "成功":
            self.progress.complete()
        elif status == "失败":
            self.progress.fail()
        elif status == "已取消":
            self.progress.cancel()

    def advance(self, step_tag: str) -> None:
        if self.progress is None:
            return
        key = step_key_of(step_tag)
        if key:
            self.progress.mark_reached(key)

    def set_result(self, result: "SummarizeResult") -> None:
        self.result = result
        self._open_btn.setEnabled(True)

    def set_video_duration(self, seconds: float) -> None:
        """喂入视频时长（秒），启用 ETA 的时长系数估算。"""
        if self.progress is not None:
            self.progress.set_video_duration(seconds)

    def set_cost(self, cost_yuan: float) -> None:
        """在卡片上显示本次总结的预估 LLM 费用（元）。"""
        self._cost_label.setText(f"预估费用 约 ¥{cost_yuan:.2f}")
        self._cost_label.show()

    # ---- 生命周期 ----
    def arm(self, task_id: str, economic: bool) -> None:
        """进入运行态：锁定输入、按经济版/完整版构建进度条并重置状态。"""
        self.task_id = task_id
        self.result = None
        self._url_edit.setEnabled(False)
        self._economic_check.setEnabled(False)
        self._remove_btn.setEnabled(False)
        self._open_btn.setEnabled(False)
        self._stop_btn.show()
        self._clear_progress()
        steps = ECONOMIC_STEPS if economic else FULL_STEPS
        self.progress = TaskProgressBar(steps, economic=economic)
        self._progress_layout.addWidget(self.progress)
        self._progress_holder.show()
        self.setFixedHeight(_CARD_HEIGHT_RUNNING)
        self.set_status("等待")

    def re_enable(self) -> None:
        """批次结束后重新开放编辑（保留最终状态与进度条作参考）。"""
        self._url_edit.setEnabled(True)
        self._economic_check.setEnabled(True)
        self._remove_btn.setEnabled(True)
        self._stop_btn.hide()

    def clear(self) -> None:
        """清空该卡片内容，回到空白输入态。"""
        self._url_edit.clear()
        self._economic_check.setChecked(False)
        self.result = None
        self._open_btn.setEnabled(False)
        self._cost_label.hide()
        self._stop_btn.hide()
        self._clear_progress()
        self._progress_holder.hide()
        self.setFixedHeight(_CARD_HEIGHT_IDLE)
        self.set_status("等待")

    def _clear_progress(self) -> None:
        while self._progress_layout.count():
            item = self._progress_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.progress = None

    # ---- 回调 ----
    def _on_remove_clicked(self) -> None:
        self._on_remove(self)

    def _on_stop_clicked(self) -> None:
        self._on_stop(self)

    def _open_result(self) -> None:
        if self.result is not None:
            # 打开存放渲染文件的目录（rendering/，经济版 rendering_economic/），
            # 而非视频根目录——summary_html.parent 正是渲染目录。
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.summary_html.parent)))


class SummaryPage(QWidget):
    """总结页：URL 卡片列表 + 输出目录 + 开始按钮。"""

    start_requested = Signal(object)  # list[tuple[str, str, bool]]：(task_id, url, economic)
    stop_requested = Signal(str)  # task_id：用户点「停止」请求取消该任务
    settings_requested = Signal()  # 缺配置时请求切到设置页引导填写

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards: list[UrlCard] = []
        self._by_task_id: dict[str, UrlCard] = {}
        self._running = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        hint = QLabel("视频 URL：每个视频一个输入框，可分别勾选「经济版」；点击开始后逐条显示进度。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #5B5F66;")
        layout.addWidget(hint)

        # ---- 产物说明（新手向：产物是什么、在哪、怎么用） ----
        about_box = QGroupBox("产物说明")
        about_layout = QVBoxLayout(about_box)
        about = QLabel(
            "每个视频总结完成后，产物保存在 输出目录/<平台>/<作者>/<标题>/ 下：\n"
            "• rendering/summary.md —— 金融分析报告（Markdown，经济版在 rendering_economic/ 下）\n"
            "• rendering/summary.html —— 网页版报告，双击即可用浏览器打开\n"
            "• rendering/summary.pdf —— PDF 版（默认生成，无需额外安装）\n"
            "任务完成后点卡片上的「打开产物」直达渲染目录（rendering/，经济版 rendering_economic/）；"
            "本次费用为预估（约，仅 DeepSeek LLM token，不含腾讯云 ASR / embedding），会显示在卡片上。"
        )
        about.setWordWrap(True)
        about.setStyleSheet("color: #5B5F66; font-weight: 400;")
        about_layout.addWidget(about)
        layout.addWidget(about_box)

        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(8)
        # 顶部对齐：默认卡贴滚动区顶部，多余空间留底部（否则卡片少时垂直居中）
        self._cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._cards_container)
        layout.addWidget(scroll, 1)

        add_row = QHBoxLayout()
        self._add_btn = QPushButton("＋ 添加 URL")
        self._add_btn.setObjectName("PrimaryButton")
        self._add_btn.clicked.connect(self._add_card)
        add_row.addWidget(self._add_btn)
        add_row.addStretch(1)
        layout.addLayout(add_row)

        opt_row = QHBoxLayout()
        opt_row.setSpacing(8)
        opt_row.addWidget(QLabel("输出目录："))
        self._output_edit = QLineEdit(str(app_root() / "summary_data"))
        opt_row.addWidget(self._output_edit, 1)
        browse_btn = QPushButton("浏览…")
        browse_btn.clicked.connect(self._on_browse)
        opt_row.addWidget(browse_btn)
        layout.addLayout(opt_row)

        start_row = QHBoxLayout()
        start_row.setSpacing(8)
        self._start_btn = QPushButton("开始总结")
        self._start_btn.setObjectName("PrimaryButton")
        self._start_btn.setMinimumHeight(36)
        self._start_btn.clicked.connect(self._on_start)
        start_row.addWidget(self._start_btn)
        open_dir_btn = QPushButton("打开输出目录")
        open_dir_btn.setMinimumHeight(36)
        open_dir_btn.clicked.connect(self.open_output_dir)
        start_row.addWidget(open_dir_btn)
        self._summary_label = QLabel("就绪")
        self._summary_label.setStyleSheet("color: #5B5F66;")
        start_row.addWidget(self._summary_label)
        start_row.addStretch(1)
        layout.addLayout(start_row)

        self._add_card()

    # ---- 卡片管理 ----
    def _add_card(self) -> None:
        index = len(self._cards) + 1
        card = UrlCard(index, self._remove_card, self._request_stop, self)
        self._cards.append(card)
        self._cards_layout.addWidget(card)

    def _request_stop(self, card: UrlCard) -> None:
        if card.task_id:
            self.stop_requested.emit(card.task_id)

    def _remove_card(self, card: UrlCard) -> None:
        if self._running:
            return
        if len(self._cards) == 1:
            card.clear()
            return
        self._cards.remove(card)
        self._cards_layout.removeWidget(card)
        card.deleteLater()
        self._renumber()

    def _renumber(self) -> None:
        for i, card in enumerate(self._cards, 1):
            card.set_index(i)

    # ---- 开始 ----
    def _on_start(self) -> None:
        if self._running:
            return
        # 开始前校验必填配置：缺失则直接提示失败，不启动任务，并引导去设置页填写。
        missing = missing_required_keys()
        if missing:
            QMessageBox.warning(
                self,
                "总结失败",
                "因为没有配置好以下必填项，无法开始总结：\n"
                + "、".join(missing)
                + "\n\n请到「设置」页填写并保存后重试。",
            )
            self.settings_requested.emit()
            return
        tasks: list[tuple[str, str, bool]] = []
        self._by_task_id = {}
        n = 0
        for card in self._cards:
            url = card.url()
            if not url:
                continue
            n += 1
            task_id = str(n)
            card.arm(task_id, card.economic())
            self._by_task_id[task_id] = card
            tasks.append((task_id, url, card.economic()))
        if not tasks:
            QMessageBox.warning(self, "提示", "请先添加并填写至少一个视频 URL")
            return
        self._running = True
        self._start_btn.setEnabled(False)
        self._add_btn.setEnabled(False)
        self.start_requested.emit(tasks)

    # ---- 供 MainWindow 调用 ----
    def output_root(self) -> Path:
        return Path(self._output_edit.text().strip() or str(app_root() / "summary_data"))

    def set_task_status(self, task_id: str, status: str) -> None:
        card = self._by_task_id.get(task_id)
        if card is not None:
            card.set_status(status)

    def advance_task(self, task_id: str, step_tag: str) -> None:
        card = self._by_task_id.get(task_id)
        if card is not None:
            card.advance(step_tag)

    def set_task_result(self, task_id: str, result: "SummarizeResult") -> None:
        card = self._by_task_id.get(task_id)
        if card is not None:
            card.set_result(result)

    def set_task_duration(self, task_id: str, seconds: float) -> None:
        card = self._by_task_id.get(task_id)
        if card is not None:
            card.set_video_duration(seconds)

    def set_task_cost(self, task_id: str, cost_yuan: float) -> None:
        card = self._by_task_id.get(task_id)
        if card is not None:
            card.set_cost(cost_yuan)

    def set_summary_label(self, text: str) -> None:
        self._summary_label.setStyleSheet("color: #5B5F66;")
        self._summary_label.setText(text)

    def on_batch_finished(self, ok: int, fail: int, cancelled: int, total_cost: float | None) -> None:
        """批次收尾：恢复编辑，并在汇总栏内联显示完成状态与费用合计（不弹窗）。"""
        self._running = False
        self._start_btn.setEnabled(True)
        self._add_btn.setEnabled(True)
        for card in self._cards:
            card.re_enable()
        cost_txt = f" · 预估费用合计 约 ¥{total_cost:.2f}" if total_cost is not None else ""
        cancel_txt = f" · 已取消 {cancelled}" if cancelled else ""
        if fail == 0 and cancelled == 0:
            text, color = f"全部完成：成功 {ok} 个{cost_txt}", "#16A34A"
        elif ok == 0 and fail == 0:
            text, color = f"已全部取消{cost_txt}", "#6B7280"
        elif ok == 0:
            text, color = f"全部失败：共 {fail} 个{cancel_txt}{cost_txt}", "#DC2626"
        else:
            text, color = f"部分完成：成功 {ok} · 失败 {fail}{cancel_txt}{cost_txt}", "#D97706"
        self._summary_label.setText(text)
        self._summary_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def open_output_dir(self) -> None:
        path = self.output_root()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ---- 内部 ----
    def _on_browse(self) -> None:
        current = self._output_edit.text().strip() or str(app_root() / "summary_data")
        chosen = QFileDialog.getExistingDirectory(self, "选择输出目录", current)
        if chosen:
            self._output_edit.setText(chosen)
