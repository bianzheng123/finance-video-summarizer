"""设置页：两块表单——「API 密钥」与「运行配置」，顶部附「如何配置密钥」指引。

- 配置指引：富文本 QLabel 展示 DeepSeek / 腾讯云 / COS 的申请地址（链接可点击，
  调系统默认浏览器打开），文案与 INSTALL_WINDOWS.md / doc/配置说明.md 保持一致。
- API 密钥：复用 :data:`settings.SETTINGS_FIELDS`，按 LLM / ASR / COS 分组渲染，
  密码类字段掩码、必填项标注并做缺失校验；每个字段悬停显示作用说明（tooltip）。
- 运行配置：只暴露「并发数（同时处理几个视频）」一个简单项。
- 保存统一走 :mod:`settings` 的读写函数（写 .env + 回写 ``os.environ``）。
- 整页包一层 QScrollArea，窗口较小时滚动查看。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .settings import (
    CONCURRENCY_MAX,
    CONCURRENCY_MIN,
    SETTINGS_FIELDS,
    read_concurrency,
    read_settings,
    save_concurrency,
    save_settings,
)

# 各字段的作用说明（悬停 tooltip），与 doc/配置说明.md / .env.example 保持一致。
_FIELD_HELP: dict[str, str] = {
    "DEEPSEEK_API_KEY": "DeepSeek 大模型密钥（LLM 分析）。申请：https://platform.deepseek.com",
    "TENCENT_LLM_API_KEY": "腾讯云混元 LLM 密钥（联网搜索 / 主题去重，可选）。",
    "TENCENT_SECRET_ID": "腾讯云 API SecretId（ASR 语音识别）。获取：https://console.cloud.tencent.com/cam/capi",
    "TENCENT_SECRET_KEY": "腾讯云 API SecretKey（ASR 语音识别）。获取：https://console.cloud.tencent.com/cam/capi",
    "TENCENT_HOTWORD_ID": "ASR 热词表 ID（可选，提升金融专有名词识别率）。",
    "TENCENT_COS_BUCKET": "COS 存储桶名（音频/字幕中转）。在腾讯云对象存储控制台创建。",
    "TENCENT_COS_REGION": "COS 地域，如 ap-shanghai。",
}

# 配置指引（富文本，链接可点击）。文案与 INSTALL_WINDOWS.md「密钥从哪来」一致。
_GUIDE_HTML = (
    "<b>密钥从哪来（首次使用必看）：</b><br>"
    "1. <b>DeepSeek 密钥</b>：访问 "
    '<a href="https://platform.deepseek.com">platform.deepseek.com</a> 注册并创建 API Key；<br>'
    "2. <b>腾讯云密钥</b>：访问 "
    '<a href="https://console.cloud.tencent.com/cam/capi">console.cloud.tencent.com/cam/capi</a> '
    "创建 SecretId / SecretKey；<br>"
    "3. <b>COS 存储桶</b>：在腾讯云对象存储控制台创建存储桶，把桶名与地域填到下方；<br>"
    "4. 其余为可选：混元 LLM 密钥（联网搜索/主题去重）、ASR 热词表 ID。<br>"
    "标 <b>*</b> 为必填项，保存后写入应用目录下的 .env 文件。"
)


class SettingsPage(QWidget):
    """设置页：配置指引 + API 密钥 + 运行配置，底部保存。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._edits: dict[str, QLineEdit] = {}
        self._required_keys = {key for key, *_rest, req in SETTINGS_FIELDS if req}

        # 外层：滚动容器包裹全部内容，窗口较小时滚动查看
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # ---- 配置指引块 ----
        guide_box = QGroupBox("如何配置密钥")
        guide_layout = QVBoxLayout(guide_box)
        guide = QLabel(_GUIDE_HTML)
        guide.setWordWrap(True)
        guide.setOpenExternalLinks(True)
        guide.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse
            | Qt.TextInteractionFlag.TextSelectableByMouse,
        )
        guide_layout.addWidget(guide)
        layout.addWidget(guide_box)

        # ---- API 密钥块 ----
        api_box = QGroupBox("API 密钥")
        api_layout = QVBoxLayout(api_box)
        api_layout.setSpacing(8)

        intro = QLabel("按上方指引申请密钥后填入下面各项；标 * 的为必填项，缺失将无法开始总结。")
        intro.setWordWrap(True)
        api_layout.addWidget(intro)

        groups: dict[str, QFormLayout] = {}
        for key, label, group, required in SETTINGS_FIELDS:
            if group not in groups:
                box = QGroupBox(group)
                form = QFormLayout(box)
                api_layout.addWidget(box)
                groups[group] = form
            edit = QLineEdit()
            edit.setToolTip(_FIELD_HELP.get(key, ""))
            if key.endswith("_KEY") or key == "TENCENT_SECRET_KEY":
                edit.setEchoMode(QLineEdit.EchoMode.Password)
            display = label if not required else f"{label} *"
            groups[group].addRow(display, edit)
            self._edits[key] = edit
        layout.addWidget(api_box)

        # ---- 运行配置块 ----
        run_box = QGroupBox("运行配置")
        run_form = QFormLayout(run_box)

        self._concurrency_spin = QSpinBox()
        self._concurrency_spin.setRange(CONCURRENCY_MIN, CONCURRENCY_MAX)
        self._concurrency_spin.setValue(read_concurrency())
        self._concurrency_spin.setSuffix(" 个视频")
        self._concurrency_spin.setToolTip("同时并行总结几个视频（默认 2，避免 ASR / LLM 同时过载）")
        run_form.addRow("同时处理几个视频", self._concurrency_spin)
        layout.addWidget(run_box)

        # ---- 保存 ----
        save_btn = QPushButton("保存设置")
        save_btn.setObjectName("PrimaryButton")
        save_btn.clicked.connect(self._on_save)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #16A34A;")
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(self._status_label)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)

        # 预填当前值
        current = read_settings()
        for key, edit in self._edits.items():
            edit.setText(current.get(key, ""))

    def _on_save(self) -> None:
        values = {key: edit.text() for key, edit in self._edits.items()}
        missing = [key for key in self._required_keys if not (values.get(key) or "").strip()]
        if missing:
            QMessageBox.warning(
                self,
                "缺少必填项",
                "以下必填项未填写：\n" + "\n".join(f"- {key}" for key in missing),
            )
            return
        save_settings(values)
        save_concurrency(self._concurrency_spin.value())
        self._status_label.setText("已保存")
