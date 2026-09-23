"""设置弹窗：填写 DEEPSEEK / 腾讯云 key，读写 exe 旁（或项目根）的 .env。

字段清单对齐仓库根 ``.env.example``。密码类字段用掩码显示；保存时用 python-dotenv 的
``set_key`` / ``unset_key`` 写回 .env，并立即回写 ``os.environ`` 供当前进程使用。
"""

from __future__ import annotations

import os

from dotenv import set_key, unset_key
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from .env_loader import env_path

# (env key, 显示名, 分组, 是否必填)
SETTINGS_FIELDS: list[tuple[str, str, str, bool]] = [
    ("DEEPSEEK_API_KEY", "DeepSeek API Key", "LLM", True),
    ("TENCENT_LLM_API_KEY", "腾讯云混元 LLM API Key（可选）", "LLM", False),
    ("TENCENT_SECRET_ID", "腾讯云 SecretId", "ASR", True),
    ("TENCENT_SECRET_KEY", "腾讯云 SecretKey", "ASR", True),
    ("TENCENT_HOTWORD_ID", "腾讯云热词表 ID（可选）", "ASR", False),
    ("TENCENT_COS_BUCKET", "腾讯云 COS Bucket", "COS", True),
    ("TENCENT_COS_REGION", "腾讯云 COS 地域", "COS", False),
]


def read_settings() -> dict[str, str]:
    """从 os.environ 读取当前配置（已由 load_app_env 填充）。"""
    return {key: (os.environ.get(key) or "") for key, *_ in SETTINGS_FIELDS}


def save_settings(values: dict[str, str]) -> None:
    """把字段值写回 .env 并立即刷新 os.environ。空值从 .env 移除。"""
    path = env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    for key, *_ in SETTINGS_FIELDS:
        val = (values.get(key) or "").strip()
        if val:
            set_key(str(path), key, val)
            os.environ[key] = val
        else:
            unset_key(str(path), key)
            os.environ.pop(key, None)


class SettingsDialog(QDialog):
    """API Key 配置弹窗。分组展示 LLM / ASR / COS 字段，必填项缺失时提示。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("配置 API Key")
        self.setMinimumWidth(520)

        self._edits: dict[str, QLineEdit] = {}
        self._required_keys: set[str] = {key for key, *_ , req in SETTINGS_FIELDS if req}

        layout = QVBoxLayout(self)

        intro = QLabel(
            "首次使用请填写以下密钥，保存后会写入应用目录下的 .env 文件。\n"
            "（可随时从主窗口「设置」按钮重新修改）"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # 按分组归组字段
        groups: dict[str, QFormLayout] = {}
        for key, label, group, _required in SETTINGS_FIELDS:
            if group not in groups:
                box = QGroupBox(group)
                form = QFormLayout(box)
                layout.addWidget(box)
                groups[group] = form
            edit = QLineEdit()
            if key.endswith("_KEY") or key == "TENCENT_SECRET_KEY":
                edit.setEchoMode(QLineEdit.EchoMode.Password)
            display = label if not _required else f"{label} *"
            groups[group].addRow(display, edit)
            self._edits[key] = edit

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # 预填当前值
        current = read_settings()
        for key, edit in self._edits.items():
            edit.setText(current.get(key, ""))

    def _on_save(self) -> None:
        values = {key: edit.text() for key, edit in self._edits.items()}
        missing = [key for key in self._required_keys if not (values.get(key) or "").strip()]
        if missing:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self, "缺少必填项",
                "以下必填项未填写：\n" + "\n".join(f"- {key}" for key in missing),
            )
            return
        save_settings(values)
        self.accept()
