"""全局浅色主题：集中式 QSS 样式表。

单点维护整套配色体系（背景 / 卡片 / 主色 / 功能色），各页面只通过
``setObjectName`` 暴露少量样式锚点（``Sidebar`` / ``NavButton`` / ``UrlCard`` /
``PrimaryButton``），其余控件按类型选择器统一覆盖。页面内联样式（状态徽标、
进度条三态）优先级高于本样式表，配色语义保持一致。
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

# ---- 配色常量 ----
WINDOW_BG = "#F3F4F6"      # 窗口背景（浅灰）
PAGE_BG = "#F9FAFB"        # 页面 / 滚动区 / 日志区背景
CARD_BG = "#FFFFFF"        # 卡片 / 输入区
PRIMARY = "#2563EB"        # 主色（沿用「运行中」蓝）
PRIMARY_HOVER = "#1D4ED8"
PRIMARY_PRESSED = "#1E40AF"
BORDER = "#E5E7EB"         # 细边框
TEXT_PRIMARY = "#1F2937"
TEXT_SECONDARY = "#6B7280"
TEXT_DISABLED = "#9CA3AF"
SUCCESS = "#16A34A"
DANGER = "#DC2626"

APP_QSS = f"""
/* ---- 全局 ---- */
QMainWindow, QDialog, QMessageBox {{
    background-color: {WINDOW_BG};
}}
QWidget {{
    color: {TEXT_PRIMARY};
}}
QLabel {{
    background-color: transparent;
}}
QToolTip {{
    background-color: #FFFFFF;
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    padding: 4px 6px;
}}

/* ---- 侧边栏导航 ---- */
QWidget#Sidebar {{
    background-color: #1E293B;
}}
QPushButton#NavButton {{
    background-color: transparent;
    color: #CBD5E1;
    border: none;
    border-left: 4px solid transparent;
    font-size: 18px;
    font-weight: 600;
    text-align: left;
    padding-left: 24px;
}}
QPushButton#NavButton:hover {{
    background-color: #334155;
    color: #F1F5F9;
}}
QPushButton#NavButton:checked {{
    background-color: #334155;
    color: #FFFFFF;
    border-left: 4px solid {PRIMARY};
}}

/* ---- 页面容器与滚动区 ---- */
QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: {PAGE_BG};
    border: none;
}}
QScrollArea {{
    background-color: transparent;
}}

/* ---- URL 卡片 ---- */
QFrame#UrlCard {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

/* ---- 按钮 ---- */
QPushButton {{
    background-color: {CARD_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background-color: #F3F4F6;
    border-color: {TEXT_DISABLED};
}}
QPushButton:pressed {{
    background-color: #E5E7EB;
}}
QPushButton:disabled {{
    background-color: {WINDOW_BG};
    color: {TEXT_DISABLED};
    border-color: {BORDER};
}}
QPushButton#PrimaryButton {{
    background-color: {PRIMARY};
    color: #FFFFFF;
    border: none;
    font-weight: 600;
}}
QPushButton#PrimaryButton:hover {{
    background-color: {PRIMARY_HOVER};
}}
QPushButton#PrimaryButton:pressed {{
    background-color: {PRIMARY_PRESSED};
}}
QPushButton#PrimaryButton:disabled {{
    background-color: #93C5FD;
    color: #EFF6FF;
}}

/* ---- 输入控件 ---- */
QLineEdit, QSpinBox {{
    background-color: {CARD_BG};
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {PRIMARY};
    selection-color: #FFFFFF;
}}
QLineEdit:focus, QSpinBox:focus {{
    border: 1px solid {PRIMARY};
}}
QLineEdit:disabled, QSpinBox:disabled {{
    background-color: {WINDOW_BG};
    color: {TEXT_SECONDARY};
    border-color: {BORDER};
}}
QCheckBox {{
    background-color: transparent;
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid #D1D5DB;
    border-radius: 4px;
    background-color: {CARD_BG};
}}
QCheckBox::indicator:checked {{
    background-color: {PRIMARY};
    border-color: {PRIMARY};
}}
QCheckBox::indicator:disabled {{
    background-color: {WINDOW_BG};
    border-color: {BORDER};
}}

/* ---- 分组框 ---- */
QGroupBox {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 18px;
    padding-top: 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 2px 6px;
    color: {TEXT_PRIMARY};
}}
QGroupBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid #D1D5DB;
    border-radius: 4px;
    background-color: {CARD_BG};
}}
QGroupBox::indicator:checked {{
    background-color: {PRIMARY};
    border-color: {PRIMARY};
}}

/* ---- 进度条 ---- */
QProgressBar {{
    background-color: #E5E7EB;
    border: none;
    border-radius: 5px;
    max-height: 10px;
}}
QProgressBar::chunk {{
    background-color: {PRIMARY};
    border-radius: 5px;
}}

/* ---- 日志区 ---- */
QPlainTextEdit {{
    background-color: {PAGE_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
    color: {TEXT_PRIMARY};
    selection-background-color: {PRIMARY};
    selection-color: #FFFFFF;
}}

/* ---- 滚动条 ---- */
QScrollBar:vertical {{
    background-color: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background-color: #D1D5DB;
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_DISABLED};
}}
QScrollBar:horizontal {{
    background-color: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background-color: #D1D5DB;
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {TEXT_DISABLED};
}}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
    border: none;
    width: 0;
    height: 0;
}}
"""


def apply_theme(app: QApplication) -> None:
    """把全局浅色主题样式表应用到整个应用。"""
    app.setStyleSheet(APP_QSS)
