# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：把财经视频总结桌面 GUI 打成单文件可执行程序。

数据文件收集策略（最小侵入，不改源码里散落的 ``Path(__file__)`` 相对路径）：
- ``src/**`` 下的 ``.jinja2`` / ``.txt`` / ``.json`` 模板与配置 → 按同构目录打进
  ``_MEIPASS/src/...``，使 frozen 下 ``FileSystemLoader(Path(__file__).parent)`` 仍命中；
- 仓库根 ``database/``、``config/`` → 打进 ``_MEIPASS/database``、``_MEIPASS/config``，
  使 ``Path(__file__).parent.parent.parent / "database|config"`` 仍命中；
- ``.env`` 不打包真实值：GUI 启动时读 exe 旁 ``.env``（见 ``src/desktop/env_loader.py``）。
"""

import glob
import os

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# ---- 收集 src 下数据文件（模板 / 提示词 / JSON 配置） ----
datas = []
for pattern in ("**/*.jinja2", "**/*.txt", "**/*.json"):
    for path in glob.glob(os.path.join(SPECPATH, "src", pattern), recursive=True):
        rel = os.path.relpath(path, SPECPATH)
        datas.append((path, os.path.dirname(rel)))

# ---- 仓库根数据目录（database / config） ----
for name in ("database", "config"):
    root = os.path.join(SPECPATH, name)
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.isfile(path):
            rel = os.path.relpath(path, SPECPATH)
            datas.append((path, os.path.dirname(rel)))

# ---- weasyprint（PDF 回退路径，主路径用系统 Edge/Chrome 无头）：收集其数据文件与动态库 ----
wp_datas, wp_binaries, wp_hiddenimports = collect_all("weasyprint")

a = Analysis(
    [os.path.join(SPECPATH, "desktop_app.py")],
    pathex=[SPECPATH],
    binaries=wp_binaries,
    datas=datas + wp_datas,
    hiddenimports=wp_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 不打进：torch（体积巨大的防御，非本项目依赖）、pytest（dev 组）、tkinter（GUI 用 PySide6）
    excludes=["torch", "pytest", "tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="finance-video-summarizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI 程序，不弹控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
