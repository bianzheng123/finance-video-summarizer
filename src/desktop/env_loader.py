"""桌面应用的 .env 定位与加载。

打包（frozen）时 .env 放在可执行文件旁（用户可写、可编辑），开发态用项目根 .env。
必须在 import ``src.*`` 之前调用 :func:`load_app_env`：src 各模块会在 import 时
``load_dotenv``（路径指向 ``_MEIPASS/.env``，frozen 下不存在 → no-op），先注入
``os.environ`` 才能保证 API Key 就位、不被覆盖。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ENV_FILENAME = ".env"

# 必填项：缺任意一个就认为「未配置」，主窗口启动时引导用户进设置页。
REQUIRED_KEYS = (
    "DEEPSEEK_API_KEY",
    "TENCENT_SECRET_ID",
    "TENCENT_SECRET_KEY",
    "TENCENT_COS_BUCKET",
)


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包产物中。"""
    return getattr(sys, "frozen", False)


def app_root() -> Path:
    """可执行文件所在目录（frozen）或项目根目录（开发态）。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    # src/desktop/env_loader.py → 项目根
    return Path(__file__).resolve().parent.parent.parent


def env_path() -> Path:
    """exe 旁 / 项目根的 .env 完整路径。"""
    return app_root() / _ENV_FILENAME


def load_app_env() -> Path:
    """加载 exe 旁 / 项目根的 .env 到 ``os.environ``（override=True），返回实际路径。

    文件不存在时静默返回路径（后续设置页会引导用户填写），不抛异常。
    """
    path = env_path()
    if path.exists():
        load_dotenv(dotenv_path=path, override=True)
    return path


def is_configured() -> bool:
    """必填项是否已全部就位（来源：os.environ，已由 load_app_env 填充）。"""
    return all((os.environ.get(k) or "").strip() for k in REQUIRED_KEYS)
