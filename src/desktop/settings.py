"""设置读写：API Key 字段清单 + 界面偏好（并发数）的 .env 读写。

纯数据读写层，不含任何 UI：字段清单与保存逻辑供 :mod:`settings_page` 使用；
密码类字段在上层用掩码显示。保存统一走 python-dotenv 的 ``set_key`` / ``unset_key``
写回 .env，并立即回写 ``os.environ`` 供当前进程使用。
"""

from __future__ import annotations

import os

from dotenv import set_key, unset_key

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

# 界面偏好（非敏感，持久化到应用 .env）。
CONCURRENCY_KEY = "DESKTOP_CONCURRENCY"
DEFAULT_CONCURRENCY = 2
CONCURRENCY_MIN = 1
CONCURRENCY_MAX = 8


def read_settings() -> dict[str, str]:
    """从 ``os.environ`` 读取当前 API Key 配置（已由 ``load_app_env`` 填充）。"""
    return {key: (os.environ.get(key) or "") for key, *_ in SETTINGS_FIELDS}


def save_settings(values: dict[str, str]) -> None:
    """把 API Key 字段值写回 .env 并立即刷新 ``os.environ``。空值从 .env 移除。"""
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


def _read_int(key: str, default: int) -> int:
    """读取整型 env 偏好，非法值回退默认。"""
    try:
        return int((os.environ.get(key) or "").strip() or default)
    except ValueError:
        return default


def _save_int(key: str, value: int) -> None:
    """整型偏好写 .env 并回写 ``os.environ``。"""
    set_key(str(env_path()), key, str(value))
    os.environ[key] = str(value)


def read_concurrency() -> int:
    """读取同时处理几个视频的并发数，缺省 2；越界钳制到合法区间。"""
    value = _read_int(CONCURRENCY_KEY, DEFAULT_CONCURRENCY)
    return max(CONCURRENCY_MIN, min(CONCURRENCY_MAX, value))


def save_concurrency(value: int) -> None:
    """保存并发数（越界钳制到合法区间）。"""
    _save_int(CONCURRENCY_KEY, max(CONCURRENCY_MIN, min(CONCURRENCY_MAX, value)))
