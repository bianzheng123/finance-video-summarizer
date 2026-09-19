"""client 层共享基类与工具 — 录制器 / 弹幕监控 / 短视频列举的公共抽象。

不依赖任何可选三方库（betterproto / playwright / py_mini_racer 等），
可安全地在包顶层导入。各平台的具体 Recorder / Monitor / Lister 实现按需导入其依赖。
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

logger = logging.getLogger(__name__)

# cookie 集中存放于 config_local/cookie/，与 bilibili client 保持一致
COOKIE_DIR = Path(__file__).resolve().parent.parent.parent / "config_local" / "cookie"


# ─────────────────────────────────────────────
# 弹幕监控
# ─────────────────────────────────────────────

class PlatformMonitor(ABC):
    """弹幕监控基类。各平台实现 is_live / run_capture / shutdown。"""

    name: ClassVar[str]

    @abstractmethod
    async def is_live(self) -> tuple[bool, str]:
        """Return (living, title). title may be empty string."""

    @abstractmethod
    async def run_capture(self, writer: Any, stop_event: asyncio.Event) -> None:
        """Stream events into writer. Return when stream ends or stop_event is set."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Idempotent cleanup of websocket / browser / sessions."""


# ─────────────────────────────────────────────
# 直播录制
# ─────────────────────────────────────────────

class LiveStreamRecorder(ABC):
    """直播流录制器基类。各平台实现 is_living / record。"""

    platform: ClassVar[str]

    @abstractmethod
    async def is_living(self) -> tuple[bool, str]:
        """检测是否开播。返回 (是否开播, 直播标题)。"""

    @abstractmethod
    async def record(self, output_path: Path) -> None:
        """录制直播流到 output_path（阻塞直到直播结束或流断开）。"""


# ─────────────────────────────────────────────
# cookie 工具
# ─────────────────────────────────────────────

def load_json_cookie(path: Optional[Path]) -> Union[dict, list, None]:
    """加载 JSON cookie 文件，返回原始 dict / list（弹幕监控用）。"""
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


_COOKIE_STRING_CACHE: dict[str, Optional[str]] = {}


def load_cookie_string(cookie_file: Path) -> Optional[str]:
    """加载 JSON cookie 文件并拼成 cookie 字符串（录制 / 爬取共用）。

    支持三种格式：
      - dict：{"ttwid": "..."} → "ttwid=..."
      - Cookie-Editor 导出的 list：[{"name": "did", "value": "..."}, ...] → "did=..."
      - 已拼好的字符串：原样返回
    按文件路径缓存：同一份 cookie 只读盘并打印一次日志，避免每个 recorder
    重复加载刷屏（一个平台有多个主播时尤为明显）。
    """
    key = str(cookie_file)
    if key in _COOKIE_STRING_CACHE:
        return _COOKIE_STRING_CACHE[key]

    if not cookie_file.exists():
        _COOKIE_STRING_CACHE[key] = None
        return None
    with cookie_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, str):
        result = data or None
    elif isinstance(data, dict) and data:
        result = "; ".join(f"{k}={v}" for k, v in data.items())
    elif isinstance(data, list):
        # Cookie-Editor 导出的 [{"name": .., "value": ..}, ...]
        parts = [
            f"{c['name']}={c['value']}"
            for c in data
            if isinstance(c, dict) and c.get("name") and c.get("value") is not None
        ]
        result = "; ".join(parts) or None
    else:
        result = None

    _COOKIE_STRING_CACHE[key] = result
    if result:
        logger.info("[client] 已加载登录 cookie: %s", cookie_file.name)
    return result


# ─────────────────────────────────────────────
# 短视频列举
# ─────────────────────────────────────────────

@dataclass
class VideoItem:
    """一条待处理的短视频（列举结果）。"""

    platform: str          # bilibili / douyin / kuaishou
    video_id: str          # 平台视频 ID：bvid / aweme_id / photo_id
    title: str
    publish_date: str      # YYYY-MM-DD（发布时间，本地时区）
    publish_ts: int        # 发布 unix 秒
    url: str               # 规范视频页 URL（bilibili 可直接转写；douyin/kuaishou 供落库/参考）
    blogger: str           # 归属博主名（watch_list 的 name），作为落库 author
    extra: dict = field(default_factory=dict)  # 平台专用下载线索（如 kuaishou 的 play_url）


class VideoLister(ABC):
    """平台短视频列举器接口。"""

    platform: str = "unknown"

    @abstractmethod
    def list_in_window(self, account_id: str, blogger: str, since: date, until: date) -> list[VideoItem]:
        """列出 account_id 在 [since, until]（含端点）发布的短视频。

        实现约定：
        - 按发布时间倒序翻页，翻到早于 since 即停止。
        - 任何网络/解析失败都应抛异常（上层按"硬失败停下"处理）。
        """
        raise NotImplementedError
