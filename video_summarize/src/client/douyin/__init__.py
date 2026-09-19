"""抖音客户端模块 — 视频下载 + 直播录制 + 弹幕监控 + 短视频爬取。

DouyinMonitor（弹幕）依赖 betterproto / execjs / py_mini_racer，故不在此导出，
由调用方按需 `from src.client.douyin.danmaku import DouyinMonitor`。
短视频列举/下载（lister.py / downloader.py / abogus.py）同样按需导入，不在顶层导出。
"""

from .client import DouyinClient
from .recorder import DouyinRecorder
from .exceptions import (
    DouyinError,
    DouyinDownloadError,
    DouyinVideoNotFoundError,
)

__all__ = [
    "DouyinClient",
    "DouyinRecorder",
    "DouyinError",
    "DouyinDownloadError",
    "DouyinVideoNotFoundError",
]
