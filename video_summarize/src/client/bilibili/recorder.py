"""Bilibili 直播录制器 — streamlink + ffmpeg 管道。"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

from ..base import LiveStreamRecorder
from .client import BilibiliClient

logger = logging.getLogger(__name__)


class BilibiliRecorder(LiveStreamRecorder):
    """Bilibili 直播录制器 — 使用 streamlink + ffmpeg 管道。"""

    platform = "bilibili"

    def __init__(self, room_id: int) -> None:
        self.room_id = room_id
        self._client = BilibiliClient()

    async def is_living(self) -> tuple[bool, str]:
        try:
            return await self._client.check_live_status(self.room_id)
        except Exception as e:
            logger.error("[bilibili] 查询 room_id=%d 失败: %s", self.room_id, e)
            raise

    async def record(self, output_path: Path) -> None:
        url = f"https://live.bilibili.com/{self.room_id}"
        logger.info("[bilibili] 开始录制: %s → %s", url, output_path)
        streamlink_cmd = ["streamlink", "--stdout", url, "best"]
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", "pipe:0",
            "-c:v", "libx264",
            "-b:v", "5000k",
            "-maxrate", "6000k",
            "-bufsize", "10000k",
            "-vf", "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease,"
                   "crop=trunc(iw/2)*2:trunc(ih/2)*2,fps=30",
            "-g", "60",
            "-x264-params", "keyint=60:min-keyint=60",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-movflags", "+faststart",
            str(output_path),
        ]
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._run_pipeline, streamlink_cmd, ffmpeg_cmd)

    @staticmethod
    def _run_pipeline(streamlink_cmd: list, ffmpeg_cmd: list) -> None:
        with subprocess.Popen(streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as sl:
            result = subprocess.run(ffmpeg_cmd, stdin=sl.stdout, capture_output=True)
            sl.stdout.close()
        if result.returncode != 0:
            logger.warning("[bilibili] ffmpeg 退出码: %d（直播可能已结束）", result.returncode)
        else:
            logger.info("[bilibili] 录制完成")
