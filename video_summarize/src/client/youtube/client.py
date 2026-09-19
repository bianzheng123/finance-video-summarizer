"""YouTube 视频下载客户端 — 经 proxychains 翻墙调用 yt-dlp 子进程。

YouTube 在本机环境被墙，in-process 的 yt-dlp 直连无法出网。故 YouTube 的
取信息 / 下载都改为子进程方式：`proxychains -q python -m yt_dlp ...`，
经 /etc/proxychains.conf 配置的 socks5 代理出网（实测 stdout 干净、可解析 JSON）。

- get_video_info：`yt_dlp -J` 拿完整 info dict（含 upload_date/timestamp 等发布元信息）。
- download_video：`yt_dlp ... --print after_move:filepath` 拿最终文件路径，取不到再 glob 兜底。
- get_video_info_from_url：bilibili 降级用（国内直连），保持 in-process 不变。
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yt_dlp

from .exceptions import YouTubeDownloadError, YouTubeVideoNotFoundError

logger = logging.getLogger(__name__)

# YouTube cookie（可选）：存在才传给 yt-dlp
_YOUTUBE_COOKIE_PATH = (
    Path(__file__).parent.parent.parent.parent / "config_local" / "cookie" / "youtube.txt"
)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# 媒体文件后缀（download 后 glob 兜底用）
_MEDIA_EXTS = (".mp4", ".mkv", ".webm", ".m4a", ".mp3", ".flv", ".mov", ".aac", ".opus")


def _proxychains_prefix() -> list[str]:
    """返回 proxychains 命令前缀；找不到二进制则告警并回退直连（[]）。

    可用环境变量 YOUTUBE_PROXYCHAINS 覆盖二进制名（默认 proxychains）。
    """
    binary = os.environ.get("YOUTUBE_PROXYCHAINS", "proxychains")
    path = shutil.which(binary)
    if not path:
        logger.warning("[YouTube] 未找到 %s，将尝试直连（很可能被墙）", binary)
        return []
    # -q 让 proxychains 静默，保证 stdout 只有 yt-dlp 的输出
    return [path, "-q"]


def _ytdlp_base_cmd() -> list[str]:
    """`python -m yt_dlp`，确保命中当前 venv 的 yt-dlp。"""
    return [sys.executable, "-m", "yt_dlp"]


def _cookie_args() -> list[str]:
    if _YOUTUBE_COOKIE_PATH.exists():
        return ["--cookies", str(_YOUTUBE_COOKIE_PATH)]
    return []


class YouTubeClient:
    """YouTube 视频下载客户端（经 proxychains 翻墙）。

    进程内按 output_dir 单例，避免重复初始化。
    """

    _INSTANCES: dict[str, "YouTubeClient"] = {}

    def __new__(cls, output_dir: Path | None = None) -> "YouTubeClient":
        key = str(output_dir)
        inst = cls._INSTANCES.get(key)
        if inst is None:
            inst = super().__new__(cls)
            cls._INSTANCES[key] = inst
        return inst

    def __init__(self, output_dir: Path | None = None) -> None:
        """初始化客户端

        Args:
            output_dir: 视频下载输出目录，默认为临时目录
        """
        if getattr(self, "_initialized", False):
            return
        self.output_dir = output_dir
        self._initialized = True
        logger.info("YouTubeClient 初始化成功（经 proxychains 翻墙）")

    # ── 子进程执行 ──────────────────────────────────────────

    @staticmethod
    def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
        """执行子进程，捕获 stdout/stderr（文本）。"""
        logger.info("[YouTube] 执行: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    async def download_video(
        self,
        url: str,
        output_dir: Path | None = None,
        format: str = "bestvideo+bestaudio/best",
        merge_output_format: str = "mp4",
        **kwargs: Any,
    ) -> Path:
        """下载 YouTube 视频（经 proxychains 翻墙）

        保持 async 签名（调用方用 asyncio.run 包裹）；内部走阻塞子进程即可，
        同一事件循环内无并发下载。

        Args:
            url: YouTube 视频 URL
            output_dir: 输出目录（可选，覆盖默认值）
            format: 视频格式（yt-dlp 格式字符串）
            merge_output_format: 合并输出格式（默认 mp4）
            **kwargs: 兼容旧签名，忽略

        Returns:
            下载的视频文件路径

        Raises:
            YouTubeDownloadError: 下载失败
            YouTubeVideoNotFoundError: 视频不存在
        """
        target_dir = output_dir or self.output_dir or Path(tempfile.mkdtemp())
        target_dir.mkdir(parents=True, exist_ok=True)

        outtmpl = str(target_dir / "video.%(ext)s")
        cmd = [
            *_proxychains_prefix(),
            *_ytdlp_base_cmd(),
            "-f", format,
            "--merge-output-format", merge_output_format,
            "-o", outtmpl,
            "--no-playlist",
            "--no-simulate",
            "--print", "after_move:filepath",
            "--no-warnings",
            "--retries", "10",
            "--fragment-retries", "10",
            "--user-agent", _UA,
            *_cookie_args(),
            url,
        ]

        try:
            # 下载可能很久：给 2 小时上限
            proc = self._run(cmd, timeout=7200)
        except subprocess.TimeoutExpired:
            raise YouTubeDownloadError(f"下载超时: {url}")

        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            logger.error("[YouTube] 下载失败: %s\n%s", url, err[-1000:])
            if "Video unavailable" in err or "not available" in err:
                raise YouTubeVideoNotFoundError(f"视频不存在或无法访问: {url}")
            raise YouTubeDownloadError(f"下载失败: {err[-500:]}")

        # 从 stdout 末行取最终文件路径
        video_path = self._resolve_downloaded_path(proc.stdout, target_dir)
        if video_path is None:
            raise YouTubeDownloadError(f"下载完成但未找到输出文件: {target_dir}")

        logger.info("[YouTube] 下载成功: %s", video_path.name)
        return video_path

    @staticmethod
    def _resolve_downloaded_path(stdout: str, target_dir: Path) -> Path | None:
        """从 --print filepath 的 stdout 取路径；取不到就在 target_dir glob 兜底。"""
        for line in reversed((stdout or "").splitlines()):
            line = line.strip()
            if line and Path(line).exists():
                return Path(line)
        # 兜底：glob 目录里最大的媒体文件
        candidates = [
            p for p in target_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _MEDIA_EXTS
        ]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_size)
        return None

    def get_video_info(self, url: str) -> dict[str, Any]:
        """获取 YouTube 视频完整信息（不下载，经 proxychains 翻墙）

        返回 yt-dlp 的完整 info dict（剔除 formats 等超大字段前的完整字段仍在），
        含 title/uploader/duration/view_count/timestamp/upload_date 等发布元信息，
        供 video_info_adapter 取用并落盘 metadata.json。

        Raises:
            YouTubeVideoNotFoundError: 视频不存在或获取失败
        """
        cmd = [
            *_proxychains_prefix(),
            *_ytdlp_base_cmd(),
            "-J",
            "--no-playlist",
            "--no-warnings",
            "--user-agent", _UA,
            *_cookie_args(),
            url,
        ]
        try:
            proc = self._run(cmd, timeout=180)
        except subprocess.TimeoutExpired:
            raise YouTubeVideoNotFoundError(f"获取视频信息超时: {url}")

        if proc.returncode != 0 or not proc.stdout.strip():
            err = (proc.stderr or "").strip()
            logger.error("[YouTube] 获取视频信息失败: %s\n%s", url, err[-1000:])
            raise YouTubeVideoNotFoundError(f"获取视频信息失败: {err[-500:] or '空输出'}")

        try:
            info = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            logger.error("[YouTube] 解析 yt-dlp JSON 失败: %s - %s", url, e)
            raise YouTubeVideoNotFoundError(f"解析视频信息失败: {url}")

        if not isinstance(info, dict):
            raise YouTubeVideoNotFoundError(f"视频信息格式异常: {url}")

        info.setdefault("url", url)
        logger.info(
            "获取 YouTube 视频信息成功: uploader=%s, title=%s",
            info.get("uploader"), info.get("title"),
        )
        return info

    def get_video_info_from_url(self, url: str) -> dict[str, Any] | None:
        """从视频 URL 获取信息（不下载，in-process yt-dlp）。

        供 video_info_adapter 的 bilibili 降级路径使用（国内直连，不走代理）。
        支持 yt-dlp 支持的所有平台。获取失败返回 None。
        """
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "http_headers": {
                "User-Agent": _UA,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info("获取视频信息(in-process): %s", url)
                info = ydl.extract_info(url, download=False)
                if info is None:
                    logger.warning("视频不存在或无法访问: %s", url)
                    return None
                return ydl.sanitize_info(info)
        except Exception as e:
            logger.error("获取视频信息失败: %s - %s", url, e)
            return None
