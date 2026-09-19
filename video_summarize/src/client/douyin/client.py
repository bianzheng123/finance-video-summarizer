"""抖音视频下载客户端 — yt-dlp + cookie 下载，支持 URL 归一化。

抖音网页端常见的几种 URL 形态：
- 规范视频页：https://www.douyin.com/video/{aweme_id}
- 用户主页弹窗：https://www.douyin.com/user/{sec_uid}?modal_id={aweme_id}
- 分享短链：https://v.douyin.com/{code}/

yt-dlp 的 DouyinIE 只认 https://www.douyin.com/video/{id}，
故下载/取信息前统一用 normalize_url 归一化为规范视频页。
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

import requests
import yt_dlp

from .exceptions import DouyinDownloadError, DouyinVideoNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_COOKIE_PATH = Path(__file__).parent.parent.parent.parent / "config_local" / "cookie" / "douyin.json"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)


class DouyinClient:
    """抖音视频下载客户端（使用 cookie）

    进程内按 cookie_path 单例：同一份 cookie 只加载一次。
    """

    _INSTANCES: dict[str, "DouyinClient"] = {}

    def __new__(cls, cookie_path: Path | None = None) -> "DouyinClient":
        key = str(cookie_path or DEFAULT_COOKIE_PATH)
        inst = cls._INSTANCES.get(key)
        if inst is None:
            inst = super().__new__(cls)
            cls._INSTANCES[key] = inst
        return inst

    def __init__(self, cookie_path: Path | None = None) -> None:
        """初始化客户端

        Args:
            cookie_path: cookie 配置文件路径，默认为 config_local/cookie/douyin.json
        """
        if getattr(self, "_initialized", False):
            return
        self.cookie_path = cookie_path or DEFAULT_COOKIE_PATH
        self._cookies = self._load_cookies()
        self._initialized = True
        logger.info("DouyinClient 初始化成功")

    @staticmethod
    def normalize_url(url: str) -> str:
        """把各种抖音 URL 归一化为 yt-dlp 可识别的规范视频页。

        Args:
            url: 任意抖音 URL（用户主页弹窗 / 规范视频页 / 分享短链）

        Returns:
            https://www.douyin.com/video/{aweme_id} 形式的 URL；
            若无法提取 aweme_id 则原样返回。
        """
        # 分享短链先跟随重定向拿到真实地址
        if "v.douyin.com" in url:
            try:
                resp = requests.head(url, allow_redirects=True, timeout=10,
                                     headers={"User-Agent": _UA})
                url = resp.url
            except requests.RequestException as e:
                logger.warning("[douyin] 短链解析失败，沿用原 URL: %s - %s", url, e)

        aweme_id = DouyinClient._extract_aweme_id(url)
        if aweme_id:
            return f"https://www.douyin.com/video/{aweme_id}"

        logger.warning("[douyin] 无法从 URL 提取 aweme_id，沿用原 URL: %s", url)
        return url

    @staticmethod
    def _extract_aweme_id(url: str) -> str | None:
        """从抖音 URL 提取 aweme_id（视频 ID）。"""
        parsed = urlparse(url)

        # 1. 用户主页弹窗：?modal_id={id}
        modal_id = parse_qs(parsed.query).get("modal_id", [None])[0]
        if modal_id and modal_id.isdigit():
            return modal_id

        # 2. 规范路径：/video/{id} 或 /note/{id}
        m = re.search(r"/(?:video|note|share/video)/(\d+)", parsed.path)
        if m:
            return m.group(1)

        return None

    def download_video(
        self,
        url: str,
        output_dir: Path | None = None,
        format: str = "bestvideo+bestaudio/best",
        merge_output_format: str = "mp4",
        **kwargs: Any,
    ) -> Path:
        """下载抖音视频（使用 cookie）

        Args:
            url: 抖音视频 URL（任意形态，内部会归一化）
            output_dir: 输出目录（可选，默认临时目录）
            format: 视频格式（yt-dlp 格式字符串）
            merge_output_format: 合并输出格式（默认 mp4）
            **kwargs: 其他 yt-dlp 选项

        Returns:
            下载的视频文件路径

        Raises:
            DouyinDownloadError: 下载失败
            DouyinVideoNotFoundError: 视频不存在
        """
        normalized = self.normalize_url(url)
        target_dir = output_dir or Path(tempfile.mkdtemp())
        target_dir.mkdir(parents=True, exist_ok=True)

        cookie_file = self._write_netscape_cookies(self._cookies) if self._cookies else None

        ydl_opts = {
            "format": format,
            "merge_output_format": merge_output_format,
            "outtmpl": str(target_dir / "%(title)s.%(ext)s"),
            "quiet": False,
            "no_warnings": False,
            "retries": 10,
            "fragment_retries": 10,
            "http_headers": {
                "User-Agent": _UA,
                "Referer": "https://www.douyin.com/",
            },
            **({"cookiefile": str(cookie_file)} if cookie_file else {}),
            **kwargs,
        }

        try:
            logger.info("开始下载抖音视频: %s", normalized)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(normalized, download=True)

                if info is None:
                    raise DouyinVideoNotFoundError(f"视频不存在或无法访问: {normalized}")

                filename = ydl.prepare_filename(info)
                video_path = Path(filename)

                if video_path.exists():
                    logger.info("下载成功: %s", video_path.name)
                    return video_path
                raise DouyinDownloadError(f"下载完成但文件不存在: {filename}")

        except (DouyinDownloadError, DouyinVideoNotFoundError):
            raise
        except Exception as e:
            logger.error("下载失败: %s - %s", normalized, e)
            raise DouyinDownloadError(f"下载失败: {e}")
        finally:
            if cookie_file and cookie_file.exists():
                cookie_file.unlink()

    def get_video_info(self, url: str) -> dict[str, Any] | None:
        """获取抖音视频信息（不下载）

        Args:
            url: 抖音视频 URL（任意形态）

        Returns:
            视频信息字典（title / uploader / duration / view_count / url），
            获取失败返回 None
        """
        normalized = self.normalize_url(url)
        cookie_file = self._write_netscape_cookies(self._cookies) if self._cookies else None

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "http_headers": {
                "User-Agent": _UA,
                "Referer": "https://www.douyin.com/",
            },
            **({"cookiefile": str(cookie_file)} if cookie_file else {}),
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info("获取抖音视频信息: %s", normalized)
                info = ydl.extract_info(normalized, download=False)
                if info is None:
                    logger.warning("抖音视频不存在或无法访问: %s", normalized)
                    return None
                # 返回 yt-dlp 完整 info（经 sanitize 变 JSON 安全），保留
                # timestamp/upload_date/uploader_id/like_count 等发布元信息，
                # 供 video_info_adapter 落盘 metadata.json。仍含 title/uploader/
                # duration/view_count 供上层规范化取用。
                full = ydl.sanitize_info(info)
                full.setdefault("url", normalized)
                return full
        except Exception as e:
            logger.error("获取抖音视频信息失败: %s - %s", normalized, e)
            return None
        finally:
            if cookie_file and cookie_file.exists():
                cookie_file.unlink()

    def _load_cookies(self) -> dict[str, Any]:
        """加载抖音 cookie，配置文件不存在则返回空字典。"""
        if self.cookie_path.exists():
            with open(self.cookie_path, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            logger.info("已从 %s 加载 cookie", self.cookie_path)
            return cookies if isinstance(cookies, dict) else {}

        logger.warning("未找到抖音 cookie 文件，将尝试无 cookie 下载: %s", self.cookie_path)
        return {}

    @staticmethod
    def _write_netscape_cookies(cookies: dict[str, str]) -> Path:
        """将 cookie 字典写入 Netscape 格式临时文件供 yt-dlp 使用。"""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        f.write("# Netscape HTTP Cookie File\n")
        for name, value in cookies.items():
            f.write(f".douyin.com\tTRUE\t/\tFALSE\t0\t{name}\t{value}\n")
        f.close()
        return Path(f.name)
