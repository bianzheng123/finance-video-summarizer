"""Bilibili API 统一客户端"""

import logging
from pathlib import Path
import tempfile
from typing import Any

import yt_dlp

from bilibili_api import select_client, request_settings

from .exceptions import BilibiliAPIError

logger = logging.getLogger(__name__)


# bilibili_api 默认走 curl_cffi 客户端，但 curl_cffi 0.14.0 连 upos 上传 CDN 时
# 会协商出 HTTP/2 并触发帧解析错误（curl: (16) / CURLE_HTTP2），导致大文件上传秒崩。
# 切到 httpx 客户端并强制 HTTP/1.1，从根上绕开该 bug。整库请求都走 get_client()，
# 故全局设置一次即可覆盖全部请求。
select_client("httpx")
request_settings.set("http2", False)


class BilibiliClient:
    """Bilibili 视频下载客户端（无需登录态）。"""

    def download_video(
        self,
        url: str,
        output_dir: Path | None = None,
        format: str = "bestvideo+bestaudio/best",
        merge_output_format: str = "mp4",
        **kwargs: Any
    ) -> Path:
        """下载 B站视频

        Args:
            url: B站视频 URL
            output_dir: 输出目录（可选，覆盖默认值）
            format: 视频格式（yt-dlp 格式字符串）
            merge_output_format: 合并输出格式（默认 mp4）
            **kwargs: 其他 yt-dlp 选项

        Returns:
            下载的视频文件路径

        Raises:
            BilibiliAPIError: 下载失败
        """

        target_dir = output_dir or Path(tempfile.mkdtemp())
        target_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            "format": format,
            "merge_output_format": merge_output_format,
            "outtmpl": str(target_dir / "%(title)s.%(ext)s"),
            "quiet": False,
            "no_warnings": False,
            "retries": 10,
            "fragment_retries": 10,
            "concurrent_fragment_downloads": 4,
            "extractor_args": {"bilibili": {"prefer_multi_flv": ["1"]}},
            "format_sort": ["proto:https", "proto:http"],
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://www.bilibili.com",
                "Origin": "https://www.bilibili.com",
            },
            **kwargs,
        }

        try:

            logger.info("开始下载 B站视频: %s", url)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                if info is None:
                    raise BilibiliAPIError(f"视频不存在或无法访问: {url}")

                filename = ydl.prepare_filename(info)
                video_path = Path(filename)

                if video_path.exists():
                    logger.info("下载成功: %s", video_path.name)
                    return video_path
                else:
                    raise BilibiliAPIError(f"下载完成但文件不存在: {filename}")

        except BilibiliAPIError:
            raise
        except Exception as e:
            logger.error("下载失败: %s - %s", url, e)
            raise BilibiliAPIError(f"下载失败: {e}")
