"""视频信息适配器：统一处理不同平台的视频信息获取"""

import asyncio
import json
import logging
import re
from typing import Any, Optional
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from bilibili_api import video

from src.client.youtube.client import YouTubeClient
from src.client.douyin import DouyinClient

logger = logging.getLogger(__name__)


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "bilibili.com" in host:
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            return f"bilibili:{parts[-1]}"
    if "youtube.com" in host:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            return f"youtube:{video_id}"
    return url


def extract_bvid(url: str) -> str | None:
    """从 B 站 URL 中提取 bvid，非 B 站 URL 返回 None。"""
    parsed = urlparse(url)
    if "bilibili.com" in parsed.netloc.lower():
        parts = [p for p in parsed.path.split("/") if p]
        for part in parts:
            if part.startswith("BV"):
                return part
    return None

class VideoInfoAdapter:
    """视频信息适配器，统一获取不同平台的视频元信息"""

    def __init__(self) -> None:
        self._youtube_client = None
        self._douyin_client = None

    def _get_youtube_client(self) -> YouTubeClient:
        """延迟初始化 YouTubeClient"""
        if self._youtube_client is None:
            self._youtube_client = YouTubeClient()
        return self._youtube_client

    def _get_douyin_client(self) -> DouyinClient:
        """延迟初始化 DouyinClient"""
        if self._douyin_client is None:
            self._douyin_client = DouyinClient()
        return self._douyin_client

    def get_video_info(self, url: str) -> dict[str, Any] | None:
        """获取视频信息（统一接口）
        
        Args:
            url: 视频页面 URL
            
        Returns:
            视频信息字典，包含：
            - author: UP主/上传者名称
            - title: 视频标题
            - platform: 平台名称（bilibili/youtube/unknown）
            - duration: 视频时长（秒）
            - view_count: 观看次数
            
            如果获取失败返回 None
        """
        platform = self._get_platform(url)
        
        if platform == "bilibili":
            return self._get_bilibili_video_info(url)
        elif platform == "youtube":
            return self._get_youtube_video_info(url)
        elif platform == "douyin":
            return self._get_douyin_video_info(url)
        else:
            logger.warning("不支持的平台: %s", url)
            return None

    def _get_platform(self, url: str) -> str:
        """从 URL 推断平台名称"""
        url_lower = url.lower()
        if "bilibili.com" in url_lower:
            return "bilibili"
        elif "youtube.com" in url_lower or "youtu.be" in url_lower:
            return "youtube"
        elif "douyin.com" in url_lower:
            return "douyin"
        return "unknown"

    def _get_bilibili_video_info(self, url: str) -> dict[str, Any] | None:
        """获取 B站视频信息"""
        bvid = extract_bvid(url)
        if not bvid:
            logger.warning("无法从 URL 提取 BVID: %s", url)
            return None

        try:
            # 使用 bilibili_api 获取视频信息（公开视频无需登录态）
            v = video.Video(bvid=bvid)
            info = asyncio.run(v.get_info())
            
            # 提取视频信息
            author = info.get("owner", {}).get("name", "")
            title = info.get("title", "")
            duration = info.get("duration", 0)
            view_count = info.get("view", 0)
            
            if not author:
                logger.warning("未能获取 B站视频作者信息: %s", url)
                return None
            
            logger.info("获取 B站视频信息成功: author=%s, title=%s", author, title)
            return {
                "author": author,
                "title": title,
                "platform": "bilibili",
                "duration": duration,
                "view_count": view_count,
                "url": url,
                # 完整原始元信息（含 pubdate/owner/stat/desc/pages），供落盘 metadata.json
                "raw": info,
            }
        except Exception as e:
            logger.error("获取 B站视频信息失败: %s - %s", url, e)
            # 降级使用 yt-dlp
            return self._get_video_info_fallback(url, "bilibili")

    def _get_youtube_video_info(self, url: str) -> dict[str, Any] | None:
        """获取 YouTube 视频信息"""
        try:
            client = self._get_youtube_client()
            info = client.get_video_info(url)
            
            if info:
                author = info.get("uploader", "")
                title = info.get("title", "")
                
                if not author:
                    logger.warning("未能获取 YouTube 视频作者信息: %s", url)
                    return None
                
                logger.info("获取 YouTube 视频信息成功: author=%s, title=%s", author, title)
                return {
                    "author": author,
                    "title": title,
                    "platform": "youtube",
                    "duration": info.get("duration", 0),
                    "view_count": info.get("view_count", 0),
                    "url": url,
                    # yt-dlp 完整 info（含 upload_date/timestamp 等），供落盘 metadata.json
                    "raw": info,
                }
            return None
        except Exception as e:
            logger.error("获取 YouTube 视频信息失败: %s - %s", url, e)
            return None

    def _get_douyin_video_info(self, url: str) -> dict[str, Any] | None:
        """获取抖音视频信息"""
        try:
            client = self._get_douyin_client()
            info = client.get_video_info(url)

            if info:
                author = info.get("uploader", "")
                title = info.get("title", "")

                if not author:
                    logger.warning("未能获取抖音视频作者信息: %s", url)
                    return None

                logger.info("获取抖音视频信息成功: author=%s, title=%s", author, title)
                return {
                    "author": author,
                    "title": title,
                    "platform": "douyin",
                    "duration": info.get("duration", 0),
                    "view_count": info.get("view_count", 0),
                    "url": url,
                    # yt-dlp 完整 info（含 upload_date/timestamp 等），供落盘 metadata.json
                    "raw": info,
                }
            return None
        except Exception as e:
            logger.error("获取抖音视频信息失败: %s - %s", url, e)
            return None

    def _get_video_info_fallback(self, url: str, platform: str) -> dict[str, Any] | None:
        """降级方案：使用 yt-dlp 获取视频信息"""
        try:
            client = self._get_youtube_client()
            info = client.get_video_info_from_url(url)
            
            if info:
                author = info.get("uploader", "") or info.get("channel", "")
                
                if not author:
                    logger.warning("降级方案也未能获取作者信息: %s", url)
                    return None
                
                logger.info("降级方案获取视频信息成功: author=%s, title=%s", author, info.get("title"))
                return {
                    "author": author,
                    "title": info.get("title", ""),
                    "platform": platform,
                    "duration": info.get("duration", 0),
                    "view_count": info.get("view_count", 0),
                    "url": url,
                    # 降级路径拿到的完整 info，供落盘 metadata.json
                    "raw": info,
                }
            return None
        except Exception as e:
            logger.error("降级方案获取视频信息失败: %s - %s", url, e)
            return None


# 全局实例
video_info_adapter = VideoInfoAdapter()
