"""视频发布元信息落盘 — 下载/处理视频时在总结目录写 metadata.json。

在 summary_data/<platform>/<author>/<title>/ 下写一个 metadata.json，
尽量详细地记录视频的发布时间与其他发布元信息。三个能下载视频的平台
（bilibili / youtube / douyin）都通过本模块统一落盘。

设计要点：
- 只要拿到 save_path 就写/刷新（跳过下载分支也写），旧数据重跑会补上。
- 写入全程 try/except：metadata 落盘失败绝不阻断转录/渲染主流程。
- 保留一份规范化头部字段（发布时间、播放量等），外加完整 raw 原始元信息。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .llm_user_id import read_user_id
from ..log_config import step_logger

logger = logging.getLogger(__name__)

METADATA_FILENAME = "metadata.json"

# yt-dlp 原始 info 里体积巨大且对"发布元信息"无意义的字段，落盘前剔除，
# 避免 metadata.json 膨胀到几百 KB（formats 尤其夸张）。
_YTDLP_HEAVY_KEYS = (
    "formats",
    "requested_formats",
    "requested_downloads",
    "automatic_captions",
    "subtitles",
    "thumbnails",
    "heatmap",
    "fragments",
    "http_headers",
)


def _to_local_iso(unix_ts: int) -> tuple[str, str]:
    """unix 秒 → (本地 'YYYY-MM-DD HH:MM:SS', 本地 'YYYY-MM-DD')。"""
    dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S"), dt.strftime("%Y-%m-%d")


def _parse_yyyymmdd(value: Any) -> Optional[int]:
    """把 yt-dlp 的 upload_date/release_date（'YYYYMMDD'）转成当天 0 点的 unix 秒。"""
    if not value:
        return None
    s = str(value).strip()
    if len(s) != 8 or not s.isdigit():
        return None
    try:
        dt = datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]))
        return int(dt.timestamp())
    except (ValueError, OverflowError):
        return None


def extract_publish_time(platform: str, raw: Optional[dict]) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """从平台原始元信息里提取发布时间。

    Returns:
        (unix_timestamp, 'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD')，取不到则相应位为 None。
    """
    if not raw:
        return None, None, None

    ts: Optional[int] = None

    if platform == "bilibili":
        # bilibili_api get_info(): pubdate 为发布 unix 秒；ctime 为投稿时间兜底
        ts = raw.get("pubdate") or raw.get("ctime")
    else:
        # youtube / douyin 走 yt-dlp：timestamp 为 unix 秒；否则解析 upload_date
        ts = raw.get("timestamp") or raw.get("release_timestamp")
        if not ts:
            ts = _parse_yyyymmdd(raw.get("upload_date")) or _parse_yyyymmdd(raw.get("release_date"))

    try:
        ts = int(ts) if ts else None
    except (TypeError, ValueError):
        ts = None

    if not ts or ts <= 0:
        return None, None, None

    publish_time, publish_date = _to_local_iso(ts)
    return ts, publish_time, publish_date


def _prune_raw(platform: str, raw: Optional[dict]) -> Optional[dict]:
    """落盘前裁剪原始元信息里的超大字段（仅对 yt-dlp 系平台生效）。"""
    if not isinstance(raw, dict):
        return raw
    if platform == "bilibili":
        # bilibili get_info() 的 dict 体积可控，保留全部（含 owner/stat/desc/pages）
        return raw
    return {k: v for k, v in raw.items() if k not in _YTDLP_HEAVY_KEYS}


def build_metadata(
    url: str,
    platform: str,
    author: Optional[str],
    title: Optional[str],
    video_info: Optional[dict],
) -> dict:
    """构造 metadata.json 的内容：规范化头部字段 + 完整 raw 原始元信息。

    Args:
        url: 视频页面 URL
        platform: 平台名（bilibili/youtube/douyin/...）
        author: 作者名（已由上层从 video_info 取得，作为兜底）
        title: 标题（同上）
        video_info: video_info_adapter.get_video_info() 的返回；可能为 None，
                    其 "raw" 键含平台完整原始元信息。

    Returns:
        可 json 序列化的 dict。
    """
    raw = (video_info or {}).get("raw") if isinstance(video_info, dict) else None
    ts, publish_time, publish_date = extract_publish_time(platform, raw)

    # bilibili 的播放/点赞/评论在 raw["stat"] 里，不在顶层
    stat = raw.get("stat") if isinstance(raw, dict) and isinstance(raw.get("stat"), dict) else {}

    def _rget(*keys: str) -> Any:
        """从 raw（及 bilibili 的 stat 子字典）里按顺序取第一个非空值。"""
        if not isinstance(raw, dict):
            return None
        for k in keys:
            v = raw.get(k)
            if v in (None, "", 0) and stat:
                v = stat.get(k)
            if v not in (None, "", 0):
                return v
        return None

    # 平台视频 id：bilibili 用 bvid，yt-dlp 系用 id
    video_id = None
    if isinstance(raw, dict):
        video_id = raw.get("bvid") or raw.get("id") or raw.get("aweme_id")

    meta = {
        "platform": platform,
        "url": url,
        "video_id": video_id,
        "author": (video_info or {}).get("author") if isinstance(video_info, dict) else author,
        "title": (video_info or {}).get("title") if isinstance(video_info, dict) else title,
        "uploader": _rget("uploader", "owner"),
        "uploader_id": _rget("uploader_id", "channel_id", "uploader_url"),
        "duration": ((video_info or {}).get("duration") if isinstance(video_info, dict) else None) or _rget("duration"),
        "view_count": ((video_info or {}).get("view_count") if isinstance(video_info, dict) else None) or _rget("view_count", "view"),
        "like_count": _rget("like_count", "like"),
        "comment_count": _rget("comment_count", "reply_count", "reply"),
        "publish_timestamp": ts,
        "publish_time": publish_time,
        "publish_date": publish_date,
        "description": _rget("description", "desc"),
        "tags": _rget("tags", "categories"),
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw": _prune_raw(platform, raw),
    }
    # author/title 兜底：video_info 没有时用入参
    if not meta["author"]:
        meta["author"] = author
    if not meta["title"]:
        meta["title"] = title
    return meta


def save_metadata(
    save_path: Path,
    url: str,
    platform: str,
    author: Optional[str] = None,
    title: Optional[str] = None,
    video_info: Optional[dict] = None,
) -> Optional[Path]:
    """在 save_path 目录写 metadata.json（含发布时间等详细发布元信息）。

    只要拿到 save_path 就写/刷新（跳过下载分支也调用，旧数据重跑会补上）。
    全程 try/except：失败只 warning，绝不抛出，以免阻断转录/渲染主流程。

    Returns:
        写成功返回 metadata.json 路径，失败返回 None。
    """
    with step_logger("视频信息"):
        try:
            folder = Path(save_path)
            folder.mkdir(parents=True, exist_ok=True)
            meta = build_metadata(url, platform, author, title, video_info)
            # 覆盖写时保留已落盘的 user_id（LLM 缓存隔离 id，见 utils/llm_user_id），
            # 避免旧数据重跑换新 id 导致 DeepSeek 前缀缓存连续性断裂。
            existing_user_id = read_user_id(folder)
            if existing_user_id:
                meta["user_id"] = existing_user_id
            out = folder / METADATA_FILENAME
            with open(out, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
            logger.info(
                "已保存 metadata.json（发布时间=%s）: %s",
                meta.get("publish_time") or "未知", out,
            )
            return out
        except Exception as e:
            logger.warning("保存 metadata.json 失败（忽略，不影响主流程）: %s - %s", save_path, e)
            return None

