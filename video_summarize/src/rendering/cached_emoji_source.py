"""本地磁盘缓存的 emoji source。

pilmoji 默认每渲染一个 emoji 就去 emojicdn.elk.sh 拉一次 PNG，CDN 不稳定时
会抛 SSL/连接异常拖垮渲染。这里在 TwitterEmojiSource 外面套一层磁盘缓存：

- 命中缓存 → 直接读本地 PNG，零网络
- 未命中 → 拉一次 CDN，成功则落盘，后续永久走本地

缓存目录随项目走（assets/emoji_cache/），可纳入版本控制，部署后无需联网。
"""
from io import BytesIO
from pathlib import Path

from pilmoji.source import TwitterEmojiSource

# 缓存落在 video_summarize/assets/emoji_cache/
_CACHE_DIR = Path(__file__).resolve().parents[2] / "assets" / "emoji_cache"


def _cache_key(emoji: str) -> str:
    """用码点序列做文件名，避免文件系统对 emoji 字符的兼容性问题。"""
    return "-".join(f"{ord(c):x}" for c in emoji) + ".png"


class CachedTwitterEmojiSource(TwitterEmojiSource):
    """带本地磁盘缓存的 Twitter emoji source。"""

    def __init__(self) -> None:
        super().__init__()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_emoji(self, emoji: str, /) -> BytesIO | None:
        cache_path = _CACHE_DIR / _cache_key(emoji)

        if cache_path.exists():
            return BytesIO(cache_path.read_bytes())

        # 未命中：拉一次 CDN（父类内部已对 HTTPError 容错，网络异常仍可能抛出）
        result = super().get_emoji(emoji)
        if result is not None:
            data = result.getvalue()
            cache_path.write_bytes(data)
            return BytesIO(data)
        return None
