"""抖音直播录制器 — streamget 获取流地址 + ffmpeg 拉流。"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from streamget import DouyinLiveStream

from ..base import COOKIE_DIR, LiveStreamRecorder, load_cookie_string

logger = logging.getLogger(__name__)

# streamget 的 async_req 会把网络层异常「吞」成字符串响应体返回（见其 async_http.py:
# except 分支 return str(e)），随后 json.loads 把这段错误文字当 JSON 解析必然抛
# JSONDecodeError（如 "Expecting value: line 1 column 2 (char 1)"），真实原因藏在
# e.doc 里。下列标记用于从还原出的文字里识别「网络瞬断」，这类错误值得像 bilibili
# 那样重试，而不是当成一次查询失败——否则一次 DNS 抖动就会触发服务重启。
_TRANSIENT_NET_MARKERS = (
    "Temporary failure in name resolution",
    "Name or service not known",
    "Connection refused",
    "Connection reset",
    "Connection aborted",
    "All connection attempts failed",
    "timed out",
    "Timeout",
    "SSL",
    "EOF occurred",
    "Errno",
)

# 与 bilibili check_live_status 保持一致的重试策略
_MAX_RETRIES = 3
_RETRY_DELAYS = (5, 15)


def _unmask_streamget_error(e: Exception) -> str:
    """还原 streamget 吞掉的真实错误信息。

    streamget 把网络异常塞进响应体后 json.loads 失败，真实原因（DNS/连接/超时）
    藏在 JSONDecodeError.doc 里；其它异常按原样返回 str(e)。
    """
    if isinstance(e, json.JSONDecodeError):
        return e.doc or str(e)
    return str(e)


class DouyinRecorder(LiveStreamRecorder):
    """抖音直播录制器 — 使用 streamget 获取 flv_url + ffmpeg 拉流。"""

    platform = "douyin"

    def __init__(self, room_id: int | str) -> None:
        self.live_url = f"https://live.douyin.com/{room_id}"
        # 默认加载 config_local/cookie/douyin.json 作为登录态 cookie（含 ttwid）
        # 日志由 load_cookie_string 按文件缓存统一打印，避免多主播时重复刷屏
        self.cookies = load_cookie_string(COOKIE_DIR / "douyin.json")
        self._stream = None
        self._last_flv_url: str | None = None

    def _get_stream(self) -> DouyinLiveStream:
        if self._stream is None:
            self._stream = DouyinLiveStream(cookies=self.cookies)
        return self._stream

    async def _fetch_raw_with_retry(self) -> dict[str, Any]:
        """取原始响应，网络瞬断按 bilibili 同款策略（[5,15]s）重试。

        streamget 把网络异常吞成响应体导致 json.loads 抛 JSONDecodeError，此处先用
        _unmask_streamget_error 还原真实原因，再据此分类：网络瞬断 → 重试；风控/cookie
        失效等 → 直接抛出（重试无益）。
        """
        stream = self._get_stream()
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await stream.fetch_web_stream_data(self.live_url, process_data=False)
            except Exception as e:
                err = _unmask_streamget_error(e)
                transient = any(m in err for m in _TRANSIENT_NET_MARKERS)
                if transient and attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAYS[attempt - 1] if attempt - 1 < len(_RETRY_DELAYS) else _RETRY_DELAYS[-1]
                    logger.warning(
                        "[douyin] 查询直播状态失败（第%d次，网络瞬断），%ds 后重试: %s",
                        attempt, delay, err,
                    )
                    await asyncio.sleep(delay)
                    continue
                # 不可重试或已达上限 → 分类记录后抛出（用还原后的 err，不再是 "未知原因"）
                if "risk control" in err:
                    logger.error("[douyin] 风控拦截（请求被屏蔽），可能需要更新 cookie: %s", err)
                elif any(kw in err for kw in ("直播已结束", "直播间不存在", "Fetch failed")):
                    logger.error("[douyin] API 返回异常提示（很可能是 cookie 失效或风控软拦截）: %s", err)
                elif transient:
                    logger.error("[douyin] 查询直播状态失败（网络异常，已重试 %d 次）: %s", _MAX_RETRIES, err)
                else:
                    logger.error("[douyin] 查询直播状态失败: %s", err)
                # 抛出还原后的原因，避免上层再看到令人误解的 JSONDecodeError
                raise RuntimeError(err) from e

    async def is_living(self) -> tuple[bool, str]:
        stream = self._get_stream()
        # 先用 process_data=False 取原始响应自行判定开播状态：
        # 抖音 web API 在主播未开播时返回的 data['data'] 是空数组且无 prompts 字段，
        # 此时 streamget 的 _get_web_stream_data 会误抛 "VR live is not supported"。
        # 我们直接看原始响应里 data['data'] 是否为空来区分"未开播"与真正的异常，
        # 避免把正常的未开播当成查询失败。
        raw = await self._fetch_raw_with_retry()

        room_payload = raw.get("data", {}) if isinstance(raw, dict) else {}
        inner = room_payload.get("data")
        anchor_name = room_payload.get("user", {}).get("nickname", "") or "直播中"

        # 未开播：inner 为空数组。直接返回，不再走 streamget 的处理分支（它会误抛 VR live 异常）。
        if not inner:
            logger.debug("[douyin] %s 未开播或直播已结束", anchor_name)
            return False, anchor_name

        room_data = inner[0]
        status = room_data.get("status", 4)
        title = room_data.get("title", "") or anchor_name
        is_live = status == 2
        if is_live:
            # 已开播 → 重新走 process_data=True 拿到带流地址的结构，预取流 URL
            data = await stream.fetch_web_stream_data(self.live_url)
            stream_data = await stream.fetch_stream_url(data, video_quality="HD")
            self._last_flv_url = stream_data.flv_url or stream_data.m3u8_url
        return is_live, title

    async def record(self, output_path: Path) -> None:
        # 如果 is_living 已经预取了 URL 就直接用，否则重新获取
        flv_url = self._last_flv_url
        if not flv_url:
            stream = self._get_stream()
            data = await stream.fetch_web_stream_data(self.live_url)
            stream_data = await stream.fetch_stream_url(data, video_quality="HD")
            flv_url = stream_data.flv_url or stream_data.m3u8_url
        if not flv_url:
            raise RuntimeError("[douyin] 无法获取直播流 URL")

        self._last_flv_url = None
        logger.info("[douyin] 开始录制: %s → %s", flv_url[:80], output_path)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._run_ffmpeg, flv_url, output_path)

    @staticmethod
    def _run_ffmpeg(stream_url: str, output_path: Path) -> None:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-headers", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36\r\n",
            "-i", stream_url,
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
            "-movflags", "frag_keyframe+empty_moov",
            str(output_path),
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True)
        if result.returncode != 0:
            logger.warning("[douyin] ffmpeg 退出码: %d（直播可能已结束）", result.returncode)
        else:
            logger.info("[douyin] 录制完成")
