"""Bilibili API 统一客户端"""

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
import tempfile
from typing import Any

import yt_dlp

import requests
from bilibili_api import live, user, Credential
from bilibili_api.exceptions import ResponseCodeException
from bilibili_api import video_uploader
from bilibili_api import select_client, request_settings

from .credential_manager import CredentialManager
from .exceptions import (
    BilibiliAPIError,
    LiveRoomNotFoundError,
    VideoUploadError,
    SubtitleSubmitError,
    ArticleDraftError,
    SeasonError,
)

logger = logging.getLogger(__name__)

# B 站投稿默认分区：207 = 财经商业。所有视频都投到这个分区。
FINANCE_TID = 207


class _FixedVideoMeta(video_uploader.VideoMeta):
    """修正 add/v3 的字段类型 bug（与"精选评论"功能无关）：

    上游 VideoMeta.__dict__() 把 up_selection_reply 序列化成 int（1/0），而 B 站
    add/v3 现在严格要求它是 JSON bool，于是投稿报 21001 参数错误（2026-07 实测：
    同一 payload，该字段传 int 1 报 21001，传 bool False/True 均通过）。这里覆盖为
    输出真正的 bool，并把 web_os 对齐网页版的 3，只为让投稿本身不报 21001。

    注意：add/v3 里的 up_selection_reply **并不能真正开启"仅展示精选评论"**（实测
    小号评论仍可见）。真正的精选开关是发布后调 reply/subject/modify，见
    BilibiliClient.set_comment_selection。因此本类固定把它按 False 输出即可。"""

    def __dict__(self) -> dict:
        d = super().__dict__()
        d["up_selection_reply"] = bool(self.up_selection_reply)
        d["web_os"] = 3
        return d


# bilibili_api 默认走 curl_cffi 客户端，但 curl_cffi 0.14.0 连 upos 上传 CDN 时
# 会协商出 HTTP/2 并触发帧解析错误（curl: (16) / CURLE_HTTP2），导致大文件上传秒崩。
# 切到 httpx 客户端并强制 HTTP/1.1，从根上绕开该 bug。整库请求都走 get_client()，
# 故全局设置一次即可覆盖上传分片在内的全部请求。
select_client("httpx")
request_settings.set("http2", False)


class BilibiliClient:
    """Bilibili API 统一客户端

    进程内按 cookie_path 单例：同一份 cookie 只加载一次 Credential。避免
    模块级客户端、SubtitleRecognizer、各 BilibiliRecorder 各自重复初始化。
    """

    DEFAULT_COOKIE_PATH = Path(__file__).parent.parent.parent.parent / "config_local" / "cookie" / "bilibili.json"

    _INSTANCES: dict[str, "BilibiliClient"] = {}

    def __new__(cls, cookie_path: Path | None = None) -> "BilibiliClient":
        key = str(cookie_path or cls.DEFAULT_COOKIE_PATH)
        inst = cls._INSTANCES.get(key)
        if inst is None:
            inst = super().__new__(cls)
            cls._INSTANCES[key] = inst
        return inst

    def __init__(self, cookie_path: Path | None = None) -> None:
        """初始化客户端

        Args:
            cookie_path: Cookie 配置文件路径，默认为 config_local/cookie/bilibili.json
        """
        if getattr(self, "_initialized", False):
            return
        self.cookie_path = cookie_path or self.DEFAULT_COOKIE_PATH
        self.credential = CredentialManager.load_from_file(self.cookie_path)
        self._cookies = CredentialManager.get_cookie_dict(self.cookie_path)
        self._initialized = True
        logger.info("BilibiliClient 初始化成功")

    def download_video(
        self,
        url: str,
        output_dir: Path | None = None,
        format: str = "bestvideo+bestaudio/best",
        merge_output_format: str = "mp4",
        **kwargs: Any
    ) -> Path:
        """下载 B站视频（使用 cookie）

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

        cookies = self._load_cookies()
        cookie_file = self._write_netscape_cookies(cookies) if cookies else None

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
            **({"cookiefile": str(cookie_file)} if cookie_file else {}),
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
        finally:
            if cookie_file and cookie_file.exists():
                cookie_file.unlink()
    
    def _write_netscape_cookies(self, cookies: dict) -> Path:
        """将 cookie 字典写入 Netscape 格式临时文件供 yt-dlp 使用"""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        f.write("# Netscape HTTP Cookie File\n")
        for name, value in cookies.items():
            f.write(f".bilibili.com\tTRUE\t/\tFALSE\t0\t{name}\t{value}\n")
        f.close()
        return Path(f.name)

    def _load_cookies(self) -> dict:
        """加载 B站 cookie

        Returns:
            cookie 字典，如果配置文件不存在则返回空字典
        """
        cookie_path = Path(__file__).parent.parent.parent.parent / "config_local" / "cookie" / "bilibili.json"
        if cookie_path.exists():
            try:
                with open(cookie_path, "r", encoding="utf-8") as f:
                    cookies = json.load(f)
                logger.info("已从 %s 加载 cookie", cookie_path)
                return cookies if isinstance(cookies, dict) else {}
            except Exception as e:
                logger.warning("加载 cookie 失败: %s - %s", cookie_path, e)

        logger.warning("未找到 cookie 文件，将尝试无 cookie 下载")
        return {}
    
    async def check_live_status(
        self,
        room_id: int,
        max_retries: int = 3,
        retry_delays: list[int] = None
    ) -> tuple[bool, str]:
        """检查直播间状态

        Args:
            room_id: 直播间 ID
            max_retries: 最大重试次数
            retry_delays: 重试延迟列表（秒）

        Returns:
            (是否开播, 直播标题)

        Raises:
            LiveRoomNotFoundError: 直播间不存在
            BilibiliAPIError: API 调用失败
        """
        if retry_delays is None:
            retry_delays = [5, 15]

        for attempt in range(1, max_retries + 1):
            try:
                room = live.LiveRoom(room_display_id=room_id, credential=self.credential)
                info = await room.get_room_info()
                room_info = info.get("room_info", {})
                status = room_info.get("live_status", 0)
                title = room_info.get("title", "直播录像")
                return status == 1, title
            except Exception as e:
                if attempt < max_retries:
                    delay = retry_delays[attempt - 1] if attempt - 1 < len(retry_delays) else retry_delays[-1]
                    logger.warning(
                        "查询 room_id=%d 失败（第%d次），%ds 后重试: %s",
                        room_id, attempt, delay, e
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("查询 room_id=%d 失败，已重试 %d 次，放弃: %s", room_id, max_retries, e)
                    raise BilibiliAPIError(f"查询直播间状态失败: {e}")

    async def get_latest_video(
        self,
        uid: int,
        max_retries: int = 3,
        retry_delays: list[int] = None
    ) -> str | None:
        """获取 UP 主最新视频的 BVID

        Args:
            uid: UP 主 UID
            max_retries: 最大重试次数
            retry_delays: 重试延迟列表（秒）

        Returns:
            最新视频的 BVID，如果没有视频则返回 None

        Raises:
            BilibiliAPIError: API 调用失败
        """
        if retry_delays is None:
            retry_delays = [5, 15]

        for attempt in range(1, max_retries + 1):
            try:
                u = user.User(uid=uid, credential=self.credential)
                res = await u.get_videos(pn=1, ps=1)
                if res is None:
                    logger.warning("get_videos 返回 None，uid=%d（第%d次），可能是限流或 Cookie 过期", uid, attempt)
                    if attempt < max_retries:
                        delay = retry_delays[attempt - 1] if attempt - 1 < len(retry_delays) else retry_delays[-1]
                        await asyncio.sleep(delay)
                    continue
                videos = res.get("list", {}).get("vlist", [])
                if videos:
                    return videos[0]["bvid"]
                return None
            except Exception as e:
                if attempt < max_retries:
                    delay = retry_delays[attempt - 1] if attempt - 1 < len(retry_delays) else retry_delays[-1]
                    logger.warning(
                        "查询 uid=%d 失败（第%d次），%ds 后重试: %s",
                        uid, attempt, delay, e
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("查询 uid=%d 失败，已重试 %d 次，放弃: %s", uid, max_retries, e)
                    raise BilibiliAPIError(f"获取最新视频失败: {e}")

    @staticmethod
    def _build_upload_meta(
        tid: int,
        title: str,
        desc: str,
        cover_path: Path,
        tags: list[str],
        original: bool,
        watermark: bool,
        up_selection_reply: bool,
    ) -> dict:
        """构造投稿接口 add/v3 的 meta 字典。

        用 _FixedVideoMeta 生成——它已修正 up_selection_reply（精选评论）的类型：
        B 站现在严格要求该字段是 JSON bool，上游 VideoMeta 会误写成 int 而报 21001。

        cover 由 VideoUploader 单独上传后回填 URL，这里不塞封面路径。
        """
        return _FixedVideoMeta(
            tid=tid,
            title=title,
            desc=desc,
            cover=str(cover_path),
            tags=tags,
            original=original,
            watermark=watermark,
            up_selection_reply=up_selection_reply,
        ).__dict__()

    # 投稿被拒且重试无意义的错误码（账号 / 内容层面的确定性拒绝）。
    # 137001 账号封禁中；21070 投稿数超上限；21001 参数非法。
    # 不含网络类错误码（-504/-509 等）——那些仍走重试。
    _UPLOAD_FATAL_CODES = {137001, 21070, 21001}

    async def upload_video(
        self,
        video_path: Path,
        title: str,
        cover_path: Path,
        tid: int = FINANCE_TID,  # 财经商业
        desc: str = "直播录像自动上传",
        tags: list[str] = None,
        original: bool = True,
        watermark: bool = True,
        enable_comment_selection: bool = True,
    ) -> tuple[str | None, int | None]:
        """上传视频到 B站

        Args:
            video_path: 视频文件路径
            title: 视频标题
            cover_path: 封面图片路径
            tid: 分区 ID，默认 207 财经商业（所有视频都投财经分区）
            desc: 视频简介
            tags: 标签列表（最多 10 个）
            original: 是否原创
            watermark: 是否添加水印，默认开启
            enable_comment_selection: 上传成功后是否顺带开启"仅展示精选评论"，默认开启。
                     走 reply/subject/modify（见 set_comment_selection）。注意：视频若仍
                     在审核中、评论区未开放，会开启失败（12002），此时只告警不重试。
            可见范围：B 站投稿默认即"公开可见"，无需额外字段。

        Returns:
            (BVID, CID)，失败返回 (None, None)

        Raises:
            VideoUploadError: 视频上传失败
        """
        if tags is None:
            tags = ["直播录像", "自动上传"]

        if not video_path.exists():
            raise VideoUploadError(f"视频文件不存在: {video_path}")

        if not cover_path.exists():
            raise VideoUploadError(f"封面文件不存在: {cover_path}")

        try:
            VideoUploaderPage = video_uploader.VideoUploaderPage
            VideoUploader = video_uploader.VideoUploader

            # up_selection_reply 走 add/v3 并不能真正开精选（见 _FixedVideoMeta），
            # 固定按 False 投稿；真正的精选在上传成功后调 set_comment_selection。
            meta = self._build_upload_meta(
                tid=tid,
                title=title,
                desc=desc,
                cover_path=cover_path,
                tags=tags,
                original=original,
                watermark=watermark,
                up_selection_reply=False,
            )

            # B站上传 CDN 偶发 HTTP/2 框架错误（curl: (16)）/ 网络抖动，
            # 整段上传失败时重试若干次（每次重建 uploader，避免复用脏连接状态）。
            # 注意：uploader 的构造（VideoUploaderPage / VideoUploader）也可能触发
            # 网络请求并抛错，必须放进内层 try 才能被重试覆盖，否则会直接穿透到
            # 外层 except、零重试就失败。
            MAX_ATTEMPTS = 3
            last_exc: Exception | None = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                captured_cid = None
                try:
                    page = VideoUploaderPage(path=str(video_path), title=title)
                    uploader = VideoUploader(
                        pages=[page],
                        meta=meta,
                        credential=self.credential,
                        cover=str(cover_path),
                    )

                    @uploader.on("PRE_SUBMIT")
                    async def on_pre_submit(data: dict[str, Any]) -> None:
                        nonlocal captured_cid
                        videos = data.get("videos", [])
                        if videos:
                            captured_cid = videos[0]["cid"]
                            logger.info("捕获到 cid=%s", captured_cid)

                    result = await uploader.start()
                    bvid = result.get("bvid") if isinstance(result, dict) else None

                    if not bvid:
                        logger.error("视频上传失败: %s", result)
                        raise VideoUploadError(f"视频上传失败: {result}")

                    logger.info("视频上传成功，bvid=%s", bvid)

                    # 上传成功后顺带开启"仅展示精选评论"。视频若仍在审核、评论区未开放
                    # 会失败（12002），set_comment_selection 内部只告警不抛异常、不重试。
                    if enable_comment_selection:
                        from bilibili_api import bvid2aid
                        self.set_comment_selection(bvid2aid(bvid), self.COMMENT_SELECTION_ON)

                    return bvid, captured_cid
                except VideoUploadError:
                    raise
                except ResponseCodeException as e:
                    # 账号 / 内容层面的确定性拒绝，重试必然同样失败。而每次重试都要把整个
                    # 视频重新推一遍 CDN（GB 级、十来分钟），白烧流量还拖后面的总结交付，
                    # 故立即放弃。
                    if e.code in self._UPLOAD_FATAL_CODES:
                        logger.error("视频上传被拒绝（code=%s，确定性失败，不重试）: %s", e.code, e.msg)
                        raise VideoUploadError(f"视频上传被拒绝（code={e.code}）: {e.msg}")
                    last_exc = e
                    if attempt < MAX_ATTEMPTS:
                        backoff = 10 * attempt
                        logger.warning("视频上传第 %d/%d 次失败: %s，%d 秒后重试",
                                       attempt, MAX_ATTEMPTS, e, backoff)
                        await asyncio.sleep(backoff)
                    else:
                        logger.error("视频上传异常（已重试 %d 次）: %s", MAX_ATTEMPTS, e)
                except Exception as e:
                    last_exc = e
                    if attempt < MAX_ATTEMPTS:
                        backoff = 10 * attempt
                        logger.warning("视频上传第 %d/%d 次失败: %s，%d 秒后重试",
                                       attempt, MAX_ATTEMPTS, e, backoff)
                        await asyncio.sleep(backoff)
                    else:
                        logger.error("视频上传异常（已重试 %d 次）: %s", MAX_ATTEMPTS, e)

            raise VideoUploadError(f"视频上传异常（已重试 {MAX_ATTEMPTS} 次）: {last_exc}")

        except VideoUploadError:
            raise
        except Exception as e:
            logger.error("视频上传异常: %s", e)
            raise VideoUploadError(f"视频上传异常: {e}")

    async def get_current_user_info(self) -> dict | None:
        """获取当前登录用户信息

        Returns:
            用户信息字典，包含 'name'（昵称）等字段；失败返回 None
        """
        try:
            # 从 cookie 中获取用户 UID
            uid = int(self._cookies.get("dedeuserid", 0))
            if not uid:
                logger.warning("无法从 cookie 获取用户 UID")
                return None

            u = user.User(uid=uid, credential=self.credential)
            info = await u.get_user_info()
            if info and "name" in info:
                logger.info("获取当前用户信息成功: %s", info["name"])
                return info
            
            logger.warning("获取用户信息失败")
            return None
        
        except Exception as e:
            logger.error("获取当前用户信息异常: %s", e)
            return None

    async def submit_subtitle(
        self,
        bvid: str,
        subtitle_path: Path,
        captured_cid: int
    ) -> bool:
        """提交字幕到视频

        Args:
            bvid: 视频 BVID
            subtitle_path: SRT 字幕文件路径
            captured_cid: 视频 CID

        Returns:
            是否提交成功

        Raises:
            SubtitleSubmitError: 字幕提交失败
        """
        if not subtitle_path.exists():
            logger.warning("字幕文件不存在，跳过字幕提交: %s", subtitle_path)
            return False

        try:
            csrf = self._cookies["BILI_JCT"]
            cookie_str = (
                f"SESSDATA={self._cookies['SESSDATA']}; "
                f"bili_jct={csrf}; "
                f"BUVID3={self._cookies['BUVID3']}; "
                f"DedeUserID={self._cookies['dedeuserid']}"
            )
            headers = {
                "Cookie": cookie_str,
                "Referer": "https://member.bilibili.com/",
                "Origin": "https://member.bilibili.com",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }

            archive_resp = requests.get(
                f"https://member.bilibili.com/x/vupre/web/archive/view?bvid={bvid}",
                headers=headers,
            )
            archive_data = archive_resp.json()
            if archive_data.get("code") != 0:
                logger.error("获取视频信息失败: %s", archive_data)
                raise SubtitleSubmitError(f"获取视频信息失败: {archive_data}")

            aid = archive_data["data"]["videos"][0]["aid"]
            logger.info("获取到 aid=%s", aid)

            ts = int(time.time() * 1000)
            with open(subtitle_path, "rb") as f:
                upload_resp = requests.post(
                    f"https://api.bilibili.com/x/upload/web/image?t={ts}&csrf={csrf}",
                    headers=headers,
                    files={"file": (subtitle_path.name, f, "application/x-subrip")},
                    data={"bucket": "subtitle", "csrf": csrf, "content_type": "application/x-subrip"},
                )

            upload_data = upload_resp.json()
            if upload_data.get("code") != 0:
                logger.error("SRT 上传失败: %s", upload_data)
                raise SubtitleSubmitError(f"SRT 上传失败: {upload_data}")

            location = upload_data["data"]["location"]
            logger.info("SRT 上传成功: %s", location)

            ts = int(time.time() * 1000)
            save_resp = requests.post(
                f"https://api.bilibili.com/x/v2/dm/subtitle/draft/preSave?t={ts}&csrf={csrf}",
                headers=headers,
                data={
                    "oid": str(captured_cid),
                    "type": "1",
                    "files": json.dumps([{"url": location, "lan": "zh", "subtitle_id": 0}]),
                    "aid": str(aid),
                    "csrf": csrf,
                },
            )

            save_data = save_resp.json()
            if save_data.get("code") != 0:
                logger.error("字幕提交失败: %s", save_data)
                raise SubtitleSubmitError(f"字幕提交失败: {save_data}")

            logger.info("字幕提交成功！")
            return True

        except SubtitleSubmitError:
            raise
        except Exception as e:
            logger.error("字幕提交异常: %s", e)
            raise SubtitleSubmitError(f"字幕提交异常: {e}")

    # 评论精选 action 取值（reply/subject/modify）：
    COMMENT_SELECTION_ON = 1   # 开启评论精选（"仅展示精选评论"）
    COMMENT_SELECTION_OFF = 2  # 关闭评论精选
    # 注：action=3 关闭评论区 / action=4 开放评论区，危险，本类不封装。

    # 可通过重试恢复的临时错误码（网络抖动 / 限流 / 网关超时）；其余码视为确定性失败，
    # 重试无意义（如 12002 评论区未开放、-111 csrf 失效）。
    _SELECTION_RETRYABLE_CODES = {-504, -509, -412}

    def set_comment_selection(
        self,
        aid: int,
        action: int = COMMENT_SELECTION_ON,
        max_attempts: int = 3,
    ) -> bool:
        """开启/关闭视频的"评论精选"（网页版"仅展示精选评论"）。

        走 reply/subject/modify（表单：type=1&oid=<aid>&action=<action>&csrf）。
        action=1 开启，2 关闭（见 COMMENT_SELECTION_ON/OFF）。

        网络抖动或限流（-509 等）会重试最多 max_attempts 次（指数退避）；确定性失败
        （如 12002 评论区未开放、csrf 失效）不重试。本方法不抛异常，失败返回 False，
        以免阻断上传主流程。

        Returns:
            True 表示成功；False 表示失败（重试耗尽或确定性失败）。
        """
        csrf = self._cookies["BILI_JCT"]
        headers = {
            "Cookie": (
                f"SESSDATA={self._cookies['SESSDATA']}; bili_jct={csrf}; "
                f"BUVID3={self._cookies['BUVID3']}; DedeUserID={self._cookies['dedeuserid']}"
            ),
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.bilibili.com/",
            "Origin": "https://www.bilibili.com",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        body = f"type=1&oid={aid}&action={action}&csrf={csrf}"

        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    "https://api.bilibili.com/x/v2/reply/subject/modify",
                    headers=headers,
                    data=body,
                    timeout=15,
                )
                data = resp.json()
            except Exception as e:
                # 网络层异常（连接失败 / 超时 / JSON 解析失败）——可重试
                if attempt < max_attempts:
                    backoff = 2 * attempt
                    logger.warning("设置评论精选网络异常（aid=%s，第 %d/%d 次）：%s，%ds 后重试",
                                   aid, attempt, max_attempts, e, backoff)
                    time.sleep(backoff)
                    continue
                logger.warning("设置评论精选异常（aid=%s，已重试 %d 次）：%s", aid, max_attempts, e)
                return False

            code = data.get("code")
            if code == 0:
                toast = (data.get("data") or {}).get("action_toast", "")
                logger.info("评论精选设置成功（aid=%s）: %s", aid, toast)
                return True

            # 应用层错误码：仅临时性错误重试，确定性错误直接放弃
            if code in self._SELECTION_RETRYABLE_CODES and attempt < max_attempts:
                backoff = 2 * attempt
                logger.warning("设置评论精选临时失败（aid=%s，code=%s，第 %d/%d 次），%ds 后重试",
                               aid, code, attempt, max_attempts, backoff)
                time.sleep(backoff)
                continue

            logger.warning("设置评论精选失败（aid=%s）: %s", aid, data)
            return False

        return False

    # ── 专栏图文草稿 ───────────────────────────────────────────────────────────
    # 以下接口为社区逆向整理（非官方文档），字段名可能随 B 站调整而失效。
    # 首次跑通前以日志打印原始响应，便于核对。
    _MEMBER_BASE = "https://api.bilibili.com"

    def _member_headers(self) -> dict:
        """创作中心接口统一的 Cookie / Referer 头（与 submit_subtitle 同款）。"""
        csrf = self._cookies["BILI_JCT"]
        cookie_str = (
            f"SESSDATA={self._cookies['SESSDATA']}; "
            f"bili_jct={csrf}; "
            f"BUVID3={self._cookies['BUVID3']}; "
            f"DedeUserID={self._cookies['dedeuserid']}"
        )
        return {
            "Cookie": cookie_str,
            "Referer": "https://member.bilibili.com/read/editor/",
            "Origin": "https://member.bilibili.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

    def upload_article_image(self, image_path: Path) -> str:
        """上传一张图片到 B 站图床，返回图床 URL（用于嵌入专栏正文）。

        走创作中心专栏图片上传接口 article/creative/article/upcover。
        """
        if not image_path.exists():
            raise ArticleDraftError(f"图片不存在: {image_path}")

        csrf = self._cookies["BILI_JCT"]
        try:
            with open(image_path, "rb") as f:
                resp = requests.post(
                    f"{self._MEMBER_BASE}/x/article/creative/article/upcover",
                    headers=self._member_headers(),
                    files={"binary": (image_path.name, f, "image/png")},
                    data={"csrf": csrf},
                )
            data = resp.json()
        except Exception as e:
            raise ArticleDraftError(f"上传专栏图片异常: {e}")

        logger.info("图片上传响应: %s", data)
        if data.get("code") != 0:
            raise ArticleDraftError(f"上传专栏图片失败: {data}")

        url = data["data"].get("url") or data["data"].get("image_url")
        if not url:
            raise ArticleDraftError(f"上传响应中未找到图片 URL: {data}")
        # B 站常返回 //i0.hdslb.com/... 形式，补全协议
        if url.startswith("//"):
            url = "https:" + url
        logger.info("专栏图片上传成功: %s", url)
        return url

    def list_article_collections(self) -> list[dict]:
        """拉取当前账号的专栏文集列表，返回 [{"id": int, "name": str, ...}, ...]。

        走创作中心 list/all 接口；用于按博主名匹配出对应文集的 list_id。
        """
        try:
            resp = requests.get(
                f"{self._MEMBER_BASE}/x/article/creative/list/all",
                headers=self._member_headers(),
            )
            data = resp.json()
        except Exception as e:
            raise ArticleDraftError(f"拉取专栏文集列表异常: {e}")

        logger.info("文集列表响应: %s", data)
        if data.get("code") != 0:
            raise ArticleDraftError(f"拉取专栏文集列表失败: {data}")

        lists = (data.get("data") or {}).get("lists") or []
        return lists

    def find_collection_id(self, name: str) -> int:
        """按文集名匹配 list_id（先精确、后包含匹配）；找不到返回 0。"""
        lists = self.list_article_collections()
        for item in lists:
            if item.get("name") == name:
                return int(item.get("id", 0))
        for item in lists:
            if name and name in str(item.get("name", "")):
                return int(item.get("id", 0))
        logger.warning("未在文集列表中找到「%s」，将不归入文集（list_id=0）", name)
        return 0

    def create_collection(self, name: str, summary: str = "", image_url: str = "") -> int:
        """新建一个专栏文集，返回新文集的 list_id。

        走创作中心 list/add 接口；字段 name / summary / image_url / csrf。
        """
        csrf = self._cookies["BILI_JCT"]
        payload = {"name": name, "summary": summary, "image_url": image_url, "csrf": csrf}
        try:
            resp = requests.post(
                f"{self._MEMBER_BASE}/x/article/creative/list/add",
                headers=self._member_headers(),
                data=payload,
            )
            data = resp.json()
        except Exception as e:
            raise ArticleDraftError(f"新建专栏文集异常: {e}")

        logger.info("新建文集响应: %s", data)
        if data.get("code") != 0:
            raise ArticleDraftError(f"新建专栏文集失败: {data}")

        new_id = (data.get("data") or {}).get("id")
        if not new_id:
            raise ArticleDraftError(f"新建文集响应中未找到 id: {data}")
        logger.info("专栏文集新建成功，「%s」list_id=%s", name, new_id)
        return int(new_id)

    def save_article_draft(
        self,
        title: str,
        content_html: str,
        list_id: int = 0,
        category: int = 0,
        summary: str = "",
        cover_url: str = "",
        aid: int = 0,
        comment_open: bool = True,
        comment_selected: bool = False,
        cover_anchor_x: int = 0,
        cover_anchor_y: int = 0,
    ) -> int:
        """保存专栏图文草稿，返回草稿 aid。

        Args:
            title: 标题
            content_html: 正文 HTML（本地图片需先经 upload_article_image 替换成图床 URL）
            list_id: 文集 ID，0 表示不归入文集
            category: 分类 ID，0 为默认
            summary: 摘要，留空则 B 站自动截取
            cover_url: 封面图床 URL（先经 upload_article_image 上传得到），留空则无封面
            aid: 已有草稿 aid，传 0 表示新建
            comment_open: True=打开评论开关
            comment_selected: True=开启精选评论（默认 False=关闭精选评论）
            cover_anchor_x: 封面裁剪起点 x 像素偏移（逆向字段，可能不生效）
            cover_anchor_y: 封面裁剪起点 y 像素偏移（逆向字段，可能不生效）

        Returns:
            草稿 aid

        说明：可见范围（充电专属等）不在此设置，留到草稿箱手动设置；评论 / 创作声明 /
        封面裁剪锚点等字段为逆向所得，草稿接口返回 code:0 仅代表无报错，实际生效需肉眼核对。
        """
        csrf = self._cookies["BILI_JCT"]
        payload = {
            "title": title,
            "content": content_html,
            "category": category,
            "list_id": list_id,
            "tid": 4,            # 专栏封面布局：4=单图封面
            "reprint": 0,        # 0=原创
            "original": 1,       # 原创声明
            "summary": summary,
            "words": 0,
            "spoiler": 0,        # 非剧透
            "csrf": csrf,
            # ── 评论 / 发布 ──
            # 可见范围（充电专属 / 仅充电可见等）不在此设置，留到草稿箱手动勾选
            "up_reply_closed": 0 if comment_open else 1,   # 0=评论开
            "comment_selected": 1 if comment_selected else 0,  # 0=关闭精选评论
            "timer_pub_time": 0,   # 0=不定时发布
            "dynamic_intro": "",   # 不附加动态附言
            "media_id": 0,         # 不关联付费内容
            "top_video_bvid": "",  # 不置顶视频
        }
        if cover_url:
            # 专栏封面：origin_image_urls 为封面图列表，image_urls 同步带上
            payload["banner_url"] = cover_url
            payload["origin_image_urls"] = cover_url
            payload["image_urls"] = cover_url
            # 封面裁剪锚点（逆向字段，B 站官方接口未公开，可能不生效，留作可调参数）
            if cover_anchor_x or cover_anchor_y:
                payload["cover_anchor_x"] = cover_anchor_x
                payload["cover_anchor_y"] = cover_anchor_y
        if aid:
            payload["aid"] = aid

        try:
            resp = requests.post(
                f"{self._MEMBER_BASE}/x/article/creative/draft/addupdate",
                headers=self._member_headers(),
                data=payload,
            )
            data = resp.json()
        except Exception as e:
            raise ArticleDraftError(f"保存专栏草稿异常: {e}")

        logger.info("保存草稿响应: %s", data)
        if data.get("code") != 0:
            raise ArticleDraftError(f"保存专栏草稿失败: {data}")

        new_aid = data["data"].get("aid")
        if not new_aid:
            raise ArticleDraftError(f"保存草稿响应中未找到 aid: {data}")
        logger.info("专栏草稿保存成功，aid=%s", new_aid)
        return new_aid

    # ── 视频合集（season）─────────────────────────────────────────────────────
    # 创作中心「合集管理」接口（社区逆向，非官方文档）。用于把每期直播录像按博主名
    # 归入同名合集：如「钱博士直播录像」。已实测跑通：查列表 / 建合集 / 传封面 / 加视频。
    _SEASON_BASE = "https://member.bilibili.com/x2/creative/web"

    def list_seasons(self) -> list[dict]:
        """拉取当前账号的全部合集，返回 [{season_id, title, section_id}, ...]。

        section_id 取该合集第一个小节（新建合集会自动生成一个「正片」小节）；
        无小节时为 0。走 seasons 列表接口。
        """
        try:
            resp = requests.get(
                f"{self._SEASON_BASE}/seasons",
                headers=self._member_headers(),
                params={"pn": 1, "ps": 50},
            )
            data = resp.json()
        except Exception as e:
            raise SeasonError(f"拉取合集列表异常: {e}")

        if data.get("code") != 0:
            raise SeasonError(f"拉取合集列表失败: {data}")

        result = []
        for item in (data.get("data") or {}).get("seasons") or []:
            season = item.get("season", {})
            sections = (item.get("sections") or {}).get("sections") or []
            result.append({
                "season_id": season.get("id"),
                "title": season.get("title", ""),
                "section_id": sections[0].get("id") if sections else 0,
            })
        return result

    def find_season(self, title: str) -> dict | None:
        """按合集标题匹配（先精确、后包含）；找不到返回 None。"""
        seasons = self.list_seasons()
        for s in seasons:
            if s.get("title") == title:
                return s
        for s in seasons:
            if title and title in str(s.get("title", "")):
                return s
        return None

    def _get_season_section_id(self, season_id: int) -> int:
        """查合集详情，返回其第一个小节 ID（新建合集自动生成的「正片」小节）。"""
        try:
            resp = requests.get(
                f"{self._SEASON_BASE}/season",
                headers=self._member_headers(),
                params={"id": season_id},
            )
            data = resp.json()
        except Exception as e:
            raise SeasonError(f"查询合集详情异常: {e}")

        if data.get("code") != 0:
            raise SeasonError(f"查询合集详情失败: {data}")

        sections = ((data.get("data") or {}).get("sections") or {}).get("sections") or []
        if not sections:
            raise SeasonError(f"合集 {season_id} 无可用小节: {data}")
        return sections[0].get("id")

    def _upload_season_cover(self, cover_path: Path) -> str:
        """上传合集封面到 B 站，返回 archive.biliimg.com 封面 URL。

        走投稿封面接口 x/vu/web/cover/up，图片需 base64 编码。
        """
        if not cover_path.exists():
            raise SeasonError(f"合集封面不存在: {cover_path}")

        csrf = self._cookies["BILI_JCT"]
        b64 = "data:image/jpeg;base64," + base64.b64encode(cover_path.read_bytes()).decode()
        try:
            resp = requests.post(
                "https://member.bilibili.com/x/vu/web/cover/up",
                headers=self._member_headers(),
                data={"cover": b64, "csrf": csrf},
            )
            data = resp.json()
        except Exception as e:
            raise SeasonError(f"上传合集封面异常: {e}")

        if data.get("code") != 0:
            raise SeasonError(f"上传合集封面失败: {data}")
        url = (data.get("data") or {}).get("url")
        if not url:
            raise SeasonError(f"上传封面响应中未找到 url: {data}")
        logger.info("合集封面上传成功: %s", url)
        return url

    def create_season(self, title: str, cover_path: Path, desc: str = "") -> int:
        """新建合集，返回其自动生成的「正片」小节 section_id（后续加视频用）。

        注：合集有人工审核，但不影响立即往里加视频。
        """
        cover_url = self._upload_season_cover(cover_path)
        csrf = self._cookies["BILI_JCT"]
        try:
            resp = requests.post(
                f"{self._SEASON_BASE}/season/add",
                headers=self._member_headers(),
                data={"title": title, "desc": desc, "cover": cover_url,
                      "season_price": 0, "csrf": csrf},
            )
            data = resp.json()
        except Exception as e:
            raise SeasonError(f"新建合集异常: {e}")

        if data.get("code") != 0:
            raise SeasonError(f"新建合集失败: {data}")

        season_id = data.get("data")
        if isinstance(season_id, dict):
            season_id = season_id.get("id")
        if not season_id:
            raise SeasonError(f"新建合集响应中未找到 season_id: {data}")
        logger.info("合集《%s》新建成功，season_id=%s", title, season_id)
        return self._get_season_section_id(season_id)

    def add_video_to_season(self, section_id: int, aid: int, cid: int, title: str) -> bool:
        """把视频（aid/cid）作为一集加入合集小节。"""
        csrf = self._cookies["BILI_JCT"]
        body = {
            "sectionId": section_id,
            "episodes": [{"title": title, "aid": aid, "cid": cid, "charging_pay": 0}],
        }
        headers = {**self._member_headers(), "Content-Type": "application/json"}
        try:
            resp = requests.post(
                f"{self._SEASON_BASE}/season/section/episodes/add",
                headers=headers,
                params={"csrf": csrf, "t": int(time.time() * 1000)},
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            )
            data = resp.json()
        except Exception as e:
            raise SeasonError(f"加入合集异常: {e}")

        if data.get("code") != 0:
            raise SeasonError(f"加入合集失败: {data}")
        logger.info("视频 aid=%s 已加入合集小节 %s", aid, section_id)
        return True

    def get_or_create_season(self, title: str, cover_path: Path, desc: str = "") -> int:
        """按标题拿到合集小节 section_id：存在同名合集则复用，否则新建。

        返回可直接用于 add_video_to_season 的 section_id。
        """
        existing = self.find_season(title)
        if existing and existing.get("section_id"):
            logger.info("命中已有合集《%s》，section_id=%s", title, existing["section_id"])
            return existing["section_id"]
        if existing and not existing.get("section_id"):
            # 合集存在但无小节，取详情兜底
            return self._get_season_section_id(existing["season_id"])
        logger.info("未找到合集《%s》，新建", title)
        return self.create_season(title, cover_path, desc)
