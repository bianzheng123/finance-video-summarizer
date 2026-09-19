"""Bilibili API 自定义异常类"""


class BilibiliAPIError(Exception):
    """Bilibili API 基础异常"""
    pass


class LiveRoomNotFoundError(BilibiliAPIError):
    """直播间不存在"""
    pass


class VideoUploadError(BilibiliAPIError):
    """视频上传失败"""
    pass


class SubtitleSubmitError(BilibiliAPIError):
    """字幕提交失败"""
    pass


class CredentialLoadError(BilibiliAPIError):
    """凭证加载失败"""
    pass


class APIRateLimitError(BilibiliAPIError):
    """API 限流错误"""
    pass


class ArticleDraftError(BilibiliAPIError):
    """专栏草稿保存失败"""
    pass


class SeasonError(BilibiliAPIError):
    """合集（season）操作失败"""
    pass
