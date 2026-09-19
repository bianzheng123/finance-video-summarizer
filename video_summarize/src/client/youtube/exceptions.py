"""YouTube API 自定义异常类"""


class YouTubeError(Exception):
    """YouTube 基础异常"""
    pass


class YouTubeDownloadError(YouTubeError):
    """YouTube 视频下载失败"""
    pass


class YouTubeVideoNotFoundError(YouTubeError):
    """YouTube 视频不存在"""
    pass
