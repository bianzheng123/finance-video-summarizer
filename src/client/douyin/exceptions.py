"""抖音客户端自定义异常类"""


class DouyinError(Exception):
    """抖音基础异常"""
    pass


class DouyinDownloadError(DouyinError):
    """抖音视频下载失败"""
    pass


class DouyinVideoNotFoundError(DouyinError):
    """抖音视频不存在"""
    pass
