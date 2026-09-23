"""时间处理工具函数集合。"""


def fmt_time(seconds: float | str) -> str:
    """将秒数格式化为 HH:MM:SS 字符串。

    Args:
        seconds: 秒数（支持浮点数或空字符串）

    Returns:
        格式化后的时间字符串，格式为 HH:MM:SS
    """
    s = float(seconds) if seconds != "" else 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02.0f}"


def hms_to_seconds(hms: str) -> int:
    """将 HH:MM:SS 格式的时间字符串转换为秒数。

    支持两种格式：
    - 完整格式：HH:MM:SS
    - 简短格式：MM:SS（自动补零为 00:MM:SS）

    Args:
        hms: 时间字符串

    Returns:
        对应的秒数（整数）
    """
    parts = hms.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(float(parts[1]))
    return 0
