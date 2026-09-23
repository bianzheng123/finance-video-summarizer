"""渲染模块：提供多种格式的总结渲染功能。"""

from .rendering_base import BaseRenderer
from .rendering_summary import SummaryRenderer
from .rendering_gongzhonghao import GongzhonghaoRenderer

__all__ = [
    "BaseRenderer",
    "SummaryRenderer",
    "GongzhonghaoRenderer",
]
