"""基础渲染类，提供通用的渲染逻辑供子类继承。"""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable


_JARGON_PREFIXES = ("个股黑话：", "板块黑话：")


class BaseRenderer(ABC):
    """渲染器基类，定义统一的接口规范。"""

    def __init__(self, unresolved_jargons: dict[str, str] | None = None) -> None:
        self._unresolved_jargons: dict[str, str] = dict(unresolved_jargons or {})

    def decorate_jargon(self, name: str) -> str:
        """对未解析黑话加 `个股黑话：xxx` / `板块黑话：xxx` 前缀；其他名字原样返回。"""
        if not isinstance(name, str) or not name:
            return name
        if name.startswith(_JARGON_PREFIXES):
            return name
        jargon_type = self._unresolved_jargons.get(name)
        if not jargon_type:
            return name
        return f"{jargon_type}：{name}"

    def decorate_topic_label(self, label: str) -> str:
        """对带时长后缀的主题标签 `主题 (XX分XX秒)` 仅装饰主题部分；无后缀时整体装饰。"""
        if not isinstance(label, str) or not label:
            return label
        m = re.match(r"^(.+?)\s+(\(.*\))\s*$", label)
        if m:
            return f"{self.decorate_jargon(m.group(1))} {m.group(2)}"
        return self.decorate_jargon(label)

    @staticmethod
    def has_nested_dicts(d: dict) -> bool:
        """检查 dict 的值中是否有嵌套 dict。"""
        return any(isinstance(v, dict) for k, v in d.items())

    @staticmethod
    def has_complex_items(items: list) -> bool:
        """检查列表项是否包含嵌套 dict 值或引用字段。

        返回 True 的条件（任一满足即可）：
        - 列表项含有嵌套 dict 值（如"短线"、"中长线"子结构）
        - 列表项含有"引用"字段（需要展开渲染引用块而非作为表格列）
        用于区分简单列表（适合表格渲染）和复杂列表（需要展开渲染）。
        """
        for item in items:
            if not isinstance(item, dict):
                continue
            if "引用" in item:
                return True
            for k, v in item.items():
                if isinstance(v, dict):
                    return True
        return False

    @staticmethod
    def heading_link(text: str) -> str:
        """将标题文本转换为锚点链接格式。"""
        slug = text.lower()
        slug = re.sub(r'[^\w一-鿿\- ]', '', slug)
        slug = slug.replace(' ', '-')
        return slug

    @abstractmethod
    def render_header(self, author: str | None, title: str | None, bvid: str | None) -> list[str]:
        """渲染头部信息（视频来源、微信号等）。子类必须实现。"""
        pass

    @abstractmethod
    def render_time_references(self, time_ranges: list) -> list[str]:
        """渲染引用时间列表。子类必须实现。"""
        pass

    @abstractmethod
    def render_citations(self, citations: list) -> str:
        """渲染引用块。子类必须实现。"""
        pass

    @abstractmethod
    def render_section_title(self, title: str, is_first: bool = False) -> str:
        """渲染一级标题。子类必须实现。"""
        pass

    @abstractmethod
    def render_subsection_title(self, title: str) -> str:
        """渲染二级标题。子类必须实现。"""
        pass

    @abstractmethod
    def render_dict_fields(self, value_dict: dict, cit_fn: Callable[[list], str]) -> list[str]:
        """渲染带引用的字典字段。子类必须实现。"""
        pass

    @abstractmethod
    def render_complex_list(self, items: list, cit_fn: Callable[[list], str]) -> list[str]:
        """渲染包含嵌套 dict 的复杂列表项。子类必须实现。"""
        pass

    @abstractmethod
    def render_object_list(self, items: list, column_order: list[str] | None = None) -> list[str]:
        """将纯对象数组渲染为列表或表格。子类必须实现。"""
        pass
