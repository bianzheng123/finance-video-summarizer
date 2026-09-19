"""基础渲染类，提供 summary.md 的 Markdown 渲染实现。"""

from collections.abc import Callable

from .rendering_base import BaseRenderer


class SummaryRenderer(BaseRenderer):
    """基础 Markdown 渲染器（summary.md 使用）。"""

    def render_header(self, author: str | None, title: str | None, bvid: str | None) -> list[str]:
        """渲染头部信息（视频来源、微信号等）。"""
        parts = []
        video_url = f"https://www.bilibili.com/video/{bvid}" if bvid else ""
        parts.append(f"本文基于B站UP主**{author or '未知作者'}**的视频《**{title or '未知标题'}**》总结得来")
        if video_url:
            parts.append(f"，网址：[{video_url}]({video_url})")
        parts.append("\n")

        parts.append("免责声明：本文仅为内容总结，不构成投资建议。\n\n**总结可能存在遗漏或错误，不能完全反映原视频观点，请谨慎对待。**\n")
        parts.append("**精华总结已涵盖全部内容，其余均为引用，请以原视频及引用部分为准。**\n\n")
        return parts

    def render_time_references(self, time_ranges: list) -> list[str]:
        """渲染引用时间列表，使用反引号包裹时间范围。"""
        parts = []
        if not time_ranges:
            parts.append("无\n\n")
            return parts

        time_strs = []
        for tr in time_ranges:
            start = tr.get("开始时间", "")
            end = tr.get("结束时间", "")
            if start and end:
                time_strs.append(f"`({start}-{end})`")

        if time_strs:
            parts.append(f"{'，'.join(time_strs)}\n\n")

        return parts

    def render_citations(self, citations: list) -> str:
        """渲染引用块（基础版，无超链接）。"""
        if not citations:
            return ""
        lines = []
        for c in citations:
            if not isinstance(c, dict):
                continue
            start_time = c.get("开始时间", "")
            end_time = c.get("结束时间", "")
            text = c.get("原文", "") or c.get("引用原文", "")
            if start_time and end_time:
                lines.append(f"`{start_time} - {end_time} {text}`")
            else:
                lines.append(f"`{text}`")
            lines.append("")
        return "\n".join(lines)

    def render_section_title(self, title: str, is_first: bool = False) -> str:
        """渲染一级标题。"""
        if is_first:
            return f'<h1 align="center">{title}</h1>\n\n'
        else:
            return f'\n\n---\n\n<h1 align="center">{title}</h1>\n\n'

    def render_subsection_title(self, title: str) -> str:
        """渲染二级标题。"""
        return f"## {title}\n\n"

    def render_dict_fields(self, value_dict: dict, cit_fn: Callable[[list], str]) -> list[str]:
        """渲染一个扁平 dict：遍历所有非引用字段作为内容行，最后追加引用块。"""
        parts = []
        for field_key, field_val in value_dict.items():
            if field_key == "引用":
                continue
            if field_key == "原始黑话":
                continue
            if isinstance(field_val, str):
                parts.append(f"{field_key}：{field_val or '无'}\n\n")
            elif isinstance(field_val, list):
                items_str = "、".join(self.decorate_jargon(str(x)) for x in field_val)
                parts.append(f"{field_key}：**{items_str or '无'}**\n\n")
            elif field_val is not None:
                parts.append(f"{field_key}：{field_val or '无'}\n\n")

        citations = value_dict.get("引用", [])
        if citations:
            cit_text = cit_fn(citations)
            if cit_text:
                parts.append(cit_text + "\n\n")

        return parts

    def render_complex_list(self, items: list, cit_fn: Callable[[list], str]) -> list[str]:
        """渲染包含嵌套 dict 的复杂列表项。"""
        parts = []
        title_keys = ["标题", "标的名称"]

        for item in items:
            if not isinstance(item, dict):
                continue

            item_title = ""
            title_field = None
            for tk in title_keys:
                if tk in item:
                    item_title = item[tk]
                    title_field = tk
                    break

            if item_title:
                parts.append(f"### {self.decorate_jargon(item_title)}\n\n")

            for field_key, field_val in item.items():
                if field_key == title_field:
                    continue
                if field_key == "引用":
                    continue
                if field_key == "原始黑话":
                    continue
                if field_key == "因果链" and isinstance(field_val, list) and field_val:
                    for link in field_val:
                        if not isinstance(link, dict):
                            continue
                        link_type = link.get("环节类型", "")
                        content = (link.get("内容") or "").strip()
                        if not content:
                            continue
                        parts.append(f"- **{link_type}**：{content}\n")
                        cit_text = cit_fn(link.get("引用") or [])
                        if cit_text:
                            # 空行分隔：懒续行会把引用并入字段内容同段（HTML 挤到同一行），空行让引用成为列表项内独立段落
                            parts.append("\n" + cit_text + "\n\n")
                    continue

                if isinstance(field_val, dict):
                    if not field_val:
                        continue
                    viewpoint = field_val.get("观点")
                    logic = field_val.get("逻辑", "")
                    if viewpoint:
                        parts.append(f"{field_key}：{viewpoint}\n\n")
                        # 修复：只有当观点是"未表态"且逻辑为空时，才不渲染引用
                        # 否则（未表态但逻辑不为空，或非未表态）都渲染逻辑和引用
                        if "未表态" not in viewpoint or logic:
                            remaining = {k: v for k, v in field_val.items() if k != "观点"}
                            parts.extend(self.render_dict_fields(remaining, cit_fn))
                    else:
                        parts.append(f"**{field_key}**\n\n")
                        parts.extend(self.render_dict_fields(field_val, cit_fn))
                elif isinstance(field_val, str):
                    parts.append(f"{field_key}：{field_val or '无'}\n\n")
                elif isinstance(field_val, list) and field_val:
                    if all(isinstance(x, str) for x in field_val):
                        items_str = "、".join(self.decorate_jargon(x) for x in field_val)
                        parts.append(f"{field_key}：**{items_str}**\n\n")
                    else:
                        items_str = "、".join(str(x) for x in field_val)
                        parts.append(f"{field_key}：**{items_str}**\n\n")
                elif isinstance(field_val, bool):
                    parts.append(f"{field_key}：{field_val or '无'}\n\n")
                elif isinstance(field_val, list) and not field_val:
                    continue
                elif field_val is not None:
                    parts.append(f"{field_key}：{field_val}\n\n")

            citations = item.get("引用", [])
            if citations:
                cit_text = cit_fn(citations)
                if cit_text:
                    parts.append(cit_text + "\n\n")

        return parts

    def render_object_list(self, items: list, column_order: list[str] | None = None) -> list[str]:
        """将纯对象数组渲染为 Markdown 列表或表格。"""
        parts = []
        if not items:
            parts.append("无\n\n")
            return parts

        all_keys = []
        seen = set()
        for item in items:
            if isinstance(item, dict):
                for k in item.keys():
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)

        if len(all_keys) <= 0:
            return parts

        if column_order is not None and len(column_order) > 0:
            ordered_keys = []
            for k in column_order:
                if k in seen:
                    ordered_keys.append(k)
            for k in all_keys:
                if k not in column_order:
                    ordered_keys.append(k)
            all_keys = ordered_keys

        if len(all_keys) == 2:
            k1, k2 = all_keys[0], all_keys[1]
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = item.get(k1, "")
                desc = item.get(k2, "")
                if name:
                    label = f"[{name}](#{self.heading_link(name)})"
                else:
                    label = name
                parts.append(f"- {label} —— {desc}\n")
            parts.append("\n")
        elif len(all_keys) >= 3:
            parts.append("| " + " | ".join(all_keys) + " |\n")
            parts.append("| " + " | ".join(["---"] * len(all_keys)) + " |\n")
            for item in items:
                if not isinstance(item, dict):
                    continue
                row = []
                for k in all_keys:
                    cell_content = str(item.get(k, "")) or "无"
                    cell_content = cell_content.replace("|", "\\|").replace("\n", "<br>")
                    row.append(cell_content)
                parts.append("| " + " | ".join(row) + " |\n")
            parts.append("\n")
        else:
            k1 = all_keys[0]
            for item in items:
                if not isinstance(item, dict):
                    continue
                val = item.get(k1, "")
                parts.append(f"- {val}\n")
            parts.append("\n")

        return parts

