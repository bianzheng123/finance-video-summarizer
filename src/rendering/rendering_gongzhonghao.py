"""公众号版渲染类，提供公众号 HTML 的渲染实现。"""

from collections.abc import Callable

from markdown_it import MarkdownIt

from .rendering_base import BaseRenderer


_GZH_CSS = """
* { box-sizing: border-box; }

#wenyan {
    font-family: -apple-system, "PingFang SC", "Helvetica Neue", Arial, "Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", sans-serif;
    font-size: 15px;
    line-height: 1.8;
    color: #333333;
    max-width: 677px;
    margin: 0 auto;
    padding: 20px 16px;
    overflow-wrap: break-word;
}

#wenyan h1 {
    font-size: 20px;
    font-weight: bold;
    color: #1a1a1a;
    margin: 28px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #f4b183;
    text-align: center;
}

#wenyan h2 {
    font-size: 17px;
    font-weight: bold;
    color: #1a1a1a;
    margin: 22px 0 10px 0;
    padding: 4px 0 4px 12px;
    border-left: 4px solid #f4b183;
}

#wenyan h3 {
    font-size: 15px;
    font-weight: bold;
    color: #333333;
    margin: 16px 0 8px 0;
}

#wenyan p {
    margin: 8px 0;
    line-height: 1.8;
    color: #333333;
}

#wenyan ul, #wenyan ol {
    margin: 8px 0;
    padding-left: 22px;
}

#wenyan li {
    margin: 4px 0;
    line-height: 1.8;
    color: #333333;
}

#wenyan blockquote {
    margin: 12px 0;
    padding: 8px 14px;
    border-left: 4px solid #f4b183;
    background: #f6f6f6;
    color: #666666;
}

#wenyan blockquote p {
    margin: 0;
    color: #666666;
}

#wenyan table {
    border-collapse: collapse;
    width: 100%;
    margin: 14px 0;
    font-size: 14px;
}

#wenyan th {
    border: 1px solid #dddddd;
    padding: 8px 10px;
    background: #f0f0f0;
    font-weight: bold;
    text-align: center;
    color: #333333;
}

#wenyan td {
    border: 1px solid #dddddd;
    padding: 8px 10px;
    color: #333333;
    text-align: center;
}

#wenyan tr:nth-child(even) td {
    background: #fafafa;
}

#wenyan code {
    background: #f4f4f4;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: "Courier New", Courier, monospace;
    font-size: 13px;
    color: #c7254e;
}

#wenyan pre {
    background: #f8f8f8;
    padding: 14px 16px;
    border-radius: 4px;
    margin: 12px 0;
    border: 1px solid #e8e8e8;
    overflow-wrap: break-word;
    white-space: pre-wrap;
}

#wenyan pre code {
    background: transparent;
    padding: 0;
    border-radius: 0;
    font-size: 13px;
    color: #333333;
    white-space: pre;
}

#wenyan strong {
    font-weight: bold;
    color: #1a1a1a;
}

#wenyan em {
    font-style: italic;
    color: #555555;
}

#wenyan hr {
    border: none;
    border-top: 1px solid #e8e8e8;
    margin: 20px 0;
}

#wenyan a {
    color: #f4b183;
    text-decoration: none;
}
"""

_GZH_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>公众号文章</title>
<style>
body {
    background: #ffffff;
    margin: 0;
    padding: 0;
}
{css}
</style>
</head>
<body>
<section id="wenyan">
{content}
</section>
</body>
</html>
"""


class GongzhonghaoRenderer(BaseRenderer):
    """公众号版 Markdown 渲染器（summary_gongzhonghao.html 使用），无超链接。"""

    def render_header(self, author: str | None, title: str | None, bvid: str | None) -> list[str]:
        """渲染头部信息（视频来源、微信号等）。"""
        parts = []
        video_url = f"https://www.bilibili.com/video/{bvid}" if bvid else ""
        parts.append(f"本文基于B站UP主**{author or '未知作者'}**的视频《**{title or '未知标题'}**》总结得来")
        if video_url:
            parts.append(f"，网址：[{video_url}]({video_url})")
        parts.append("\n\n")

        parts.append("\n免责声明：本文仅为内容总结，不构成投资建议。\n\n**总结可能存在遗漏或错误，不能完全反映原视频观点，请谨慎对待。**\n\n")
        parts.append("**精华总结已涵盖全部内容，其余均为引用，请以原视频及引用部分为准。**\n\n")
        return parts

    def render_time_references(self, time_ranges: list) -> list[str]:
        """渲染引用时间列表，格式与引用正文一致，使用反引号包裹。"""
        parts = []
        if not time_ranges:
            parts.append("无\n\n")
            return parts

        time_strs = []
        for tr in time_ranges:
            start = tr.get("开始时间", "")
            end = tr.get("结束时间", "")
            if start and end:
                time_strs.append(f"`{start} - {end}`")

        if time_strs:
            parts.append(f"{'，'.join(time_strs)}\n\n")

        return parts

    def render_citations(self, citations: list) -> str:
        """渲染引用块，公众号版无超链接。"""
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
                    if viewpoint:
                        parts.append(f"{field_key}：{viewpoint}\n\n")
                        if "未表态" not in viewpoint:
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
                parts.append(f"- {name} —— {desc}\n")
            parts.append("\n")
        elif len(all_keys) >= 3:
            col_avg_lens = []
            for k in all_keys:
                total_len = 0
                count = 0
                header_lines = k.split("\n")
                for line in header_lines:
                    total_len += len(line)
                    count += 1
                for item in items:
                    if isinstance(item, dict):
                        cell = str(item.get(k, "")) or "无"
                        cell_lines = cell.split("\n")
                        for line in cell_lines:
                            if line == "无":
                                continue
                            total_len += len(line)
                            count += 1
                avg_len = int(total_len / count) if count > 0 else 0
                col_avg_lens.append(avg_len)
            total_avg_len = sum(col_avg_lens)
            if total_avg_len > 0:
                col_percentages = [f"{(l / total_avg_len) * 100:.1f}%" for l in col_avg_lens]
            else:
                col_percentages = [f"{100.0 / len(all_keys):.1f}%"] * len(all_keys)

            parts.append('<center><table style="width: 100%; max-width: 800px; border-collapse: collapse; margin: 14px 0; font-size: 14px;">\n')
            parts.append('<colgroup>\n')
            for pct in col_percentages:
                parts.append(f'  <col style="width: {pct};">\n')
            parts.append('</colgroup>\n')
            parts.append('<thead><tr style="background: #f0f0f0;">\n')
            for k in all_keys:
                parts.append(f'  <th style="border: 1px solid #ddd; padding: 8px 12px; text-align: center; font-weight: bold; white-space: nowrap;">{k}</th>\n')
            parts.append('</tr></thead>\n')
            parts.append('<tbody>\n')
            # 构建 rowspan 信息（针对第一列）
            skip_cells = set()
            rowspan_cells = {}
            
            if all_keys:
                groups = []
                current_group = []
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    cell_text = str(item.get(all_keys[0], "")) or "无"
                    if not current_group:
                        current_group.append((idx, cell_text))
                    else:
                        if cell_text == current_group[-1][1]:
                            current_group.append((idx, cell_text))
                        else:
                            groups.append(current_group)
                            current_group = [(idx, cell_text)]
                if current_group:
                    groups.append(current_group)
                
                for group in groups:
                    if len(group) > 1:
                        start_idx = group[0][0]
                        count = len(group)
                        rowspan_cells[start_idx] = count
                        for i in range(start_idx + 1, start_idx + count):
                            skip_cells.add((i, 0))
            
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                bg_color = 'background: #fafafa;' if idx % 2 == 1 else ''
                parts.append(f'  <tr style="{bg_color}">\n')
                for ci, k in enumerate(all_keys):
                    if (idx, ci) in skip_cells:
                        continue
                    cell_content = str(item.get(k, "")) or "无"
                    cell_content = cell_content.replace("|", "\|").replace("\n", "<br>")
                    
                    if ci == 0 and idx in rowspan_cells:
                        rowspan_value = rowspan_cells[idx]
                        parts.append(f'    <td rowspan="{rowspan_value}" style="border: 1px solid #ddd; padding: 8px 12px; text-align: center; word-wrap: break-word;">{cell_content}</td>\n')
                    else:
                        parts.append(f'    <td style="border: 1px solid #ddd; padding: 8px 12px; text-align: center; word-wrap: break-word;">{cell_content}</td>\n')
                parts.append('  </tr>\n')
            parts.append('</tbody>\n')
            parts.append('</table></center>\n\n')
        else:
            k1 = all_keys[0]
            for item in items:
                if not isinstance(item, dict):
                    continue
                val = item.get(k1, "")
                parts.append(f"- {val}\n")
            parts.append("\n")

        return parts

    @staticmethod
    def markdown_to_html(md_text: str) -> str:
        md = MarkdownIt("commonmark", {"html": True}).enable("table")
        content_html = md.render(md_text)
        return _GZH_HTML_TEMPLATE.replace("{css}", _GZH_CSS).replace("{content}", content_html)
