import json
from pathlib import Path
import logging
import sys
import io
from collections.abc import Callable
from typing import Any

from .rendering.rendering_base import BaseRenderer
from .rendering.rendering_summary import SummaryRenderer as SummaryRendererImpl
from .rendering.rendering_gongzhonghao import GongzhonghaoRenderer as GongzhonghaoRendererImpl
from .rendering.rendering_html_structured import render_structured_html
from .rendering.inv_filter import inv_event_titles, inv_has_substance, inv_should_skip, is_unstated
from .log_config import step_logger
from .utils.analysis_cleanup import clean_analysis

logger = logging.getLogger(__name__)

# 非主题的顶层固定章节：既不被当作主题参与排序，也不产出主题级速览。
FIXED_SECTION_NAMES = {
    "宏观分析", "交易技巧", "精炼总结", "主讲人暗示重要话题",
}


class SummaryRenderer:
    """总结文件渲染主控制器：生成 summary.md + summary.html + summary.pdf + summary_structured.html。"""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """将秒数转为 XX分XX秒 格式"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs}秒"

    @staticmethod
    def _speaker_opinion_text(op: dict) -> str | None:
        """「观点」条目 → 观点裸文本（只展示观点文本，不拼因果链与引用时间；前缀与编号由调用方拼装）。无实质内容返回 None。"""
        text = (op.get("观点") or "").strip()
        if not text or "未表态" in text:
            return None
        return text

    @staticmethod
    def _indent_citations(cit_fn: Callable[[list], str]) -> Callable[[list], str]:
        """包装引用函数：每个非空行加 `\\t` 前缀（另起一行并与字段内容并列），空串原样返回。"""
        def wrapped(citations: list) -> str:
            text = cit_fn(citations)
            if not text:
                return ""
            return "\n".join("\t" + line if line else "" for line in text.strip("\n").split("\n"))
        return wrapped

    @staticmethod
    def _build_asset_groups(
        cleaned_invs: list[dict], speaker_opinions: list[dict]
    ) -> list[tuple[str, list[dict], list[dict]]]:
        """把 投资机会 与 观点 条目按资产名分组，返回 (资产名, 投资机会条目, 观点条目) 有序组列表。

        资产顺序：投资机会条目原顺序在前，仅观点资产按观点原顺序在后。
        名称为空的条目单独建空名组，渲染时由调用方决定是否输出组标题。
        """
        groups: list[tuple[str, list[dict], list[dict]]] = []
        group_index: dict[str, int] = {}

        def get_group(name: str) -> tuple[str, list[dict], list[dict]]:
            if name not in group_index:
                group_index[name] = len(groups)
                groups.append((name, [], []))
            return groups[group_index[name]]

        for inv in cleaned_invs:
            if not isinstance(inv, dict):
                continue
            get_group((inv.get("标题") or "").strip())[1].append(inv)
        for op in speaker_opinions:
            if not isinstance(op, dict):
                continue
            get_group((op.get("标的名称") or "").strip())[2].append(op)
        return groups

    @staticmethod
    def _summarize_hot_sector_prediction(payload: dict) -> str:
        """把结构化的热点板块预测压成一行字符串，供 TLDR 大盘概览展示。

        优先用 `总览` 字段；若缺失则从 `板块清单` 拼一段简要描述。
        """
        if not isinstance(payload, dict):
            return ""
        overview = (payload.get("总览") or "").strip()
        items = payload.get("板块清单") or []
        if overview:
            return overview
        snippets = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("板块名称", "")
            tendency = item.get("预测倾向", "")
            window = item.get("时间窗口", "")
            if not name:
                continue
            label = f"{name}({tendency}"
            if window:
                label += f", {window}"
            label += ")"
            snippets.append(label)
        return "；".join(snippets)

    @staticmethod
    def _join_macro_blocks(field_val: Any) -> str:
        """宏观分块字段 → 一行字符串：list 取各块标题以顿号串联（空标题回退观点；dict 为旧格式快照，其余返回空串）。"""
        if isinstance(field_val, list):
            parts = []
            for block in field_val:
                if isinstance(block, dict):
                    text = (block.get("标题") or block.get("观点") or "").strip()
                    if text:
                        parts.append(text)
            return "、".join(parts)
        if isinstance(field_val, dict):
            return (field_val.get("观点") or "").strip()
        return ""

    @staticmethod
    def _generate_tldr(analysis: dict) -> dict:
        """
        生成TLDR精炼总结部分
        """
        tldr_result = {}

        macro_analysis = analysis.get("宏观分析") or {}

        # 大盘概览
        market_overview = {
            "今日市场": SummaryRenderer._join_macro_blocks(macro_analysis.get("大盘情绪与状态判断")),
            "宏观事件": SummaryRenderer._join_macro_blocks(macro_analysis.get("宏观经济与事件影响分析")),
            "后市观点": SummaryRenderer._join_macro_blocks(macro_analysis.get("后市观点")),
            "仓位建议": SummaryRenderer._join_macro_blocks(macro_analysis.get("仓位建议")),
            "热点板块预测": SummaryRenderer._summarize_hot_sector_prediction(macro_analysis.get("热点板块预测") or {}),
        }
        tldr_result["大盘概览"] = market_overview

        short_term_opportunities = []
        sector_opinion_overview = []
        sector_event_overview = []
        market_knowledge_index = []
        sensitive_hints_overview: list[dict] = []

        # 获取主题列表（排除固定章节）
        topics = []
        fixed_sections = FIXED_SECTION_NAMES
        for key in analysis.keys():
            if key not in fixed_sections and isinstance(analysis.get(key), dict):
                topics.append(key)

        # 按板块引用总时长降序排序
        topics.sort(key=lambda k: analysis[k].get("引用总时长", 0), reverse=True)

        # 处理各个主题
        for topic in topics:
            topic_data = analysis.get(topic) or {}
            # 格式化板块名称（带引用时长）
            topic_duration = topic_data.get("引用总时长", 0)
            topic_duration_str = SummaryRenderer._format_duration(topic_duration) if topic_duration > 0 else ""
            topic_label = f"{topic} ({topic_duration_str})" if topic_duration_str else topic

            investment_opportunities = topic_data.get("投资机会") or []
            events_and_assets = topic_data.get("事件与资产") or []
            speaker_opinions = topic_data.get("观点") or []

            # 已匹配到投资机会条目的「观点」条目下标（循环后兜底处理未匹配项）
            matched_op_idxs: set[int] = set()

            for inv in investment_opportunities:
                if not isinstance(inv, dict):
                    continue
                title = inv.get("标题", "")
                short_term = inv.get("短线") if isinstance(inv.get("短线"), dict) else {}
                medium_long_term = inv.get("中线") or inv.get("长线") or inv.get("中长线")
                if not isinstance(medium_long_term, dict):
                    medium_long_term = {}

                # 主讲人评价（顶层「观点」数组）：强制相等匹配（标的名称 == 标题）。
                # prompt 铁律 B 已约束标的名称严格 ∈ 白名单，黑话注入对「投资机会.标题」
                # 与「观点.标的名称」统一执行，两者天然同源，精确相等即可对齐；不再做
                # 子串回退——否则父板块名（如「半导体」）会经 `"半导体" in "半导体耗材"`
                # 扩散到所有子项。名称不同的观点（如 LLM 用黑话原文）交由下方「未匹配
                # 兜底」独立成行。
                matched_ops: list[dict] = []
                op_candidates = [
                    (idx, op) for idx, op in enumerate(speaker_opinions)
                    if isinstance(op, dict)
                ]
                pairs = [
                    (idx, op) for idx, op in op_candidates
                    if (op.get("标的名称") or "") == title
                ]
                for idx, op in pairs:
                    matched_ops.append(op)
                    matched_op_idxs.add(idx)

                # 规则 3：全部立场未表态 → 整 inv 砍掉；但存在匹配的主讲人观点时保留
                # （事件已独立成「个股事件速览」表，不再参与本表的跳过判断）
                if inv_should_skip(inv, []) and not matched_ops:
                    continue

                # 短期机会速览：仅收主讲人"短期看多"的标的（看空/中性/未表态由"个股观点速览"和主题章节覆盖）
                if "看多" in (short_term.get("观点") or ""):
                    desc = (short_term.get("逻辑") or "").strip() or "主讲人短期看多"
                    short_term_opportunities.append({
                        "名称": title,
                        "描述": desc
                    })

                # 个股观点速览的观点列：字段级过滤——未表态那条不出现
                viewpoint_lines: list[str] = []

                def _format(label: str, term: dict) -> str | None:
                    vp = term.get("观点") if isinstance(term, dict) else None
                    if is_unstated(vp):
                        return None
                    logic = term.get("逻辑", "") if isinstance(term, dict) else ""
                    return f"{label}:{vp} {logic}".rstrip()

                line = _format("短线", short_term)
                if line:
                    viewpoint_lines.append(line)
                line = _format("中长线", medium_long_term)
                if line:
                    viewpoint_lines.append(line)
                price = inv.get("当前价格")
                price_view = price.get("观点", "") if isinstance(price, dict) else (price or "")
                if not is_unstated(price):
                    viewpoint_lines.append(f"当前价格:{price_view}")

                # 主讲人观点：与枚举立场行同时展示；≥2 条时合并为一个标签 + 数字编号，避免重复「主讲人观点」前缀
                op_lines: list[str] = []
                for op in matched_ops:
                    op_text = SummaryRenderer._speaker_opinion_text(op)
                    if op_text:
                        op_lines.append(op_text)
                if len(op_lines) == 1:
                    viewpoint_lines.append(f"主讲人观点：{op_lines[0]}")
                elif op_lines:
                    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(op_lines, 1))
                    viewpoint_lines.append(f"主讲人观点：{numbered}")

                sector_opinion_overview.append({
                    "板块": topic_label,
                    "名称": title,
                    "观点": "\n".join(viewpoint_lines) if viewpoint_lines else "无明确观点"
                })

            # 「观点」条目未匹配到任何投资机会条目时兜底生成独立行，保证内容不丢
            # （铁律保证白名单资产必有投资机会条目，正常不应走到这里）
            for idx, op in enumerate(speaker_opinions):
                if idx in matched_op_idxs or not isinstance(op, dict):
                    continue
                op_text = SummaryRenderer._speaker_opinion_text(op)
                if not op_text:
                    continue
                sector_opinion_overview.append({
                    "板块": topic_label,
                    "名称": op.get("标的名称", ""),
                    "观点": f"主讲人观点：{op_text}",
                })

            # 个股事件速览：每行一个事件，涉及资产为资产数组（顿号串联），不再拆散挂到个股
            for event in events_and_assets:
                if not isinstance(event, dict):
                    continue
                et = event.get("标题", "")
                if not et:
                    continue
                related = event.get("涉及资产")
                assets_text = "、".join(related) if isinstance(related, list) else ""
                sector_event_overview.append({
                    "板块": topic_label,
                    "事件": et,
                    "涉及资产": assets_text,
                })

            # 市场知识
            market_knowledge = topic_data.get("市场知识") or []
            for mk in market_knowledge:
                if isinstance(mk, dict):
                    market_knowledge_index.append({
                        "板块": topic,
                        "知识简要": mk.get("标题", ""),
                        "详细内容": mk.get("内容", "")
                    })

        # 速览行精确去重兜底：观点表按（板块, 名称）、事件表按（板块, 事件），
        # 同一键只保留第一条（正常流程不应出现重复，纯防御）
        def _dedup_rows(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
            seen: set[tuple[str, ...]] = set()
            out: list[dict] = []
            for row in rows:
                k = tuple(str(row.get(key) or "") for key in keys)
                if k in seen:
                    continue
                seen.add(k)
                out.append(row)
            return out

        sector_opinion_overview = _dedup_rows(sector_opinion_overview, ("板块", "名称"))
        sector_event_overview = _dedup_rows(sector_event_overview, ("板块", "事件"))

        tldr_result["短期机会速览"] = short_term_opportunities
        tldr_result["个股观点速览"] = sector_opinion_overview
        tldr_result["个股事件速览"] = sector_event_overview
        tldr_result["市场知识索引"] = market_knowledge_index

        # 主讲人暗示重要话题速览：从顶层聚合 section 反向构造每行一条原话的快览表
        aggregated_hints = analysis.get("主讲人暗示重要话题") or []
        for rec in aggregated_hints:
            if not isinstance(rec, dict):
                continue
            source_topics = rec.get("板块") or []
            labels: list[str] = []
            for t in source_topics:
                td = analysis.get(t, {}) if isinstance(analysis.get(t), dict) else {}
                dur = td.get("引用总时长", 0) if isinstance(td, dict) else 0
                dur_str = SummaryRenderer._format_duration(dur) if dur and dur > 0 else ""
                labels.append(f"{t} ({dur_str})" if dur_str else t)
            sensitive_hints_overview.append({
                "板块": "、".join(labels) if labels else "",
                "信号类型": (rec.get("信号类型") or "敏感提示"),
                "原话总结": (rec.get("原话总结") or "").strip(),
                "原话": (rec.get("原话") or "").strip(),
            })
        if sensitive_hints_overview:
            tldr_result["主讲人暗示重要话题"] = sensitive_hints_overview

        # 交易技巧摘要
        trading_skills_summary = []
        trading_skills = analysis.get("交易技巧", {})
        if isinstance(trading_skills, dict):
            for skill_name, skill_content in trading_skills.items():
                if isinstance(skill_content, dict) and "具体方法" in skill_content:
                    trading_skills_summary.append({
                        "名称": skill_name,
                        "描述": skill_content["具体方法"]
                    })
        tldr_result["交易技巧摘要"] = trading_skills_summary

        # 过滤空字段
        return SummaryRenderer._filter_empty_fields(tldr_result)

    @staticmethod
    def _filter_empty_fields(data: dict) -> dict:
        """递归过滤掉空数组和空字典的字段"""
        if not isinstance(data, dict):
            return data

        filtered_dict = {}

        for key, value in data.items():
            if isinstance(value, dict):
                filtered_value = SummaryRenderer._filter_empty_fields(value)
                if filtered_value:
                    filtered_dict[key] = filtered_value
            elif isinstance(value, list):
                filtered_list = []
                for item in value:
                    if isinstance(item, dict):
                        filtered_item = SummaryRenderer._filter_empty_fields(item)
                        if filtered_item:
                            filtered_list.append(filtered_item)
                    elif item not in (None, "", [], {}):
                        filtered_list.append(item)
                if filtered_list:
                    filtered_dict[key] = filtered_list
            else:
                if value not in (None, "", [], {}):
                    filtered_dict[key] = value

        return filtered_dict

    def _get_render_order(self, analysis: dict) -> list[str]:
        """获取渲染顺序：精炼总结 -> 宏观分析 -> 主讲人暗示重要话题 -> 板块分析（按引用总时长从大到小）-> 交易技巧"""
        order = []

        if "精炼总结" in analysis:
            order.append("精炼总结")
        if "宏观分析" in analysis:
            order.append("宏观分析")
        if "主讲人暗示重要话题" in analysis:
            order.append("主讲人暗示重要话题")

        # isinstance 守卫不可省：顶层 section 并非都是 dict，
        # 无守卫会在下面按 `引用总时长` 排序时 AttributeError。
        topics = []
        for key in analysis.keys():
            if key not in FIXED_SECTION_NAMES and isinstance(analysis[key], dict):
                topics.append((key, analysis[key]))

        topics.sort(key=lambda x: x[1].get("引用总时长", 0), reverse=True)
        for topic_name, _ in topics:
            order.append(topic_name)

        if "交易技巧" in analysis:
            order.append("交易技巧")

        return order

    def _render_macro_section(self, renderers: list[tuple[str, BaseRenderer]], renderer_parts: dict[str, list[str]], section_data: dict) -> None:
        """渲染宏观分析章节"""
        for name, renderer in renderers:
            renderer_parts[name].append(renderer.render_section_title("宏观分析", is_first=False))

        field_keys = ["大盘情绪与状态判断", "宏观经济与事件影响分析", "后市观点", "仓位建议", "热点板块预测"]
        for field_key in field_keys:
            field_val = section_data.get(field_key)
            if field_val is None:
                continue
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title(field_key))
                if field_key == "热点板块预测" and isinstance(field_val, dict):
                    renderer_parts[name].extend(
                        self._render_hot_sector_prediction(renderer, field_val)
                    )
                elif isinstance(field_val, list):
                    # 分块字段：每块 `### 标题` + 观点 + 该块自己的引用
                    renderer_parts[name].extend(
                        renderer.render_complex_list(field_val, renderer.render_citations)
                    )
                elif isinstance(field_val, dict):
                    renderer_parts[name].extend(
                        renderer.render_dict_fields(field_val, renderer.render_citations)
                    )
                else:
                    renderer_parts[name].append(f"{field_val}\n\n")

    @staticmethod
    def _render_hot_sector_prediction(renderer: BaseRenderer, payload: dict) -> list[str]:
        """渲染热点板块预测：先一段总览，再渲染每个板块条目。"""
        parts: list[str] = []
        overview = (payload.get("总览") or "").strip()
        if overview:
            parts.append(f"{overview}\n\n")

        items = payload.get("板块清单") or []
        if not items:
            return parts

        # 把 `板块名称` 重命名为 `标题`，让 render_complex_list 可以正确渲染条目标题
        display_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            new_item = {"标题": item.get("板块名称", "")}
            for k, v in item.items():
                if k == "板块名称":
                    continue
                # 跳过空字符串字段，让渲染更清爽
                if isinstance(v, str) and not v.strip():
                    continue
                new_item[k] = v
            display_items.append(new_item)

        parts.extend(renderer.render_complex_list(display_items, renderer.render_citations))
        return parts

    def _render_sensitive_hints_section(self, renderers: list[tuple[str, BaseRenderer]], renderer_parts: dict[str, list[str]], hints: list) -> None:
        """渲染独立的"主讲人暗示重要话题"章节。

        数据来源：section_analyzer._aggregate_sensitivity_hints 写入的顶层列表。
        每条记录含 板块（list）/信号类型/原话/原话总结/引用（list[{开始时间,结束时间,原文}]）。
        小标题用 LLM 输出的 ≤20 字 `原话总结`，引用块直接拼时间 + 原话整段。
        """
        if not isinstance(hints, list) or not hints:
            return

        for name, renderer in renderers:
            renderer_parts[name].append(renderer.render_section_title("🔒 重要暗示", is_first=False))

        for rec in hints:
            if not isinstance(rec, dict):
                continue
            signal_type = (rec.get("信号类型") or "敏感提示").strip()
            summary = (rec.get("原话总结") or "").strip()
            citations = rec.get("引用") or []

            cit = citations[0] if citations and isinstance(citations[0], dict) else {}
            start = cit.get("开始时间", "")
            end = cit.get("结束时间", "")
            quote_text = cit.get("原文", "") or rec.get("原话", "")

            title = f"{summary}--{signal_type}" if summary else signal_type

            for name, renderer in renderers:
                renderer_parts[name].append(
                    renderer.render_subsection_title(title)
                )
                if start and end and quote_text:
                    renderer_parts[name].append(f"`{start} - {end} {quote_text}`\n\n")
                elif quote_text:
                    renderer_parts[name].append(f"`{quote_text}`\n\n")

    def _render_trading_skills_section(self, renderers: list[tuple[str, BaseRenderer]], renderer_parts: dict[str, list[str]], section_data: dict) -> None:
        """渲染交易技巧章节"""
        for name, renderer in renderers:
            renderer_parts[name].append(renderer.render_section_title("交易技巧", is_first=False))

        for skill_name, skill_content in section_data.items():
            if not isinstance(skill_content, dict):
                continue
            
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title(skill_name))
                
                method = skill_content.get("具体方法", "")
                if method:
                    renderer_parts[name].append(f"**具体方法**：{method}\n\n")
                
                scope = skill_content.get("适用范围或前提", "")
                if scope:
                    renderer_parts[name].append(f"**适用范围或前提**：{scope}\n\n")
                
                citations = skill_content.get("引用", [])
                if citations:
                    cit_text = renderer.render_citations(citations)
                    if cit_text:
                        renderer_parts[name].append(cit_text + "\n\n")

    def _render_topic_section(self, renderers: list[tuple[str, BaseRenderer]], renderer_parts: dict[str, list[str]], section_key: str, section_data: dict) -> None:
        """渲染主题分析章节"""
        for name, renderer in renderers:
            decorated_key = renderer.decorate_jargon(section_key)
            renderer_parts[name].append(renderer.render_section_title(decorated_key, is_first=False))

        cite_time = section_data.get("引用时间")
        if cite_time:
            total_seconds = section_data.get("引用总时长", 0)
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            duration_label = f"引用时间：共{minutes}分{seconds}秒"
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title(duration_label))
                renderer_parts[name].extend(renderer.render_time_references(cite_time))

        invest_opportunity = section_data.get("投资机会")
        events_and_assets_for_topic = section_data.get("事件与资产", []) or []
        cleaned_invs: list[dict] = []
        if invest_opportunity:
            for inv in invest_opportunity:
                if not isinstance(inv, dict):
                    continue
                event_titles = inv_event_titles(inv, events_and_assets_for_topic)
                # 规则 3：事件无 + 全部未表态 → 整 inv 砍掉
                if inv_should_skip(inv, event_titles):
                    continue
                # 规则 2：字段级隐藏 —— 浅拷贝，把未表态的 短线/中长线/当前价格 删掉
                cleaned = dict(inv)
                for tk in ("短线", "中长线", "中线", "长线"):
                    tv = cleaned.get(tk)
                    if isinstance(tv, dict) and is_unstated(tv.get("观点")):
                        cleaned.pop(tk, None)
                price_viewpoint = (cleaned.get("当前价格") or {}).get("观点")
                if is_unstated(price_viewpoint):
                    cleaned.pop("当前价格", None)

                # 检查清理后是否还有实质性内容（非未表态的观点或价格）
                # 如果没有实质性内容，跳过该项，不渲染
                if not inv_has_substance(cleaned):
                    continue

                cleaned_invs.append(cleaned)

        # 投资机会 与 观点 合并渲染：按资产分组，资产内先投资机会、后观点
        speaker_opinions = section_data.get("观点") or []
        asset_groups = SummaryRenderer._build_asset_groups(cleaned_invs, speaker_opinions)
        if asset_groups:
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("投资机会与观点"))
                for asset_name, invs, opinions in asset_groups:
                    if asset_name:
                        renderer_parts[name].append(f"### {renderer.decorate_jargon(asset_name)}\n\n")
                    inv_items = [{k: v for k, v in inv.items() if k != "标题"} for inv in invs]
                    if inv_items:
                        renderer_parts[name].extend(renderer.render_complex_list(inv_items, renderer.render_citations))
                    opinion_items = [{k: v for k, v in op.items() if k != "标的名称"} for op in opinions]
                    # 有因果链的条目：引用在列表项内，tab 缩进安全；无因果链的条目：整条引用顶格，
                    # 避免 tab 行在段落外被 CommonMark 解析成缩进代码块（公众号 HTML 反引号外露）
                    chained = [op for op in opinion_items if op.get("因果链")]
                    if chained:
                        renderer_parts[name].extend(
                            renderer.render_complex_list(
                                chained,
                                SummaryRenderer._indent_citations(renderer.render_citations),
                            )
                        )
                    plain = [op for op in opinion_items if not op.get("因果链")]
                    if plain:
                        renderer_parts[name].extend(renderer.render_complex_list(plain, renderer.render_citations))

        personal_trades = section_data.get("个人交易记录")
        if personal_trades:
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("个人交易记录"))
                renderer_parts[name].extend(renderer.render_complex_list(personal_trades, renderer.render_citations))

        event_asset = section_data.get("事件与资产")
        if event_asset:
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("事件与资产"))
                renderer_parts[name].extend(renderer.render_complex_list(event_asset, renderer.render_citations))

        market_knowledge = section_data.get("市场知识")
        if market_knowledge:
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("市场知识"))
                renderer_parts[name].extend(renderer.render_complex_list(market_knowledge, renderer.render_citations))

        # 渲染"其他"字段，放在主题的最后面
        other_items = section_data.get("其他")
        if other_items:
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("其他"))
                renderer_parts[name].extend(renderer.render_complex_list(other_items, renderer.render_citations))

    def _prepare_analysis(self, analysis: dict) -> dict:
        """
        准备分析数据：如果没有精炼总结则生成
        注意：除了生成TLDR之外，不应该修改原始analysis数据
        （render_save 入口的防御清理除外）
        """
        if "精炼总结" not in analysis:
            logger.info("生成精炼总结...")
            tldr = self._generate_tldr(analysis)
            analysis["精炼总结"] = tldr
        
        return analysis

    @staticmethod
    def _load_unresolved_jargons(output_dir: Path) -> dict[str, str]:
        """从 incremental_database/jargon_incremental.json 加载未解析黑话表。

        返回 {黑话: "个股黑话"|"板块黑话"}；文件缺失或无未解析项时返回空 dict。
        上下文消歧（puzzled）类未解析项不计入。
        """
        path = output_dir / "incremental_database" / "jargon_incremental.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        unresolved = data.get("未解析") or []
        result: dict[str, str] = {}
        for entry in unresolved:
            if not isinstance(entry, dict):
                continue
            jargon = entry.get("黑话")
            jargon_type = entry.get("类型")
            if not isinstance(jargon, str) or not jargon:
                continue
            if jargon_type in ("个股黑话", "板块黑话"):
                result[jargon] = jargon_type
        return result

    def _build_parts(
        self,
        analysis: dict,
        renderers: list[tuple[str, BaseRenderer]],
        author: str | None,
        title: str | None,
        bvid: str | None,
        *,
        economic: bool = False,
    ) -> dict[str, str]:
        """按渲染顺序一次遍历，产出每个渲染器的完整文本，返回 {渲染器名: 文本}。"""
        renderer_parts: dict[str, list[str]] = {name: [] for name, _ in renderers}

        for name, renderer in renderers:
            renderer_parts[name].extend(renderer.render_header(author, title, bvid))

        for section_key in self._get_render_order(analysis):
            section_data = analysis.get(section_key)
            if not section_data:
                continue
            if not isinstance(section_data, (dict, list)):
                continue

            if section_key == "精炼总结":
                self._render_tldr_section_parallel(
                    renderers, renderer_parts, section_data,
                    economic=economic,
                )
            elif section_key == "宏观分析":
                self._render_macro_section(renderers, renderer_parts, section_data)
            elif section_key == "主讲人暗示重要话题":
                self._render_sensitive_hints_section(renderers, renderer_parts, section_data)
            elif section_key == "交易技巧":
                self._render_trading_skills_section(renderers, renderer_parts, section_data)
            else:
                self._render_topic_section(renderers, renderer_parts, section_key, section_data)

        return {name: "".join(parts) for name, parts in renderer_parts.items()}

    @staticmethod
    def _export_pdf(html_file: Path, pdf_file: Path) -> None:
        """由 HTML 导出 PDF；weasyprint / fontTools 的噪声日志与直接打印一并抑制。

        失败只 warn——PDF 是附件形态的附属产物，不该拖垮整个渲染步骤。
        """
        try:
            # 惰性导入：仅 PDF 导出需要 weasyprint（Windows 依赖 GTK3 运行时），
            # 不在此处顶层导入，未安装时 ImportError 由下方 except 捕获、仅告警降级。
            from weasyprint import HTML

            original_levels = {}
            loggers_to_suppress = [
                'weasyprint', 'weasyprint.layout', 'weasyprint.css',
                'fontTools', 'fontTools.subset', 'fontTools.ttLib',
                'fontTools.merge', 'fontTools.feaLib', 'fontTools.pens',
                'fontTools.varLib', 'fontTools.colorLib', 'fontTools.misc'
            ]

            for logger_name in loggers_to_suppress:
                logger_obj = logging.getLogger(logger_name)
                original_levels[logger_name] = logger_obj.level
                logger_obj.setLevel(logging.ERROR)

            old_stderr = sys.stderr
            old_stdout = sys.stdout
            sys.stderr = io.StringIO()
            sys.stdout = io.StringIO()

            try:
                HTML(filename=str(html_file)).write_pdf(str(pdf_file))
                logger.info("PDF已保存: %s", pdf_file)
            finally:
                sys.stderr = old_stderr
                sys.stdout = old_stdout
                for logger_name, level in original_levels.items():
                    logging.getLogger(logger_name).setLevel(level)
        except Exception as e:
            logger.warning("PDF导出失败: %s", e)

    def render_save(self, output_dir: Path, analysis: dict, bvid: str | None = None, author: str | None = None, title: str | None = None, *, economic: bool = False) -> str:
        """生成全部总结文件，返回完整版 Markdown 内容。

        只生成四种产物：
        - summary.md（Markdown 全文）
        - summary.html（完整版 HTML）
        - summary.pdf（由 summary.html 经 weasyprint 导出，失败只告警不阻断）
        - summary_structured.html（结构化折叠式 HTML）
        """
        with step_logger("渲染"):
            rendering_dir = output_dir / ("rendering_economic" if economic else "rendering")
            rendering_dir.mkdir(parents=True, exist_ok=True)

            # 防御清理：历史缓存 llm_analysis.json（旧版打标保留的产物）短路渲染时，
            # 同样移除校验失败条目并剥掉失败字段，杜绝泄漏进任何渲染产物
            for rec in clean_analysis(analysis):
                logger.warning(
                    "渲染前清理校验失败条目：%s（%s）",
                    rec["路径"], rec["失败原因"] or "无",
                )

            # 准备分析数据：生成精炼总结
            analysis = self._prepare_analysis(analysis)

            # 加载未解析黑话表（用于渲染时给孤立黑话加 `个股黑话：xxx` / `板块黑话：xxx` 前缀）
            unresolved_jargons = self._load_unresolved_jargons(output_dir)

            summary_impl = SummaryRendererImpl(unresolved_jargons)
            gzh_impl = GongzhonghaoRendererImpl(unresolved_jargons)

            full_parts = self._build_parts(
                analysis,
                [('summary', summary_impl), ('gzh', gzh_impl)],
                author, title, bvid,
                economic=economic,
            )

            md_summary = full_parts['summary']
            summary_file = rendering_dir / "summary.md"
            with open(summary_file, "w", encoding="utf-8-sig") as f:
                f.write(md_summary)
            logger.info("总结已保存: %s", summary_file)

            full_html = GongzhonghaoRendererImpl.markdown_to_html(full_parts['gzh'])
            full_html_file = rendering_dir / "summary.html"
            with open(full_html_file, "w", encoding="utf-8-sig") as f:
                f.write(full_html)
            logger.info("完整版 HTML 已保存: %s", full_html_file)

            # 保存结构化折叠式 HTML（交互 / 视觉对齐 visual_summarize_mobile 小程序）
            video_url = f"https://www.bilibili.com/video/{bvid}" if bvid else None
            structured_html = render_structured_html(analysis, author=author, title=title, video_url=video_url)
            structured_file = rendering_dir / "summary_structured.html"
            with open(structured_file, "w", encoding="utf-8-sig") as f:
                f.write(structured_html)
            logger.info("结构化 HTML 已保存: %s", structured_file)

            self._export_pdf(full_html_file, rendering_dir / "summary.pdf")

            return md_summary

    def _render_tldr_section_parallel(self, renderers: list[tuple[str, BaseRenderer]], renderer_parts: dict[str, list[str]], tldr: dict, *, economic: bool = False) -> None:
        """渲染精炼总结章节"""
        def _decorate_items(items: list, decorator: BaseRenderer, field_decorators: dict) -> list:
            """给每个 dict item 的指定字段套装饰函数，返回浅拷贝后的新 list。"""
            new_items = []
            for it in items or []:
                if not isinstance(it, dict):
                    new_items.append(it)
                    continue
                copy = dict(it)
                for field, mode in field_decorators.items():
                    val = copy.get(field)
                    if not isinstance(val, str) or not val:
                        continue
                    if mode == "label":
                        copy[field] = decorator.decorate_topic_label(val)
                    else:
                        copy[field] = decorator.decorate_jargon(val)
                new_items.append(copy)
            return new_items

        for name, renderer in renderers:
            renderer_parts[name].append(renderer.render_section_title("精炼总结", is_first=True))

        market_overview = tldr.get("大盘概览")
        if market_overview and isinstance(market_overview, dict):
            overview_keys = ["今日市场", "宏观事件", "后市观点", "仓位建议"]
            if not economic:
                overview_keys.append("热点板块预测")
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("大盘概览"))
                for field_key in overview_keys:
                    field_val = market_overview.get(field_key)
                    renderer_parts[name].append(f"{field_key}：{field_val or '无'}\n\n")

        sensitive_hints = tldr.get("主讲人暗示重要话题")
        if sensitive_hints:
            # 表里只展示「板块 / 信号类型 / 原话总结」三列，其他字段（原话、锚定行号、引用等）
            # 留给主题段独立渲染——这里先做一次投影，否则 render_object_list 会把所有字段都列出来。
            tldr_hints = [
                {
                    "板块": rec.get("板块", []),
                    "信号类型": rec.get("信号类型", ""),
                    "原话总结": rec.get("原话总结", ""),
                }
                for rec in sensitive_hints
                if isinstance(rec, dict)
            ]
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("🔒 重要暗示"))
                lines = []
                for r in tldr_hints:
                    summary = r.get("原话总结", "").strip()
                    signal = r.get("信号类型", "").strip()
                    if summary and signal:
                        lines.append(f"- {summary}：{signal}\n")
                    elif summary:
                        lines.append(f"- {summary}\n")
                    elif signal:
                        lines.append(f"- {signal}\n")
                if lines:
                    renderer_parts[name].append("".join(lines) + "\n")

        short_ops = tldr.get("短期机会速览")
        if short_ops:
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("短期机会速览"))
                if not short_ops:
                    renderer_parts[name].append("无\n\n")
                    continue
                decorated = _decorate_items(short_ops, renderer, {"名称": "jargon"})
                renderer_parts[name].extend(renderer.render_object_list(decorated, column_order=["名称", "描述"]))

        opinion_rows = tldr.get("个股观点速览")
        if opinion_rows:
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("个股观点速览"))
                if not opinion_rows:
                    renderer_parts[name].append("无\n\n")
                    continue
                decorated = _decorate_items(opinion_rows, renderer, {"板块": "label", "名称": "jargon"})
                renderer_parts[name].extend(renderer.render_object_list(decorated, column_order=["板块", "名称", "观点"]))

        event_rows = tldr.get("个股事件速览")
        if event_rows:
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("个股事件速览"))
                if not event_rows:
                    renderer_parts[name].append("无\n\n")
                    continue
                decorated = _decorate_items(event_rows, renderer, {"板块": "label"})
                renderer_parts[name].extend(renderer.render_object_list(decorated, column_order=["板块", "事件", "涉及资产"]))

        knowledge = tldr.get("市场知识索引")
        if knowledge:
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("市场知识索引"))
                if not knowledge:
                    renderer_parts[name].append("无\n\n")
                    continue
                decorated = _decorate_items(knowledge, renderer, {"板块": "jargon"})
                renderer_parts[name].extend(renderer.render_object_list(decorated, column_order=["主题", "标题", "内容"]))

        tips = tldr.get("交易技巧摘要")
        if tips:
            for name, renderer in renderers:
                renderer_parts[name].append(renderer.render_subsection_title("交易技巧摘要"))
                if not tips:
                    renderer_parts[name].append("无\n\n")
                    continue
                renderer_parts[name].extend(renderer.render_object_list(tips))

        for name, renderer in renderers:
            renderer_parts[name].append("---\n\n")
