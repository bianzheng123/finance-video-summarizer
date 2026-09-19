"""结构化折叠式 HTML 渲染器。

把 llm_analysis 的动态结构渲染成一个自包含（内联 CSS + 原生 JS）的 HTML 页面，
交互 / 视觉与 visual_summarize_mobile 小程序的 summary 页保持一致：
卡片布局 + 多层折叠 + 观点色块 + 原文引用折叠。

数据归一化逻辑对齐 miniprogram/utils/transform.js；折叠交互靠 `toggleFold`
切换父节点的 `open` class 实现，无任何外部依赖。
"""

import html
from typing import Any

from .inv_filter import inv_event_titles, inv_has_substance, inv_should_skip, is_unstated

# 非板块的特殊顶层键（不当作「板块」渲染）。
# 相比 transform.js 额外排除 `精炼总结`——render_save 会把它注入 analysis。
SPECIAL_KEYS = {"交易技巧", "宏观分析", "主讲人暗示重要话题", "精炼总结"}


def _esc(value: Any) -> str:
    """HTML 转义，None → 空串。"""
    return html.escape(str(value)) if value not in (None, "") else ""


# ==================== 数据归一化（对齐 transform.js） ====================

def _parse_opinion(raw: str | None) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None
    tone = "neutral"
    if "看多" in text or "看涨" in text:
        tone = "bull"
    elif "看空" in text or "看跌" in text:
        tone = "bear"
    elif "便宜" in text:
        tone = "cheap"
    elif "中性" in text or "未表态" in text:
        tone = "neutral"
    return {"text": text, "tone": tone}


def _norm_quotes(quotes: list[dict] | None) -> list[dict]:
    if not isinstance(quotes, list):
        return []
    result = []
    for q in quotes:
        if not isinstance(q, dict):
            continue
        text = (q.get("原文") or "").strip()
        if not text:
            continue
        result.append({
            "text": text,
            "start": q.get("开始时间") or "",
            "end": q.get("结束时间") or "",
        })
    return result


def _norm_stance(label: str, stance: dict | None) -> dict | None:
    if not isinstance(stance, dict):
        return None
    opinion = _parse_opinion(stance.get("观点"))
    if not opinion:
        return None
    # 只保留明确观点（看多/看空/便宜），未表态不显示
    if opinion["tone"] == "neutral":
        return None
    return {
        "label": label,
        "text": opinion["text"],
        "tone": opinion["tone"],
        "reason": (stance.get("逻辑") or "").strip(),
        "quotes": _norm_quotes(stance.get("引用")),
    }


def _norm_generic_items(items: list, title_key: str | None = None) -> list[dict]:
    """通用 dict 列表归一化（对齐 gzh render_complex_list 语义）。

    title_key 有值时取该字段作条目标题；其余键（跳过 引用/原始黑话）
    转为 {label, text} 行——str 直出、list 顿号串联、其余 str() 兜底。
    """
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = (item.get(title_key) or "").strip() if title_key else ""
        fields: list[dict] = []
        for key, val in item.items():
            if key == title_key or key == "引用" or key == "原始黑话":
                continue
            if isinstance(val, str):
                text = val.strip()
            elif isinstance(val, list) and val:
                text = "、".join(str(x) for x in val)
            elif val is not None:
                text = str(val)
            else:
                continue
            if not text:
                continue
            fields.append({"label": key, "text": text})
        quotes = _norm_quotes(item.get("引用"))
        if not title and not fields and not quotes:
            continue
        result.append({"title": title, "fields": fields, "quotes": quotes})
    return result


def _format_duration(seconds: int | float) -> str:
    if not seconds:
        return ""
    mins = int(seconds // 60)
    secs = round(seconds % 60)
    if mins > 0 and secs > 0:
        return f"{mins}分{secs}秒"
    if mins > 0:
        return f"{mins}分"
    if secs > 0:
        return f"{secs}秒"
    return ""


def _build_sectors(analysis: dict) -> list[dict]:
    groups = []
    for sector, node in analysis.items():
        if sector in SPECIAL_KEYS or not isinstance(node, dict):
            continue

        ref_times = []
        for t in node.get("引用时间") or []:
            if not isinstance(t, dict):
                continue
            ref_times.append({"start": t.get("开始时间") or "", "end": t.get("结束时间") or ""})

        opportunities = []
        for item in node.get("投资机会") or []:
            if not isinstance(item, dict):
                continue
            # 未表态过滤规则与 gzh 渲染路径对齐（inv_filter 单一实现源）
            event_titles = inv_event_titles(item, node.get("事件与资产") or [])
            if inv_should_skip(item, event_titles):
                continue
            cleaned = dict(item)
            for tk in ("短线", "中长线", "中线", "长线"):
                tv = cleaned.get(tk)
                if isinstance(tv, dict) and is_unstated(tv.get("观点")):
                    cleaned.pop(tk, None)
            price_viewpoint = (cleaned.get("当前价格") or {}).get("观点")
            if is_unstated(price_viewpoint):
                cleaned.pop("当前价格", None)
            if not inv_has_substance(cleaned):
                continue
            stances = [
                _norm_stance("短线", cleaned.get("短线")),
                _norm_stance("中长线", cleaned.get("中长线")),
                _norm_stance("中线", cleaned.get("中线")),
                _norm_stance("长线", cleaned.get("长线")),
                _norm_stance("当前价格", cleaned.get("当前价格")),
            ]
            stances = [s for s in stances if s]
            if not stances:
                continue
            opportunities.append({"title": item.get("标题") or sector, "stances": stances, "opinions": []})

        events = []
        for item in node.get("事件与资产") or []:
            if not isinstance(item, dict):
                continue
            assets = item.get("涉及资产")
            events.append({
                "title": item.get("标题") or "",
                "assets": "、".join(assets) if isinstance(assets, list) else "",
                "impact": (item.get("影响与结果") or "").strip(),
                "logic": (item.get("逻辑") or "").strip(),
                "basisType": (item.get("依据类型") or "").strip(),
                "quotes": _norm_quotes(item.get("引用")),
            })

        knowledge = []
        for k in node.get("市场知识") or []:
            if not isinstance(k, dict):
                continue
            knowledge.append({
                "title": k.get("标题") or "",
                "content": (k.get("内容") or "").strip(),
                "quotes": _norm_quotes(k.get("引用")),
            })

        opinions = []
        for op in node.get("观点") or []:
            if not isinstance(op, dict):
                continue
            chain = []
            for link in op.get("因果链") or []:
                if not isinstance(link, dict):
                    continue
                content = (link.get("内容") or "").strip()
                if not content:
                    continue
                chain.append({
                    "type": link.get("环节类型") or "",
                    "content": content,
                    "quotes": _norm_quotes(link.get("引用")),
                })
            opinions.append({
                "name": op.get("标的名称") or "",
                "text": (op.get("观点") or "").strip(),
                "chain": chain,
                "quotes": _norm_quotes(op.get("引用")),
            })

        # 投资机会与观点合并：按资产名（标题 == 标的名称）分组，对齐 summary 的
        # _build_asset_groups 精确相等口径；未匹配的观点独立成组（空 stances）。
        by_title = {o["title"]: o for o in opportunities}
        for op in opinions:
            target = by_title.get(op["name"])
            if target is not None:
                target["opinions"].append(op)
            else:
                opportunities.append({"title": op["name"], "stances": [], "opinions": [op]})

        trades = _norm_generic_items(node.get("个人交易记录"), title_key="标的名称")
        other = _norm_generic_items(node.get("其他"), title_key=None)

        if not (ref_times or opportunities or events or knowledge or trades or other):
            continue

        duration = node.get("引用总时长") or 0
        groups.append({
            "sector": sector,
            "duration": _format_duration(duration),
            "durationRaw": duration,
            "refTimes": ref_times,
            "opportunities": opportunities,
            "events": events,
            "knowledge": knowledge,
            "trades": trades,
            "other": other,
        })

    groups.sort(key=lambda g: g["durationRaw"], reverse=True)
    return groups


def _build_tips(analysis: dict) -> list[dict]:
    tips = analysis.get("交易技巧")
    if not isinstance(tips, dict):
        return []
    result = []
    for name, t in tips.items():
        t = t if isinstance(t, dict) else {}
        result.append({
            "name": name,
            "method": (t.get("具体方法") or "").strip(),
            "scope": (t.get("适用范围或前提") or "").strip(),
            "quotes": _norm_quotes(t.get("引用")),
        })
    return result


def _build_overview(analysis: dict) -> dict | None:
    macro = analysis.get("宏观分析")
    if not isinstance(macro, dict):
        return None

    sector_node = macro.get("板块分析")
    fund_node = sector_node.get("资金流向") if isinstance(sector_node, dict) else None

    def list_section(title: str, node: Any) -> dict | None:
        """分块字段：逐条保留「标题 + 观点 + 引用」，不再拍平成一行速览。

        兼容两种形态：list（真实数据，每 block 标题+观点+引用）与 dict（单条，
        观点+引用），后者等价于单条目 section。
        """
        if isinstance(node, dict):
            text = (node.get("观点") or "").strip()
            quotes = _norm_quotes(node.get("引用"))
            if not text and not quotes:
                return None
            return {
                "title": title, "kind": "list",
                "items": [{"title": "", "text": text, "quotes": quotes}],
            }
        if not isinstance(node, list):
            return None
        items = []
        for block in node:
            if not isinstance(block, dict):
                continue
            item_title = (block.get("标题") or "").strip()
            item_text = (block.get("观点") or "").strip()
            quotes = _norm_quotes(block.get("引用"))
            if not item_title and not item_text and not quotes:
                continue
            items.append({"title": item_title, "text": item_text, "quotes": quotes})
        if not items:
            return None
        return {"title": title, "kind": "list", "items": items}

    sections = [
        list_section("大盘情绪与状态判断", macro.get("大盘情绪与状态判断")),
        list_section("宏观经济与事件影响分析", macro.get("宏观经济与事件影响分析")),
        list_section("后市观点", macro.get("后市观点")),
        list_section("仓位建议", macro.get("仓位建议")),
    ]

    # 资金流向（dict：观点 + 引用）作为单条目 section，与其它字段保持同一逐条 UI
    if isinstance(fund_node, dict):
        fund_text = (fund_node.get("观点") or "").strip()
        fund_quotes = _norm_quotes(fund_node.get("引用"))
        if fund_text or fund_quotes:
            sections.append({
                "title": "资金流向",
                "kind": "list",
                "items": [{"title": "", "text": fund_text, "quotes": fund_quotes}],
            })

    # 热点板块预测：先总览，再逐板块条目
    hot_node = macro.get("热点板块预测")
    hot_summary = (hot_node.get("总览") or "").strip() if isinstance(hot_node, dict) else ""
    hot_items = []
    if isinstance(hot_node, dict) and isinstance(hot_node.get("板块清单"), list):
        for s in hot_node["板块清单"]:
            if not isinstance(s, dict) or not s.get("板块名称"):
                continue
            hot_items.append({
                "name": s.get("板块名称") or "",
                "tendency": s.get("预测倾向") or "",
                "window": s.get("时间窗口") or "",
                "condition": s.get("条件/前置事件") or "",
                "quotes": _norm_quotes(s.get("引用")),
            })
    if hot_summary or hot_items:
        sections.append({
            "title": "热点板块预测",
            "kind": "hot",
            "summary": hot_summary,
            "items": hot_items,
        })

    sections = [s for s in sections if s]
    if not sections:
        return None
    return {"sections": sections}


def _build_short_opps(analysis: dict) -> list[dict]:
    opps = []
    for sector, node in analysis.items():
        if sector in SPECIAL_KEYS or not isinstance(node, dict):
            continue
        for item in node.get("投资机会") or []:
            if not isinstance(item, dict):
                continue
            short_term = item.get("短线")
            if not isinstance(short_term, dict):
                continue
            opinion = _parse_opinion(short_term.get("观点"))
            if not opinion or opinion["tone"] != "bull":
                continue
            opps.append({
                "asset": item.get("标题") or sector,
                "sector": sector,
                "text": opinion["text"],
                "reason": (short_term.get("逻辑") or "").strip(),
                "quotes": _norm_quotes(short_term.get("引用")),
            })
    return opps


def _build_hints(analysis: dict) -> list[dict]:
    hints = analysis.get("主讲人暗示重要话题")
    if not isinstance(hints, list):
        return []
    result = []
    for h in hints:
        if not isinstance(h, dict):
            continue
        entry = {
            "type": h.get("信号类型") or "",
            "summary": h.get("原话总结") or "",
            "quotes": _norm_quotes(h.get("引用")),
        }
        if entry["summary"] or entry["type"]:
            result.append(entry)
    return result


def transform_analysis(analysis: dict) -> dict:
    """把 analysis 归一化成前端易渲染的分区数据（对齐 transform.js）。

    注意：`精炼总结`（TLDR）虽由渲染入口注入 analysis，但本页面不再渲染它
    （内容与独立分区重复）；其数据仍被 gzh / bilibili 渲染路径使用，故这里
    只是不投影到前端视图，不影响数据生成管线。
    """
    if not isinstance(analysis, dict):
        return {"overview": None, "hints": [], "shortOpps": [], "sectors": [], "tips": []}
    return {
        "overview": _build_overview(analysis),
        "hints": _build_hints(analysis),
        "shortOpps": _build_short_opps(analysis),
        "sectors": _build_sectors(analysis),
        "tips": _build_tips(analysis),
    }


# ==================== HTML 渲染 ====================

# 折叠交互：点击 .fold-head 切换父 .fold 的 open 状态；纯原生，无依赖。
_PAGE_STYLE = """
:root {
  --bull:#e03b3b; --bear:#1a9850; --neutral:#8a8f99; --primary:#2b7bba;
  --cheap:#2b7bba; --bg:#f4f5f7; --card-bg:#fff; --text-main:#1f2329;
  --text-sub:#646a73; --border:#eaecf0;
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--bg); color:var(--text-main);
  font-size:15px; line-height:1.6;
  font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
}
.page { max-width:720px; margin:0 auto; padding:12px 12px 80px; }
.card {
  background:var(--card-bg); border-radius:12px; padding:16px;
  margin-bottom:14px; box-shadow:0 1px 6px rgba(0,0,0,.05);
}
.header-title { font-size:21px; font-weight:700; margin-bottom:10px; }
.header-meta { display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:10px; }
.tag { font-size:12px; color:var(--text-sub); background:var(--bg); padding:2px 10px; border-radius:5px; }
.header-date { font-size:12px; color:var(--neutral); }
.disclaimer { font-size:12px; color:var(--neutral); border-top:1px solid var(--border); padding-top:10px; }
.section-title {
  font-size:20px; font-weight:600; margin:20px 4px 10px;
  padding-left:10px; border-left:5px solid #f4b183;
}
/* 折叠通用 */
.fold-head {
  display:flex; justify-content:space-between; align-items:center;
  cursor:pointer; user-select:none;
}
.arrow { flex-shrink:0; color:var(--primary); margin-left:10px; transition:transform .15s; }
.fold.open > .fold-head .arrow { transform:rotate(90deg); }
.fold > .fold-body { display:none; }
.fold.open > .fold-body { display:block; }
/* 概览 */
.ov-block { padding-bottom:12px; margin-bottom:12px; border-bottom:1px solid var(--border); }
.ov-block:last-child { padding-bottom:0; margin-bottom:0; border-bottom:none; }
.ov-head { display:flex; align-items:center; padding-left:10px; border-left:4px solid #f4b183; margin-bottom:8px; }
.ov-subtitle { font-size:16px; font-weight:700; color:var(--text-main); }
.ov-item { padding-bottom:10px; margin-bottom:10px; border-bottom:1px solid var(--border); }
.ov-item:last-child { padding-bottom:0; margin-bottom:0; border-bottom:none; }
.ov-item-title { font-size:14px; font-weight:600; margin-bottom:2px; }
.ov-item-text { font-size:14px; }
.hot-summary { font-size:13px; margin:8px 0; }
.hot-item { padding-bottom:10px; margin-bottom:10px; border-bottom:1px solid var(--border); }
.hot-item:last-child { border-bottom:none; margin-bottom:0; padding-bottom:0; }
.hot-head { display:flex; align-items:center; flex-wrap:wrap; gap:8px; }
.hot-name { font-size:14px; }
.hot-tendency { font-size:12px; color:var(--bull); background:rgba(224,59,59,.1); padding:2px 10px; border-radius:5px; }
.hot-window { font-size:12px; color:var(--text-sub); background:var(--bg); padding:2px 10px; border-radius:5px; }
.hot-cond { font-size:13px; color:var(--text-sub); margin-top:6px; }
/* 重要暗示 */
.hint-card { background:#fff8e6; }
.hint-item { margin-bottom:12px; }
.hint-item:last-child { margin-bottom:0; }
.hint-line { display:flex; align-items:baseline; gap:8px; }
.hint-tag { flex-shrink:0; font-size:12px; color:#b8860b; background:rgba(184,134,11,.12); padding:2px 10px; border-radius:5px; }
.hint-summary { font-size:13px; }
/* 短期机会 */
.opp-item { padding-bottom:12px; margin-bottom:12px; border-bottom:1px solid var(--border); }
.opp-item:last-child { border-bottom:none; margin-bottom:0; padding-bottom:0; }
.opp-head { display:flex; align-items:center; flex-wrap:wrap; gap:8px; }
.opp-asset { font-size:15px; font-weight:600; }
.opp-tag { font-size:12px; color:var(--bull); background:rgba(224,59,59,.1); padding:2px 10px; border-radius:5px; }
.opp-reason { font-size:13px; color:var(--text-sub); margin-top:6px; }
/* 板块 */
.sector-card { padding:0; overflow:hidden; }
.sector-head { padding:14px 16px; align-items:flex-start; }
.sector-info { flex:1; min-width:0; }
.sector-name { display:flex; align-items:center; gap:8px; font-size:16px; font-weight:600; }
.sector-dur { font-size:12px; color:var(--neutral); font-weight:400; }
.sector-counts { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
.count-chip { font-size:12px; color:var(--text-sub); background:var(--bg); padding:2px 10px; border-radius:5px; }
.sector-body { padding:0 16px 6px; }
.sub { border-top:1px solid var(--border); }
.sub-head { padding:12px 0; }
.sub-label { font-size:14px; font-weight:600; }
.reftime-list { display:flex; flex-wrap:wrap; gap:8px; padding-bottom:8px; }
.reftime-chip { font-size:12px; color:var(--text-sub); background:var(--bg); padding:2px 10px; border-radius:5px; }
.stock { padding:8px 0; }
.stock-title { font-size:15px; font-weight:600; margin-bottom:8px; }
.stance { margin-bottom:10px; }
.stance-reason { font-size:13px; color:var(--text-sub); margin-top:4px; }
.opinion { padding:4px 0; }
.opinion-text { font-size:14px; }
.opinion-label { display:inline-block; font-size:12px; color:var(--neutral); background:var(--bg); padding:1px 8px; border-radius:4px; margin-right:6px; }
.chain-link { font-size:13px; color:var(--text-sub); margin-top:3px; }
.chain-type { font-weight:600; color:var(--text-sub); margin-right:6px; }
.badge { display:inline-flex; align-items:center; padding:3px 10px; border-radius:6px; font-size:12px; margin:0 8px 4px 0; }
.badge-label { font-weight:600; margin-right:6px; opacity:.85; }
.badge-bull { background:rgba(224,59,59,.12); color:var(--bull); }
.badge-bear { background:rgba(26,152,80,.12); color:var(--bear); }
.badge-cheap { background:rgba(43,123,186,.12); color:var(--primary); }
.badge-neutral { background:rgba(138,143,153,.14); color:var(--neutral); }
.event, .knowledge { padding:8px 0; border-top:1px dashed var(--border); }
.event:first-of-type, .knowledge:first-of-type { border-top:none; }
.event-title, .knowledge-title { font-size:14px; font-weight:600; margin-bottom:4px; }
.event-assets { font-size:12px; color:var(--primary); margin-bottom:3px; }
.event-impact { font-size:14px; margin-bottom:3px; }
.event-logic { font-size:13px; color:var(--text-sub); margin-bottom:3px; }
.event-basis { display:inline-block; font-size:12px; color:var(--neutral); background:var(--bg); padding:1px 10px; border-radius:5px; }
.knowledge-content { font-size:14px; }
/* 技巧 */
.tip-name { flex:1; min-width:0; font-size:15px; font-weight:600; }
.tip-body { margin-top:10px; }
.tip-method { font-size:14px; }
.tip-scope { font-size:12px; color:var(--text-sub); background:var(--bg); padding:8px 12px; border-radius:6px; margin-top:8px; }
/* 引用 */
.quote-wrap { display:inline; }
.quote-toggle { display:inline-flex; align-items:center; gap:4px; margin-left:8px; font-size:12px; color:var(--primary); cursor:pointer; user-select:none; vertical-align:middle; }
.quote-list { display:block; margin-top:8px; padding-left:10px; border-left:2px solid var(--border); }
.quote-item { margin-bottom:10px; }
.quote-item:last-child { margin-bottom:0; }
.quote-time { font-size:12px; color:var(--neutral); margin-bottom:2px; }
.quote-text { font-size:13px; color:var(--text-sub); background:var(--bg); padding:8px 12px; border-radius:6px; }
/* 个人交易记录 / 其他 */
.trade, .other-item { padding:8px 0; border-top:1px dashed var(--border); }
.trade:first-of-type, .other-item:first-of-type { border-top:none; }
.trade-name { font-size:14px; font-weight:600; margin-bottom:4px; }
.trade-field, .other-field { font-size:13px; margin-top:3px; }
"""

_PAGE_SCRIPT = """
document.addEventListener('click', function (e) {
  var head = e.target.closest('.fold-head, .quote-toggle');
  if (!head) return;
  var fold = head.closest('.fold');
  if (fold) fold.classList.toggle('open');
});
"""


def _arrow() -> str:
    return '<span class="arrow">▸</span>'


def _quote_card(quotes: list[dict]) -> str:
    if not quotes:
        return ""
    items = []
    for q in quotes:
        time_html = ""
        if q["start"]:
            end = f" - {_esc(q['end'])}" if q["end"] else ""
            time_html = f'<div class="quote-time">{_esc(q["start"])}{end}</div>'
        items.append(f'{time_html}<div class="quote-text">{_esc(q["text"])}</div>')
    body = "".join(f'<div class="quote-item">{it}</div>' for it in items)
    return (
        '<div class="quote-wrap fold">'
        f'<div class="quote-toggle fold-head"><span class="arrow">▸</span>'
        f'<span>原文引用 ({len(quotes)})</span></div>'
        f'<div class="fold-body quote-list">{body}</div>'
        '</div>'
    )


def _render_overview(overview: dict | None) -> str:
    if not overview:
        return ""
    blocks = []
    for sec in overview["sections"]:
        if sec["kind"] == "hot":
            summary = f'<div class="hot-summary">{_esc(sec["summary"])}</div>' if sec.get("summary") else ""
            items = "".join(
                '<div class="hot-item">'
                '<div class="hot-head">'
                f'<span class="hot-name">{_esc(h["name"])}</span>'
                + (f'<span class="hot-tendency">{_esc(h["tendency"])}</span>' if h["tendency"] else "")
                + (f'<span class="hot-window">{_esc(h["window"])}</span>' if h["window"] else "")
                + ("" if h["condition"] else _quote_card(h["quotes"]))
                + '</div>'
                + (f'<div class="hot-cond">前置：{_esc(h["condition"])}{_quote_card(h["quotes"])}</div>' if h["condition"] else "")
                + '</div>'
                for h in sec["items"]
            )
            body = summary + items
        else:
            body = "".join(
                '<div class="ov-item">'
                + (f'<div class="ov-item-title">{_esc(it["title"])}</div>' if it["title"] else "")
                + f'<div class="ov-item-text">{_esc(it["text"])}{_quote_card(it["quotes"])}</div>'
                + '</div>'
                for it in sec["items"]
            )
        blocks.append(
            '<div class="ov-block">'
            f'<div class="ov-head"><span class="ov-subtitle">{_esc(sec["title"])}</span></div>'
            f'{body}'
            '</div>'
        )
    return ('<div class="section-title">大盘概览</div>'
            f'<div class="card overview">{"".join(blocks)}</div>')


def _render_hints(hints: list[dict]) -> str:
    if not hints:
        return ""
    items = "".join(
        '<div class="hint-item">'
        '<div class="hint-line">'
        f'<span class="hint-tag">{_esc(h["type"])}</span>'
        f'<div class="hint-summary">{_esc(h["summary"])}{_quote_card(h["quotes"])}</div></div>'
        '</div>'
        for h in hints
    )
    return ('<div class="section-title">🔒 重要暗示</div>'
            f'<div class="card hint-card">{items}</div>')


def _render_short_opps(short_opps: list[dict]) -> str:
    if not short_opps:
        return ""
    items = "".join(
        '<div class="opp-item">'
        '<div class="opp-head">'
        f'<span class="opp-asset">{_esc(o["asset"])}</span>'
        '<span class="opp-tag">短线看多</span>'
        + ("" if o["reason"] else _quote_card(o["quotes"]))
        + '</div>'
        + (f'<div class="opp-reason">{_esc(o["reason"])}{_quote_card(o["quotes"])}</div>' if o["reason"] else "")
        + '</div>'
        for o in short_opps
    )
    return ('<div class="section-title">⚡ 短期机会</div>'
            f'<div class="card opp-card">{items}</div>')


def _render_stances(stances: list[dict]) -> str:
    out = []
    for st in stances:
        label = f'<span class="badge-label">{_esc(st["label"])}</span>' if st["label"] else ""
        reason = (
            f'<div class="stance-reason">{_esc(st["reason"])}{_quote_card(st["quotes"])}</div>'
            if st["reason"] else ""
        )
        out.append(
            '<div class="stance">'
            f'<span class="badge badge-{st["tone"]}">{label}<span class="badge-text">{_esc(st["text"])}</span></span>'
            f'{reason}{"" if st["reason"] else _quote_card(st["quotes"])}</div>'
        )
    return "".join(out)


def _render_opinion(op: dict) -> str:
    """渲染单个观点条目：中性「观点」标签 + 观点文本 + 因果链 + 引用。"""
    text = (
        f'<span class="opinion-label">观点</span>{_esc(op["text"])}{_quote_card(op["quotes"])}'
        if op["text"] else ""
    )
    return (
        '<div class="opinion">'
        + (f'<div class="opinion-text">{text}</div>' if text else "")
        + "".join(
            f'<div class="chain-link"><span class="chain-type">{_esc(link["type"])}</span>'
            f'{_esc(link["content"])}{_quote_card(link["quotes"])}</div>'
            for link in op["chain"]
        )
        + '</div>'
    )


def _sub_fold(label: str, count: int, body: str, open_by_default: bool = False) -> str:
    cls = "sub fold open" if open_by_default else "sub fold"
    return (
        f'<div class="{cls}">'
        f'<div class="sub-head fold-head"><span class="sub-label">{_esc(label)} ({count})</span>{_arrow()}</div>'
        f'<div class="fold-body">{body}</div></div>'
    )


def _render_sector(s: dict) -> str:
    counts = []
    if s["opportunities"]:
        counts.append(f'<span class="count-chip">投资机会与观点 {len(s["opportunities"])}</span>')
    if s["events"]:
        counts.append(f'<span class="count-chip">事件与资产 {len(s["events"])}</span>')
    if s["knowledge"]:
        counts.append(f'<span class="count-chip">市场知识 {len(s["knowledge"])}</span>')
    if s["trades"]:
        counts.append(f'<span class="count-chip">个人交易记录 {len(s["trades"])}</span>')
    if s["other"]:
        counts.append(f'<span class="count-chip">其他 {len(s["other"])}</span>')
    if s["refTimes"]:
        counts.append(f'<span class="count-chip">引用时间 {len(s["refTimes"])}</span>')

    body_parts = []
    if s["opportunities"]:
        stocks = "".join(
            '<div class="stock">'
            + (f'<div class="stock-title">{_esc(o["title"])}</div>' if o["title"] else "")
            + _render_stances(o["stances"])
            + "".join(_render_opinion(op) for op in o["opinions"])
            + '</div>'
            for o in s["opportunities"]
        )
        body_parts.append(_sub_fold("投资机会与观点", len(s["opportunities"]), stocks))
    if s["events"]:
        evts = "".join(
            '<div class="event">'
            + (
                f'<div class="event-title">{_esc(e["title"])}{_quote_card(e["quotes"])}</div>'
                if not e["impact"] and not e["logic"]
                else f'<div class="event-title">{_esc(e["title"])}</div>'
            )
            + (f'<div class="event-assets">涉及：{_esc(e["assets"])}</div>' if e["assets"] else "")
            + (
                f'<div class="event-impact">{_esc(e["impact"])}{_quote_card(e["quotes"])}</div>'
                if e["impact"] and not e["logic"]
                else (f'<div class="event-impact">{_esc(e["impact"])}</div>' if e["impact"] else "")
            )
            + (f'<div class="event-logic">逻辑：{_esc(e["logic"])}{_quote_card(e["quotes"])}</div>' if e["logic"] else "")
            + (f'<span class="event-basis">{_esc(e["basisType"])}</span>' if e["basisType"] else "")
            + '</div>'
            for e in s["events"]
        )
        body_parts.append(_sub_fold("事件与资产", len(s["events"]), evts))
    if s["knowledge"]:
        knows = "".join(
            '<div class="knowledge">'
            + (f'<div class="knowledge-title">{_esc(k["title"])}</div>' if k["title"] else "")
            + f'<div class="knowledge-content">{_esc(k["content"])}{_quote_card(k["quotes"])}</div>'
            + '</div>'
            for k in s["knowledge"]
        )
        body_parts.append(_sub_fold("市场知识", len(s["knowledge"]), knows))
    if s["trades"]:
        trades_html = "".join(
            '<div class="trade">'
            + (f'<div class="trade-name">{_esc(t["title"])}</div>' if t["title"] else "")
            + "".join(
                f'<div class="trade-field">{_esc(f["label"])}：{_esc(f["text"])}'
                + (_quote_card(t["quotes"]) if i == len(t["fields"]) - 1 else "")
                + '</div>'
                for i, f in enumerate(t["fields"])
            )
            + ("" if t["fields"] else _quote_card(t["quotes"]))
            + '</div>'
            for t in s["trades"]
        )
        body_parts.append(_sub_fold("个人交易记录", len(s["trades"]), trades_html))
    if s["other"]:
        others_html = "".join(
            '<div class="other-item">'
            + "".join(
                '<div class="other-field">'
                + (f'{_esc(f["label"])}：' if f["label"] else "")
                + _esc(f["text"])
                + (_quote_card(o["quotes"]) if i == len(o["fields"]) - 1 else "")
                + '</div>'
                for i, f in enumerate(o["fields"])
            )
            + ("" if o["fields"] else _quote_card(o["quotes"]))
            + '</div>'
            for o in s["other"]
        )
        body_parts.append(_sub_fold("其他", len(s["other"]), others_html))
    if s["refTimes"]:
        chips = "".join(
            f'<span class="reftime-chip">{_esc(t["start"])}{("-" + _esc(t["end"])) if t["end"] else ""}</span>'
            for t in s["refTimes"]
        )
        body_parts.append(_sub_fold("引用时间", len(s["refTimes"]), f'<div class="reftime-list">{chips}</div>'))

    dur = f'<span class="sector-dur">{_esc(s["duration"])}</span>' if s["duration"] else ""
    return (
        '<div class="card sector-card fold">'
        '<div class="sector-head fold-head">'
        '<div class="sector-info">'
        f'<div class="sector-name"><span>{_esc(s["sector"])}</span>{dur}</div>'
        f'<div class="sector-counts">{"".join(counts)}</div></div>'
        f'{_arrow()}</div>'
        f'<div class="fold-body sector-body">{"".join(body_parts)}</div>'
        '</div>'
    )


def _render_tips(tips: list[dict]) -> str:
    if not tips:
        return ""
    cards = "".join(
        '<div class="card tip-card fold">'
        f'<div class="tip-head fold-head"><span class="tip-name">{_esc(t["name"])}</span>{_arrow()}</div>'
        '<div class="fold-body tip-body">'
        f'<div class="tip-method">{_esc(t["method"])}'
        + ("" if t["scope"] else _quote_card(t["quotes"]))
        + '</div>'
        + (f'<div class="tip-scope">适用：{_esc(t["scope"])}{_quote_card(t["quotes"])}</div>' if t["scope"] else "")
        + '</div></div>'
        for t in tips
    )
    return f'<div class="section-title">交易技巧</div>{cards}'


def render_structured_html(analysis: dict, author: str | None = None,
                           title: str | None = None, video_url: str | None = None) -> str:
    """把 analysis 渲染成自包含的折叠式结构化 HTML 字符串。"""
    view = transform_analysis(analysis)

    meta_parts = []
    if author:
        meta_parts.append(f'<span class="tag">{_esc(author)}</span>')
    if video_url:
        meta_parts.append(f'<a class="header-url" href="{_esc(video_url)}">🔗 原视频链接</a>')

    header = (
        '<div class="header card">'
        f'<div class="header-title">{_esc(title or "视频总结")}</div>'
        f'<div class="header-meta">{"".join(meta_parts)}</div>'
        '<div class="disclaimer">本文仅为内容总结，不能完全反映原视频观点，不构成投资建议。</div>'
        '</div>'
    )

    sectors_html = "".join(_render_sector(s) for s in view["sectors"])
    body = (
        header
        + _render_overview(view["overview"])
        + _render_hints(view["hints"])
        + _render_short_opps(view["shortOpps"])
        + f'<div class="section-title">板块总结（{len(view["sectors"])}）</div>{sectors_html}'
        + _render_tips(view["tips"])
    )

    return (
        '<!DOCTYPE html><html lang="zh-CN"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{_esc(title or "视频总结")}</title>'
        f'<style>{_PAGE_STYLE}</style></head>'
        f'<body><div class="page">{body}</div>'
        f'<script>{_PAGE_SCRIPT}</script>'
        '</body></html>'
    )
