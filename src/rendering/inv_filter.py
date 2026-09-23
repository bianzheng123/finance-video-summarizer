"""投资机会（inv）未表态过滤规则，summary/gzh/bilibili 与 structured 渲染器共享。

规则出处：原 rendering_controller.SummaryRenderer 的静态方法。为避免
rendering_controller 与 rendering_html_structured 循环 import（controller
顶层 import structured），把规则提取为独立模块——两侧都只依赖本模块，
规则单一实现源。
"""

from typing import Any


def is_unstated(viewpoint: Any) -> bool:
    """判断观点字段是否为'未表态'或缺失。支持字符串和 `当前价格` 对象。"""
    if viewpoint is None:
        return True
    if isinstance(viewpoint, dict):
        return is_unstated(viewpoint.get("观点"))
    if not isinstance(viewpoint, str):
        return True
    return not viewpoint.strip() or "未表态" in viewpoint


def inv_event_titles(inv: dict, events_and_assets: list) -> list[str]:
    """规则 3 辅助：返回 inv.标题 命中的 事件与资产 标题列表。

    精确匹配 `涉及资产` 成员，不用子串回退——子串会让「军工」误命中
    「国防军工」、「中国稀土」误命中「中国稀土集团」，把事件挂到不属于它的行上。
    """
    title = inv.get("标题", "") if isinstance(inv, dict) else ""
    if not title:
        return []
    hits: list[str] = []
    for ev in events_and_assets or []:
        if not isinstance(ev, dict):
            continue
        related = ev.get("涉及资产") or []
        if title in related:
            et = ev.get("标题", "")
            if et:
                hits.append(et)
    return hits


def inv_should_skip(inv: dict, event_titles_for_inv: list[str]) -> bool:
    """规则 3：事件无 + 所有观点皆未表态 → 整 inv 砍掉。"""
    if event_titles_for_inv:
        return False
    short_term = inv.get("短线") if isinstance(inv.get("短线"), dict) else {}
    medium_long = inv.get("中线") or inv.get("长线") or inv.get("中长线")
    if not isinstance(medium_long, dict):
        medium_long = {}
    short_unstated = is_unstated(short_term.get("观点"))
    long_unstated = is_unstated(medium_long.get("观点"))
    price_viewpoint = (inv.get("当前价格") or {}).get("观点")
    price_unstated = is_unstated(price_viewpoint)
    return short_unstated and long_unstated and price_unstated


def inv_has_substance(inv: dict) -> bool:
    """检查投资机会项是否有实质性内容（非未表态的观点或价格）。

    用于渲染前过滤：如果一个项的所有观点都是未表态且价格也未表态，
    则它没有实质性内容，不应该被渲染在"投资机会"章节中。
    """
    # 检查是否有非未表态的观点（短线/中线/长线/中长线）
    for tk in ("短线", "中长线", "中线", "长线"):
        tv = inv.get(tk)
        if isinstance(tv, dict) and not is_unstated(tv.get("观点")):
            return True

    # 检查当前价格是否非未表态（该字段是嵌套字典，需要获取其中的"观点"）
    price_viewpoint = (inv.get("当前价格") or {}).get("观点")
    if not is_unstated(price_viewpoint):
        return True

    return False
