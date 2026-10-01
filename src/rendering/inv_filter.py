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


def stance_has_substance(stance: Any) -> bool:
    """单个时间维度是否有实质内容：观点非「未表态」即视为有实质内容。

    「未表态」立场一律不渲染——即使 `逻辑` 含条件性方向判断、`条件/前置事件`
    非空也视为无实质内容（渲染层统一跳过，不再保留任何「未表态」维度）。
    """
    if not isinstance(stance, dict):
        return False
    return not is_unstated(stance.get("观点"))


def stance_tone(viewpoint: Any) -> str:
    """投资机会「短线/中长线/当前价格」立场文本 → 展示配色 tone（仅配色，不做渲染/丢弃判定）。

    丢弃判定由 `is_unstated` 单独负责——调用方必须先 `is_unstated` 过滤掉「未表态/空」，
    再对本函数的结果上色，两者正交。映射：
    - 看多/看涨 → bull；看空/看跌 → bear；
    - 不贵也不便宜（同时含「不贵」「不便宜」，须先于「贵/便宜」判定）→ neutral；
    - 贵 → expensive；便宜 → cheap；其余（中性等）→ neutral。
    """
    text = viewpoint.strip() if isinstance(viewpoint, str) else ""
    if "看多" in text or "看涨" in text:
        return "bull"
    if "看空" in text or "看跌" in text:
        return "bear"
    if "不贵" in text and "不便宜" in text:
        return "neutral"
    if "贵" in text:
        return "expensive"
    if "便宜" in text:
        return "cheap"
    return "neutral"


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
    short_unstated = not stance_has_substance(short_term)
    long_unstated = not stance_has_substance(medium_long)
    price_viewpoint = (inv.get("当前价格") or {}).get("观点")
    price_unstated = is_unstated(price_viewpoint)
    return short_unstated and long_unstated and price_unstated


def inv_has_substance(inv: dict) -> bool:
    """检查投资机会项是否有实质性内容。

    用于渲染前过滤：如果一个项的所有时间维度都「未表态」、价格也「未表态」，
    则它没有实质性内容，不渲染在「投资机会」章节中。任何「未表态」维度都
    不视为有实质内容（不再因逻辑含方向判断或条件/前置事件非空而保留）。
    """
    # 检查是否有实质内容的时间维度（观点非「未表态」）
    for tk in ("短线", "中长线", "中线", "长线"):
        tv = inv.get(tk)
        if isinstance(tv, dict) and stance_has_substance(tv):
            return True

    # 检查当前价格是否非未表态（该字段是嵌套字典，需要获取其中的"观点"）
    price_viewpoint = (inv.get("当前价格") or {}).get("观点")
    if not is_unstated(price_viewpoint):
        return True

    return False


def sector_tendency_rank(tendency: Any) -> int:
    """板块观点「预测倾向」→ 渲染排序档位：看多=0（最先），看空=1，其余/不明确=2（最后）。"""
    if not isinstance(tendency, str):
        return 2
    if "看多" in tendency:
        return 0
    if "看空" in tendency:
        return 1
    return 2


def sector_tendency_tone(tendency: Any) -> str:
    """板块观点「预测倾向」→ 颜色 tone：看多=bull（红），看空=bear（绿），其余/不明确=neutral（灰）。"""
    if not isinstance(tendency, str):
        return "neutral"
    if "看多" in tendency:
        return "bull"
    if "看空" in tendency:
        return "bear"
    return "neutral"
