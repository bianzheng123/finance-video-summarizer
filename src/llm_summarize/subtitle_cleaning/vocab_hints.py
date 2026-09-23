"""黑话别名表三源合并 + hotspot 线索门控（供 2_2/2_3/3_1 的 [DATABASE] 块）。

**为什么需要这个模块**：旧实现的 `jargon_map` 只合并 `jargon_stock` + `jargon_sector`
两源，漏了 `jargon_hotspot`。于是「电梯」（＝胜宏科技，源于 2026-06 电梯门事件，
词条明明已录入 `jargon_hotspot.json`）在纠错阶段查不到，被误判为 ASR 错字，
触发 3 轮 12 次联网搜索后放弃——判断所需的知识明明在库里，只是没送到那一步。

三源合并优先级：sector → stock → **hotspot**（hotspot 最后，键冲突以它为准，
因为事件类黑话带线索门控、语义更精确）。

**线索门控（必须保留）**：hotspot 词条可标 `需线索`——如「电梯」仅当上下文出现
PCB/覆铜板/电子布等产业线索词时才指胜宏科技，否则它就是普通名词「电梯」。
没有门控就会把真在说电梯的行误标成股票简称，等于用一个新错误换掉旧错误。
门控判定复用 `JargonHotspotManager` 的 `requires_clues` / `event_clue_overlap`
/ `match_by_substring`，与第 3 步的 `_gate_hotspot_keywords` 同口径。
"""

from __future__ import annotations

import logging
import re

from ...knowledge_base.jargon_hotspot_manager import MIN_CLUE_OVERLAP, get_hotspot_manager
from ...knowledge_base.jargon_sector_manager import get_sector_manager
from ...knowledge_base.jargon_stock_manager import get_stock_manager

logger = logging.getLogger(__name__)


# hotspot 事件别名的最小长度：单字别名不参与子串匹配。
# 事件的「黑话别名」里混有单字条目（如 霍尔木兹海峡封锁 拆出的 霍/尔/木/兹/海/峡/封/锁、
# 航天 拆出的 航/天），子串匹配会命中「昨天」「等了半天」这类无关文本。
# 与 `JargonHotspotManager.match_by_substring` 的既有规则一致（其注释：
# 「单字别名不参与子串匹配（防 尔/霍 误伤）」）。
# 只作用于 hotspot 源——sector/stock 的单字别名（如「光」＝光通讯）是人工精选的，
# 旧 `_match_jargon_hint` 明确支持其子串匹配，行为保持不变。
_MIN_HOTSPOT_ALIAS_LEN = 2


def build_jargon_map() -> dict[str, list[str]]:
    """{别名: [正式名]} 三源合并（sector → stock → hotspot，后者覆盖前者）。

    任一源加载失败只降级为空并 warning——词表是辅助知识，缺失不该中断纠错。
    """
    jargon_map: dict[str, list[str]] = {}
    for label, loader in (
        ("板块黑话", lambda: get_sector_manager().get_jargon_formal_map()),
        ("个股黑话", lambda: get_stock_manager().get_jargon_formal_map()),
    ):
        try:
            jargon_map.update(loader() or {})
        except Exception as e:  # noqa: BLE001 —— 词表缺失降级，不中断纠错
            logger.warning("加载%s映射失败: %s", label, e)

    # hotspot 事件别名（本次修复的核心：旧实现完全没有这一源）
    try:
        hotspot = get_hotspot_manager()
        skipped: list[str] = []
        for alias, event_key in hotspot.get_alias_to_event().items():
            if len(alias) < _MIN_HOTSPOT_ALIAS_LEN:
                skipped.append(alias)
                continue
            event = hotspot.get_event(event_key)
            formal = (event or {}).get("正式名")
            if isinstance(formal, str) and formal:
                jargon_map[alias] = [formal]
        if skipped:
            logger.debug("跳过 %d 个单字事件别名（防子串误伤）：%s", len(skipped), sorted(skipped))
    except Exception as e:  # noqa: BLE001
        logger.warning("加载事件热点别名失败: %s", e)

    return jargon_map


def match_alias_in_text(word: str, text: str) -> bool:
    """别名是否出现在文本中。

    中文键用子串匹配（「光」出现在「老美光」中也算）；ASCII 键用词边界匹配，
    避免 QQ/BABA/MLCC 这类短串误中英文长词。与旧 `_match_jargon_hint` 同口径。
    """
    if not word:
        return False
    if word.isascii():
        return re.search(
            rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", text,
        ) is not None
    return word in text


def _clues_from_text(text: str, clue_words: set[str]) -> set[str]:
    """从文本中挑出实际出现的线索词（大小写不敏感）。

    门控要求线索词**在字幕里逐字出现**，不接受模型脑补——所以这里用确定性
    字符串匹配从原文提取，而不是采信任何模型自报的线索。
    """
    upper = text.upper()
    return {w for w in clue_words if str(w).upper() in upper}


def gate_hotspot_alias(alias: str, context_text: str) -> bool:
    """「需线索」事件别名是否通过线索门控（非需线索别名恒 True）。

    Args:
        alias: 待判定的别名（如「电梯」）
        context_text: 该别名所在的上下文文本（本话题/本块字幕原文）
    """
    hotspot = get_hotspot_manager()
    event_key = hotspot.get_alias_to_event().get(alias)
    if event_key is None:
        # 别名不属于任何事件 → 可能是 stock/sector 黑话，不受事件门控约束
        return True
    if not hotspot.requires_clues(event_key):
        return True
    event = hotspot.get_event(event_key) or {}
    clue_words = set(event.get("字幕线索词") or [])
    if not clue_words:
        # 标了需线索却没配线索词 = 词条不完整，保守放行并 warning（不静默吞掉）
        logger.warning("事件「%s」标记需线索但未配置字幕线索词，别名「%s」放行", event_key, alias)
        return True
    hit = _clues_from_text(context_text, clue_words)
    passed = len(hit) >= MIN_CLUE_OVERLAP
    if not passed:
        logger.debug(
            "别名「%s」线索门控未过（命中 %d < %d）：%s",
            alias, len(hit), MIN_CLUE_OVERLAP, sorted(hit),
        )
    return passed


def build_jargon_hints(
    jargon_map: dict[str, list[str]],
    context_text: str,
) -> list[dict]:
    """构造 [DATABASE] 的黑话别名清单：只列**本次上下文里实际出现**且过门控的别名。

    只列出现的别名（而非整表）是为了控制 prompt 体积、也让模型聚焦；门控保证
    「电梯」这类需线索别名不会在无关语境里误导模型。

    Returns: `[{"word": 别名, "formal": "正式名1、正式名2"}]`，按别名排序（确定性）。
    """
    hints: list[dict] = []
    for word in sorted(jargon_map.keys()):
        formals = jargon_map.get(word) or []
        if not formals:
            continue
        if not match_alias_in_text(word, context_text):
            continue
        if not gate_hotspot_alias(word, context_text):
            continue
        hints.append({"word": word, "formal": "、".join(formals)})
    return hints


def scan_jargon_in_text(
    jargon_map: dict[str, list[str]],
    context_text: str,
) -> list[str]:
    """确定性扫描：文本里出现且过门控的黑话别名列表（零 LLM、零成本）。

    用于 2_2 跳过条件的兜底——2_1 没有词表，可能自信地把「电梯」当普通名词而
    根本不标疑似点。只要本话题文本命中任何黑话别名，就强制进 2_2 让模型带词表
    判一次，避免这类词直接漏过（这正是本次 case 的漏洞所在）。
    """
    return [h["word"] for h in build_jargon_hints(jargon_map, context_text)]
