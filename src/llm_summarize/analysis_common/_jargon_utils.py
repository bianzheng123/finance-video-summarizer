"""黑话注入共享工具。

由 MacroAnalyzer / TopicAnalyzer 在各自的 inject_jargon() 中复用，避免重复实现。

- collect_used_jargons: 采集字幕里实际出现过的黑话 → {正式名: [黑话, ...]}
- inline_single_field / inline_array_field: 把正式名替换成 "白话（黑话）" 内联形式
"""

from pathlib import Path

from ..schemas import SubtitleRow, TopicData
from ..utils.jargon_loader import load_incremental_jargon
from ...knowledge_base.jargon_stock_manager import get_stock_manager
from ...knowledge_base.jargon_sector_manager import get_sector_manager


def _format_with_jargon(formal: str, jargons: list[str]) -> str:
    """正式名 → '白话（黑话）' 内联形式。0 黑话保持原样。"""
    if not isinstance(formal, str) or not formal:
        return formal
    if not jargons:
        return formal
    if "（" in formal and formal.endswith("）"):
        return formal
    return f"{formal}（{'、'.join(jargons)}）"


def _strip_jargon_suffix(name: str) -> str:
    """'白话（黑话）' → '白话'，幂等查询用。"""
    if not isinstance(name, str):
        return name
    if "（" in name and name.endswith("）"):
        return name.split("（", 1)[0]
    return name


def collect_used_jargons(
    subtitle_l: list[SubtitleRow], base_dir: Path,
    keyword_data: dict[str, TopicData],
) -> dict[str, list[str]]:
    """采集字幕里实际出现过的黑话，返回 {正式名: [黑话, ...]}。"""
    used: dict[str, list[str]] = {}

    def _add(formal: str, jargon: str) -> None:
        if not formal or not jargon:
            return
        bucket = used.setdefault(formal, [])
        if jargon not in bucket:
            bucket.append(jargon)

    # 来源 1：本会话增量解析
    inc_stock, inc_sector = load_incremental_jargon(base_dir)
    for jargon, formals in inc_stock.items():
        for formal in formals:
            _add(formal, jargon)
    for jargon, formals in inc_sector.items():
        for formal in formals:
            _add(formal, jargon)

    # 来源 2：全局黑话库子串匹配
    stock_map = get_stock_manager().get_jargon_formal_map()
    sector_map = get_sector_manager().get_jargon_formal_map()

    subtitle_text = "".join(
        (s.text or "") for s in subtitle_l if isinstance(s, SubtitleRow)
    )

    # 正式名集合（顶层键 ∪ 各主题子项）
    allowed_formals: set[str] = set()
    for topic, td in (keyword_data or {}).items():
        allowed_formals.add(topic)
        if isinstance(td, TopicData):
            for s in td.子项 or []:
                if isinstance(s, str) and s:
                    allowed_formals.add(s)

    for jargon_map in (stock_map, sector_map):
        for jargon, formals in jargon_map.items():
            if not isinstance(jargon, str) or len(jargon) < 2:
                continue
            if not isinstance(formals, list):
                continue
            # 跳过"黑话本身是正式名子串"的条目
            if any(isinstance(f, str) and jargon in f for f in formals):
                continue
            if jargon not in subtitle_text:
                continue
            for formal in formals:
                if not isinstance(formal, str) or not formal:
                    continue
                if allowed_formals and formal not in allowed_formals:
                    continue
                _add(formal, jargon)

    return used


def inline_single_field(parent: dict, field_name: str, used: dict[str, list[str]]) -> None:
    """把 parent[field_name] 中的正式名替换为 '白话（黑话）' 内联形式，
    并把命中的黑话写入 parent['原始黑话']（list[str]）。
    """
    if not isinstance(parent, dict):
        return
    raw = parent.get(field_name)
    if not isinstance(raw, str) or not raw:
        return
    base_name = _strip_jargon_suffix(raw)
    jargons = used.get(base_name)
    if not jargons:
        return
    parent[field_name] = _format_with_jargon(base_name, jargons)
    parent["原始黑话"] = list(jargons)


def inline_array_field(parent: dict, field_name: str, used: dict[str, list[str]]) -> None:
    """把 parent[field_name] 数组里的正式名逐个替换为 '白话（黑话）' 内联形式，
    并把命中的黑话索引写入 parent['原始黑话']（dict[str, list[str]]）。
    """
    if not isinstance(parent, dict):
        return
    arr = parent.get(field_name)
    if not isinstance(arr, list) or not arr:
        return
    new_arr: list = []
    jargon_index: dict[str, list[str]] = {}
    for item in arr:
        if not isinstance(item, str) or not item:
            new_arr.append(item)
            continue
        base_name = _strip_jargon_suffix(item)
        jargons = used.get(base_name)
        if jargons:
            new_arr.append(_format_with_jargon(base_name, jargons))
            jargon_index[base_name] = list(jargons)
        else:
            new_arr.append(item)
    parent[field_name] = new_arr
    if jargon_index:
        parent["原始黑话"] = jargon_index
