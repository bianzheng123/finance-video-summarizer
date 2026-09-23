"""修复模式的待修条目：只做数据整形，文案由各阶段模板负责。

5_1 / 5_2 / 5_3 三个阶段的修复块结构一致，故共用本模块。
"""

import json


def build_repair_items(repair_targets: list[dict] | None, snippet_chars: int = 400) -> list[dict]:
    """把校验失败的条目整形成模板可直接遍历的列表。

    Args:
        repair_targets: 上游校验产出的失败条目，含 `字段` / `原始内容` / `失败原因`
        snippet_chars: `原始内容` 序列化后的截断长度，避免 prompt 过长

    Returns:
        `[{"字段": ..., "失败原因": ..., "原始内容片段": ...}]`，
        `原始内容片段` 在原始内容为空时为空字符串。
    """
    items: list[dict] = []
    for target in repair_targets or []:
        content = target.get("原始内容", "")
        items.append({
            "字段": target.get("字段", ""),
            "失败原因": target.get("失败原因", ""),
            "原始内容片段": (
                json.dumps(content, ensure_ascii=False)[:snippet_chars] if content else ""
            ),
        })
    return items
