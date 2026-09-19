"""analysis 清理：移除校验失败条目与失败标记字段。

供两处共用：
- 引用校验收尾（步骤 5 / 7 内联调用 `citation_check/`）：终态 sentinel
  （`_引用校验丢弃原因`）条目 → 删除（失败条目不进 llm_analysis.json）；
- 渲染入口：历史缓存产物残留的 内容支撑=False 条目 → 删除 + 剥掉失败字段键。

sentinel 键不带阶段号：校验是跨步骤复用的组件，键名绑某个阶段号会在阶段增删时
变成谎话（它曾叫 `_6_2丢弃原因`，而 6_2 已随「交易技巧不做校验」一起消失）。
"""

from typing import Any

_DROP_KEY = "_引用校验丢弃原因"
_FAILURE_KEYS = ("内容支撑", "失败原因", _DROP_KEY)


def clean_analysis(analysis: dict) -> list[dict]:
    """递归就地清理 analysis，返回被丢弃条目的记录列表。

    规则：
    - dict 含 _DROP_KEY（引用校验终态 sentinel）或 内容支撑 is False → 整条目删除
      （父层删键 / 列表删元素）；
    - 剥掉所有 dict 上的 内容支撑/失败原因/_DROP_KEY 键（无论取值，防御历史残留）；
    - 普通空 dict 不动（保守，不误删合法结构）。

    返回记录形如 [{"路径": "宏观分析.后市观点[1]", "失败原因": "..."}, ...]。
    """
    dropped: list[dict] = []

    def _clean(node: Any, path: str) -> bool:
        """返回 True 表示该节点应被父层删除。"""
        if isinstance(node, dict):
            if _DROP_KEY in node:
                reason = node.get(_DROP_KEY)
                dropped.append({
                    "路径": path,
                    "失败原因": reason if isinstance(reason, str) else str(reason),
                })
                return True
            if node.get("内容支撑") is False:
                dropped.append({"路径": path, "失败原因": node.get("失败原因") or ""})
                return True
            for k in list(node.keys()):
                if k in _FAILURE_KEYS:
                    del node[k]
                    continue
                if _clean(node[k], f"{path}.{k}" if path else k):
                    del node[k]
            return False
        if isinstance(node, list):
            for i in range(len(node) - 1, -1, -1):
                if _clean(node[i], f"{path}[{i}]"):
                    del node[i]
            return False
        return False

    _clean(analysis, "")
    return dropped
