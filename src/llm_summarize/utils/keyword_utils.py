"""关键词合并的共享规则：类型枚举与类型仲裁。

第 3 步消歧候选聚合（reference_resolution/jargon_disambiguation.py）与
第 3 步注释聚合（step3_refine/topic_aggregator.py）共用同一套合并语义，
保证两处对同一批逐行关键词的归一化结果一致。
"""

# 关键词类型枚举（与 3_1_关键词与剪枝.json 的 enum 同步校验）
KEYWORD_TYPES: tuple[str, ...] = (
    "关键词事件", "股票", "板块", "指数", "风格因子", "债券", "大宗商品", "外汇",
    "加密货币", "个股黑话", "板块黑话",
)

# 类型仲裁优先级：关键词事件 > 板块 > 指数 > 风格因子 > 债券 > 大宗商品 > 外汇 >
# 加密货币 > 股票 > 个股黑话 > 板块黑话
# 「风格因子」置于「板块」「指数」之后：同一词既像板块又像风格因子时优先判板块。
TYPE_PRIORITY: tuple[str, ...] = (
    "关键词事件", "板块", "指数", "风格因子", "债券", "大宗商品", "外汇", "加密货币",
    "股票", "个股黑话", "板块黑话",
)


def select_topic_type(type_candidates: set[str]) -> str:
    """按 TYPE_PRIORITY 仲裁类型；无候选时默认「板块」。"""
    if not type_candidates:
        return "板块"
    for t in TYPE_PRIORITY:
        if t in type_candidates:
            return t
    return sorted(type_candidates)[0]
