"""缺席桶共享模块：第 4 步「4_7 祖先父生成」建桶与「4_9 两层压缩」折叠共用的判据。

「缺席桶」是候选父里不在 `keyword_data`（字幕真实提取的关键词）中的板块名——
来自 4_7 祖先父生成参考 `root_sector.json` 一级行业库联想出的父板块，或标准桶名
（如「有色金属」「能源大宗」）。它没有字幕锚点（`引用` 为空）、是「组织性节点」
而非文本实体，打上 `辅助` 标记，供两层压缩折叠删除（子项上移），而文本关键词
永不删除。

4_7 祖先父生成在 LLM 拟父输出后**立即**为缺席父建桶、补入父子关系图，使 4_7
结束即得到完整图；4_9 两层压缩据此判断哪些节点可折叠。
"""

import logging
from typing import Any

from ..schemas import TopicData

logger = logging.getLogger(__name__)

# 辅助关键词标记：祖先父生成 / 标准桶 / 一级行业建桶时写入，标识「可折叠销毁」。
AUXILIARY_FLAG: str = "辅助"

# 标准桶集合（child 候选父板块不在顶层时允许即时创建）。
STANDARD_BUCKETS: set[str] = {
    "A股指数", "港股指数", "美股指数",
    "贵金属", "工业金属", "能源大宗", "化工大宗", "农产品", "稀有气体",
    "有色金属",
    "外汇", "加密货币",
}

# 非行业「交易板块」概念名：不是申万行业维度，既不作父也不作子，其下个股改归真实
# 一级行业（如「科创板」挂的「金力股份」改归电力设备/基础化工）。精确匹配，不做
# 前缀匹配——避免误伤「港股银行」「港股泡泡玛特」这类带 venue 前缀的具体标的。
# 风格/因子（红利/次新股等）已升为独立「风格因子」类型，由 `NON_INDUSTRY_TYPES`
# 类型级处理，不在此名单。
NON_INDUSTRY_CONCEPTS: set[str] = {
    # 交易板块 / 上市板
    "科创板", "创业板", "北交所", "新三板", "主板", "中小板",
}

# 非行业「正交维度」类型：指数、风格因子都横跨所有行业，不属于任一行业维度，
# 故不参与父子挂载（不生成主题 section），只作为关键词（带行号）流转到宏观分析定位。
NON_INDUSTRY_TYPES: frozenset[str] = frozenset({"指数", "风格因子"})


def is_non_industry_concept(topic: str) -> bool:
    """是否为非行业「交易板块」概念（科创板/创业板等），不参与父子挂载。"""
    return topic in NON_INDUSTRY_CONCEPTS


def is_non_industry_type(td: TopicData | None) -> bool:
    """是否为非行业「正交维度」类型（指数/风格因子），不参与父子挂载。"""
    return isinstance(td, TopicData) and td.类型 in NON_INDUSTRY_TYPES


def infer_bucket_type(bucket_name: str) -> str:
    """按桶名推断标准桶的类型。"""
    if "指数" in bucket_name:
        return "指数"
    if bucket_name == "外汇":
        return "外汇"
    if bucket_name == "加密货币":
        return "加密货币"
    return "大宗商品"


def is_auxiliary(topic_data: Any) -> bool:
    """是否为可折叠销毁的辅助桶（祖先父生成 / 标准桶 / 数据库 sector）。"""
    return isinstance(topic_data, TopicData) and bool(topic_data.辅助)


def create_bucket(keyword_data: dict[str, TopicData], parent: str) -> None:
    """为缺席父在 `keyword_data` 中建桶（引用为空、打辅助标记）。

    桶的「引用」为空——辅助桶无自身锚点，其完整引用由引用扩展递归段并集计算；
    打「辅助」标记标识其非文本提取、可折叠销毁（文本关键词无此标记、永不销毁）。
    已存在时幂等跳过。
    """
    if parent in keyword_data:
        return
    if parent in STANDARD_BUCKETS:
        ptype = infer_bucket_type(parent)
        logger.info("创建标准桶 '%s' (类型=%s)", parent, ptype)
    else:
        ptype = "板块"
        logger.info("创建数据库 sector '%s' (类型=%s)", parent, ptype)
    keyword_data[parent] = TopicData(
        类型=ptype,
        引用=[],
        子项=[],
        辅助=True,
    )
