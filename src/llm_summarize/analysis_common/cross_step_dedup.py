"""跨步骤归属仲裁：删掉步骤 5「仓位建议」里其实属于步骤 6「交易技巧」的条目。

**为什么需要这一层**：步骤 5/6/7 并行执行、互不可见，各自的引用校验（5_3 / 7_2）
只判「引用是否支撑本条观点」，不判「这条内容是否该属于这个字段」。于是同一段字幕
可以在宏观「仓位建议」和交易技巧里各成一条，读者在报告里读两遍——违反报告质量三
铁律第 1 条（不重复）。三路产物只在 `llm_analysis.py` join 后才第一次同时可见，
本模块就挂在那个唯一的汇合点上。

**为什么只裁「仓位建议 × 交易技巧」这一对**：宏观字段与主题分析的引用行号在设计上
就大面积重合（宏观从全局看、主题从标的看同一段话，实测各字段对主题的覆盖率普遍
100%），所以「引用行号重叠」本身**不是**重复信号，泛化去重会把宏观整段删空。只有
这一对在回答同一个问题——「该怎么操作」——重叠才真的意味着重复。

**为什么还要总量关键词这道否决**：光看重叠会误删。`仓位不宜低于5成` 与交易技巧的
`仓位与年线挂钩` 引用 100% 相同，但前者是本场当期仓位数值，恰恰是「仓位建议」唯一
合法的内容；后者是可复用的挂钩规则。两者的切分靠 prompt（5_2 的作用域下界 + 6_1
的对称声明），不靠代码。因此代码只在「高重叠 **且** 本条不含任何总量主张」时才删，
关键词命中一律**否决删除**——这道判据只会保住条目，不会删错内容，最坏结果是多留一
行啰嗦（宁可多留一行，不可误删）。
"""

import logging
import re

logger = logging.getLogger(__name__)

# 引用行号被单条交易技巧覆盖到这个比例才视为同一主张
_COVERAGE_THRESHOLD = 0.9

# 「总仓位/整体敞口」主张的标志词：命中即否决删除（本条是账户层面的总量建议，
# 属于宏观「仓位建议」的正当内容，哪怕它与某条交易技巧共用引用行）
_TOTAL_POSITION_RE = re.compile(
    r"仓位|成仓|几成|空仓|满仓|杠杆|敞口|总头寸|底仓|[0-9一二三四五六七八九十半]成"
)


def _citation_rows(citations: object) -> set[int]:
    """引用数组 → 行号集合（跳过形态不合法的条目）。"""
    rows: set[int] = set()
    if not isinstance(citations, list):
        return rows
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        start, end = citation.get("开始行号"), citation.get("结束行号")
        if isinstance(start, int) and isinstance(end, int) and start <= end:
            rows |= set(range(start, end + 1))
    return rows


def dedup_position_advice_against_skills(analysis: dict) -> int:
    """删除「仓位建议」里与交易技巧重复、且不含总量主张的条目，返回删除条数。

    原地修改 `analysis`。缺 `宏观分析` / `交易技巧` 任一方时直接返回 0（两步并行，
    单步失败的场景下另一步仍应照常出报告）。
    """
    macro = analysis.get("宏观分析")
    skills = analysis.get("交易技巧")
    if not isinstance(macro, dict) or not isinstance(skills, dict):
        return 0
    blocks = macro.get("仓位建议")
    if not isinstance(blocks, list) or not blocks:
        return 0

    skill_rows = {
        name: _citation_rows(item.get("引用"))
        for name, item in skills.items()
        if isinstance(item, dict)
    }

    kept: list[object] = []
    removed = 0
    for block in blocks:
        if not isinstance(block, dict):
            kept.append(block)
            continue
        title = str(block.get("标题") or "").strip()
        rows = _citation_rows(block.get("引用"))
        if not rows:
            kept.append(block)
            continue

        # 总量主张否决：本条讲的是账户层面拿多少，无论多重叠都保留
        if _TOTAL_POSITION_RE.search(title + str(block.get("观点") or "")):
            kept.append(block)
            continue

        duplicate_of = next(
            (
                name
                for name, s_rows in skill_rows.items()
                if s_rows and len(rows & s_rows) / len(rows) >= _COVERAGE_THRESHOLD
            ),
            None,
        )
        if duplicate_of is None:
            kept.append(block)
            continue

        logger.warning(
            "跨步骤归属仲裁：删除仓位建议「%s」——引用被交易技巧「%s」覆盖 %.0f%%"
            "且本条无总仓位主张（属通用交易纪律，归交易技巧）",
            title, duplicate_of,
            len(rows & skill_rows[duplicate_of]) / len(rows) * 100,
        )
        removed += 1

    if removed:
        macro["仓位建议"] = kept
        logger.info(
            "跨步骤归属仲裁完成：仓位建议 %d→%d 条（删除 %d 条与交易技巧重复）",
            len(blocks), len(kept), removed,
        )
    else:
        logger.info("跨步骤归属仲裁完成：仓位建议 %d 条无与交易技巧重复项", len(blocks))
    return removed
