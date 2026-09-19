"""上传版 analysis 渐进式裁剪（纯代码，零 LLM）。

平台（B 站专栏 / 公众号）对正文长度有上限，完整版总结动辄六七万字必然超限。
本模块把裁剪拆成**一个个最小丢弃单元**，按固定顺序逐个丢，每丢一个就真实重渲染
量一次字数，**一旦落到 `UPLOAD_CHAR_LIMIT` 以内立即停手** —— 裁得越少越好。

丢弃顺序（阶段之间严格串行，上一阶段的单元没用完不动下一阶段）：
  0. `交易技巧` —— 按 JSON 原顺序【从后往前】逐条丢
  1. 各主题的 `事件与资产` —— 主题按【引用总时长升序】，逐主题丢掉整个字段
  2. 「不带观点」的主题 —— 按【引用总时长升序】，逐个删掉整主题

固定 section（宏观分析 / 主讲人暗示重要话题 / 精炼总结）永不被裁。
`精炼总结` 由 rendering_controller 在每次量字数前重算，故本模块只需裁原始数据。

## 为什么每丢一个就真实重渲染，而不按字段自身字数预估

一次量字数只要 2~3 ms（纯字符串渲染，无 IO、不生成图片），最大单元池 84 个全跑完
0.2 秒。而预估必然不准：实测「实际减量 / 字段裸字数」在 **0.97x ~ 1.46x** 之间
飘——丢一条交易技巧会连带清空「交易技巧摘要」速览表的对应行；丢掉某主题的
`事件与资产` 会让 `inv_event_titles` 变空，靠事件兜底才没被 `inv_should_skip`
砍掉的「全未表态」投资机会跟着消失。这些二阶效应推不出来，用估算值决定停手时机
必然多砍或少砍。真实测量既准确又便宜。

单调性已在存量样本上验证：逐单元丢弃 1939 步，字数没有一步反增。
"""

import html
import logging
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from .inv_filter import inv_has_substance, is_unstated

logger = logging.getLogger(__name__)

# 上传版正文字数上限。
#
# 口径 = `visible_text()` 剥离标记后的可见文字数（见该函数），**不是** markdown
# 原始字符数——公众号渲染器为控制样式会直接吐出内联 HTML 表格骨架
# （`<table style=...>` / `<colgroup>` / 每格 `<td style=...>`），实测在一份
# 裁剪版产物上原始字符 85,198 而可见文字仅 52,956，虚高 1.42 倍；更糟的是骨架
# 大小随速览表行数变化，等于给速览表偷偷加了双倍权重，会导致裁剪过度。
#
# 阈值 58000 的依据（两个真实公众号后台读数标定，目标是后台「正文字数」≤ 50000）：
#
#   样本                                        本口径    后台   后台/本口径
#   钱博士_直播回放 2026.8.6   summary.html      69,724  60,045    0.861
#   钱博士_直播回放 2026.7.26  gongzhonghao      46,644  37,875    0.812
#
# **本口径比后台偏大**（系数 < 1）：后台不计我们算进去的部分空白/换行。取更保守的
# 大系数 0.861 反推：50000 / 0.861 ≈ 58,000。两点系数相差 6.1%，说明它随内容构成
# （表格 vs 散文占比）浮动，故留了余量；再多几个读数可收紧。
#
# 重新标定：日志会打出上传版的可见文字数，上传后对照后台读数算比值即可。
# 微信不提供字数统计 API（draft/add 只收一段 HTML），只能这样标定。
UPLOAD_CHAR_LIMIT = 58000

# 非主题的顶层键：既不参与「无观点主题」判定，也不会被当成主题逐个删除。
# **新增任何顶层 section 都必须登记到这里**——四处遍历（本模块的 `_topic_keys`、
# rendering_controller 的两处、rendering_html_structured 的 SPECIAL_KEYS）全部
# 由它派生，漏登记会让新 section 被当成"主题"参与排序与裁剪。
FIXED_SECTIONS = {
    "宏观分析", "交易技巧", "精炼总结", "主讲人暗示重要话题",
}

# 剥离标记的正则（模块级预编译，measure 每次调用都要跑）
_RE_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_RE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_RE_TABLE_RULE = re.compile(r"^[\s|:\-]+$", re.M)
_RE_INLINE_MARKS = re.compile(r"[|`*#]")
_RE_HSPACE = re.compile(r"[ \t]+")


def visible_text(rendered: str) -> str:
    """把渲染产物剥成「会显示给读者的文字」，供量字数用。

    剥掉的都是标记而非内容：`<script>/<style>` 整块、HTML 标签（含公众号表格骨架）、
    图片 `![]()`、链接语法（保留链接文字）、markdown 表格分隔行、`|` `` ` `` `*` `#`。
    HTML 实体反转义回单字符（`&amp;` 算 1 字不算 5 字），连续空格/制表符折叠为一个
    （表格对齐产生的空白不该按内容计数），换行保留。
    """
    s = _RE_SCRIPT_STYLE.sub("", rendered)
    s = _RE_TAG.sub("", s)
    s = _RE_IMAGE.sub("", s)
    s = _RE_LINK.sub(r"\1", s)
    s = _RE_TABLE_RULE.sub("", s)
    s = _RE_INLINE_MARKS.sub("", s)
    s = html.unescape(s)
    return _RE_HSPACE.sub(" ", s).strip()


def _topic_keys(analysis: dict) -> list[str]:
    """返回 analysis 里的主题键（dict 值且非固定 section）。"""
    return [
        k for k, v in analysis.items()
        if k not in FIXED_SECTIONS and isinstance(v, dict)
    ]


def _topic_has_stance(topic: dict) -> bool:
    """主题是否「带观点」：存在实质投资机会立场，或存在实质观点条目。

    - 投资机会：短线 / 中长线 / 当前价格 至少一项非「未表态」（复用 inv_has_substance）
    - 观点：`观点` 文本非空且不含「未表态」（与渲染侧 _speaker_opinion_text 同一规则）

    只剩 市场知识 / 其他 / 个人交易记录 / 事件与资产 / 引用时间 的主题返回 False。
    """
    for inv in topic.get("投资机会") or []:
        if isinstance(inv, dict) and inv_has_substance(inv):
            return True
    for op in topic.get("观点") or []:
        if not isinstance(op, dict):
            continue
        if not is_unstated(op.get("观点")):
            return True
    return False


def _cite_duration(analysis: dict, topic_key: str) -> float:
    """主题的引用总时长（秒）；缺失/非数字按 0 处理（排序时最先被丢）。"""
    value = analysis[topic_key].get("引用总时长")
    return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class _TrimUnit:
    """一个最小丢弃单元。

    phase: 所属阶段名（日志与统计用）
    label: 人类可读标识（技巧名 / 主题名带引用时长）
    apply: 就地施加到 analysis 上的丢弃动作
    """

    phase: str
    label: str
    apply: Callable[[dict], None]


def _make_drop_skill(name: str) -> Callable[[dict], None]:
    """丢掉一条交易技巧；丢完最后一条时连 section 一起 pop。

    留个空 dict 会让渲染器输出一个光秃秃的「交易技巧」章节标题、底下什么都没有。
    """
    def drop(analysis: dict) -> None:
        skills = analysis.get("交易技巧")
        if not isinstance(skills, dict):
            return
        skills.pop(name, None)
        if not skills:
            analysis.pop("交易技巧", None)
    return drop


def _make_drop_events(topic_key: str) -> Callable[[dict], None]:
    """丢掉某主题的 `事件与资产` 整个字段。

    连带效应（刻意保留）：`inv_event_titles` 变空后，靠事件兜底才没被砍的
    「全未表态」投资机会会被 `inv_should_skip` 一并砍掉，速览表「事件」列变空。
    """
    def drop(analysis: dict) -> None:
        topic = analysis.get(topic_key)
        if isinstance(topic, dict):
            topic.pop("事件与资产", None)
    return drop


def _make_drop_topic(topic_key: str) -> Callable[[dict], None]:
    def drop(analysis: dict) -> None:
        analysis.pop(topic_key, None)
    return drop


def _plan_units(analysis: dict) -> list[_TrimUnit]:
    """按三阶段排定全部候选单元；列表顺序即丢弃顺序。

    单元池**在裁剪开始前一次性排定，不在循环里重算**：阶段 1 丢掉某主题的
    `事件与资产` 后，该主题可能因为「靠事件兜底的投资机会消失」而变成无观点主题，
    但它不会被追加进阶段 2。池子动态增长会让丢弃顺序依赖丢弃历史，既不可预测
    也不好解释——阶段 2 的成员资格一律按原始完整数据判定。
    """
    units: list[_TrimUnit] = []

    # 阶段 0：交易技巧，JSON 原顺序从后往前
    skills = analysis.get("交易技巧")
    if isinstance(skills, dict):
        for name in reversed(list(skills.keys())):
            units.append(_TrimUnit("交易技巧", name, _make_drop_skill(name)))

    # 阶段 1：各主题的 事件与资产，引用总时长升序（时长少的先丢）
    with_events = [k for k in _topic_keys(analysis) if analysis[k].get("事件与资产")]
    for key in sorted(with_events, key=lambda k: _cite_duration(analysis, k)):
        units.append(_TrimUnit(
            "事件与资产",
            f"{key}({_cite_duration(analysis, key):.0f}s)",
            _make_drop_events(key),
        ))

    # 阶段 2：无观点主题，引用总时长升序（时长少的先丢）
    stanceless = [k for k in _topic_keys(analysis) if not _topic_has_stance(analysis[k])]
    for key in sorted(stanceless, key=lambda k: _cite_duration(analysis, k)):
        units.append(_TrimUnit(
            "无观点主题",
            f"{key}({_cite_duration(analysis, key):.0f}s)",
            _make_drop_topic(key),
        ))

    return units


def trim_for_upload(
    base: dict, measure: Callable[[dict], int]
) -> tuple[dict, list[str], int]:
    """逐单元丢弃到字数达标，返回 (裁剪后 analysis, 已丢弃单元列表, 最终字数)。

    Args:
        base: 未注入 `精炼总结` 的干净 analysis（本函数不修改它，内部 deepcopy）
        measure: 量字数函数，入参为候选 analysis，返回可见文字数

    单元池耗尽仍超限时返回耗尽后的结果 + WARNING —— 不自造新的丢弃阶段。
    """
    candidate = deepcopy(base)
    chars = measure(candidate)
    if chars <= UPLOAD_CHAR_LIMIT:
        logger.info("上传版无需裁剪：%d 字（上限 %d）", chars, UPLOAD_CHAR_LIMIT)
        return candidate, [], chars

    units = _plan_units(candidate)
    pool: dict[str, int] = {}
    for unit in units:
        pool[unit.phase] = pool.get(unit.phase, 0) + 1
    logger.info(
        "上传版 %d 字超过上限 %d，开始渐进式裁剪（单元池 %d 个：%s）",
        chars, UPLOAD_CHAR_LIMIT, len(units),
        "，".join(f"{p} {n}" for p, n in pool.items()) or "空",
    )

    applied: list[str] = []
    used: dict[str, int] = {}
    phase_start = chars
    current_phase = units[0].phase if units else ""

    for unit in units:
        if chars <= UPLOAD_CHAR_LIMIT:
            break

        # 阶段切换：为上一阶段收个尾（逐单元只打 DEBUG，阶段级才打 INFO——
        # 单元数可达几十个，逐条 INFO 会淹掉 workflow.log）
        if unit.phase != current_phase:
            logger.info(
                "裁剪阶段「%s」用尽 %d/%d 项，%d → %d 字",
                current_phase, used.get(current_phase, 0), pool[current_phase],
                phase_start, chars,
            )
            current_phase = unit.phase
            phase_start = chars

        unit.apply(candidate)
        applied.append(f"{unit.phase}:{unit.label}")
        used[unit.phase] = used.get(unit.phase, 0) + 1
        before = chars
        chars = measure(candidate)
        logger.debug(
            "丢弃[%s] %s：%d → %d 字（-%d）",
            unit.phase, unit.label, before, chars, before - chars,
        )

    detail = "，".join(f"{p} {n}" for p, n in used.items()) or "无"
    if chars <= UPLOAD_CHAR_LIMIT:
        logger.info(
            "上传版裁剪完成：%d 字（上限 %d），用 %d/%d 单元（%s）",
            chars, UPLOAD_CHAR_LIMIT, len(applied), len(units), detail,
        )
    else:
        logger.warning(
            "单元池 %d 个全部用尽后仍有 %d 字，超过上限 %d，按现状上传"
            "（可能被平台截断；已丢弃：%s）",
            len(units), chars, UPLOAD_CHAR_LIMIT, detail,
        )
    return candidate, applied, chars
