"""LLM 文本工具：prompt 加载 + 字幕字典转 LLM 输入文本。

缓存命中设计：user message 块序为 `[TASK]→[DATA]→[ANNOTATION]→[DATABASE]`，
静态的 `[TASK]` 置于最前，共享前缀在前缀缓存里只存一份。

`[DATA]` 的三个维度：

- **范围**：绝大多数阶段用窗口版（`render_windowed_subtitle`，只给工作范围
  ±`N_REDUNDANT_ROWS` 行冗余上下文）。各阶段的窗口中心：1_1=本块行、1_2=两段
  区间+完整桥接间隔、2_1~2_3/3_1=本批话题行、3_2=本困惑单元行、
  4_3=组内候选段行∪全部主题锚点行、7_1=本批主题引用切片、
  6_2=敏感信号锚定行、引用校验（5_3/6_3/7_2）=主题段行（主题类）/
  条目引用行（宏观、敏感类）。
  仍用全场版（`render_scoped_subtitle` 不传 `row_scope`）的都有确定性理由：
  5_2 宏观分析按 5_1 选中的段切片、**6_1 交易技巧看全场**（技巧常是对某段行情
  的复盘，切片会看不到被复盘的那段）、4_1/4_2 聚类（判定 child 是否"完全嵌入"
  parent 讨论必须看全场，切了也只省 24%）。
- **`[ROWID]` 与冗余的关系**：冗余行使 `[DATA]` ⊃ 工作范围，故带冗余的阶段
  必须保留 `[ROWID]`（或 `[TASK]` 内清单）显式声明本次负责哪段，并在 prompt 里
  写明"范围外只作上下文"；越界过滤（`rowid_scope.filter_*`）一条都不能省。
- **文本形态**：懒加载——第 2 步末把指代/黑话/段落指代**结构化写进 `annotation`**
  （不物化 `text_llm`），渲染 `[DATA]` 时由 `llm_text_of` 动态内联（`原词（指代）`/
  `别名（正式名）`）；第 3 步末拿到关键词类型后，`materialize_once` 从纯 `text`
  **一次性物化** `text_llm`（指代 + 黑话 + 类型 + 段落指代，`词（正式名，类型）`/
  `词（类型）`），此后第 4~7 步一律渲染 `text_llm` 且逐字节稳定。3_1/3_2 渲染期
  读动态内联结果（3_2 判断"这段到底在讲什么"就靠它）。`text` 保持纯纠错文本，
  **内联注释绝不能进上传到 B 站的字幕文件**。
- **行号标识列**：3_1 剪枝终判之后（步骤 4 起）`[DATA]` 是三列
  `rowID\\t文本\\t行号标识`（`重点`/`无关`/`噪音`，`with_type=True`）。
  步骤 1-3 只有两列——那时逐行类型还只是初判，给了会被当成终判采信。
  三列生效的阶段，`[ANNOTATION]` 不再重复给剪枝行号数组（双写必然打架）。
"""

import json
import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .keyword_utils import select_topic_type
from .rowid_scope import scope_id_set
from ..schemas import SubtitleRow, TopicSegment, TopicUnit

logger = logging.getLogger(__name__)

# 窗口化 [DATA] 的冗余上下文行数（工作范围前后各留多少行，按**列表位置**扩展
# 而非 rowID 数值——rowID 有缺号，`rid ± n` 会少给行）。
# 所有切片阶段共用这一个值：放大提升长距离指代/语境可得性，缩小省 token 与降噪。
# 由 env `N_REDUNDANT_ROWS` 覆盖（默认 15），便于不改代码调参实测。
N_REDUNDANT_ROWS = int(os.getenv("N_REDUNDANT_ROWS", "15"))


def load_prompt(filename: str, caller_file: str) -> str:
    return (Path(caller_file).parent / filename).read_text(encoding="utf-8").strip()


def format_speaker_label(speaker: int | str | None) -> str:
    """把 in-memory 的 speaker 字段规整成 "SpeakerX" 标签。

    上游取值不一：ASR 原始 int（0/1）、部分加载器保留的字符串（"1" 或
    "Speaker0"）。统一输出 "SpeakerX"；None / 空串 / 0 一律归到 "Speaker0"。
    """
    s = str(speaker).strip() if speaker is not None else ""
    if not s or s == "0":
        return "Speaker0"
    return s if s.startswith("Speaker") else f"Speaker{s}"


# 逐行注释在字幕行上的键名（清洗纠错阶段写入）
ANNOTATION_KEY = "annotation"
# 送进 [DATA] 的文本键名：第 2 步末尾物化的「纠错 + 指代内联」版本。
# 与 `text` 分离是硬性要求——`text` 是纯纠错文本（`text_llm` 的物化基础），
# 内联注释 `原词（指代）` 绝不能进上传到 B 站的字幕。
LLM_TEXT_KEY = "text_llm"
# 非分析对象行类型：保留在 [DATA] 里（作上下文），由行号标识列/[ANNOTATION] 标出，
# 下游阶段不得引用 / 分析这些行
_NON_ANALYSIS_TYPES = ("无关", "噪音")

# 可内联进 `text_llm` 的关键词真实类型（排除未解析黑话 `个股黑话`/`板块黑话`）：
# 只有解析出真实实体/事件/标的的类型才内联，未解析黑话的正式名=词本身、不可采信。
_INLINABLE_KEYWORD_TYPES: frozenset[str] = frozenset({
    "关键词事件", "股票", "板块", "指数", "风格因子", "债券", "大宗商品", "外汇",
    "加密货币",
})


def llm_text_of(sub: SubtitleRow) -> str:
    """取该行送进 `[DATA]` 的文本：`text_llm` 已物化则直接返回；否则**渲染期懒加载**，
    读 `text` + `annotation` 动态内联（指代/黑话 + 段落指代），不落盘。

    懒加载是刻意的：`text_llm` 只在第 3 步末一次性物化（指代 + 黑话 + 类型齐了
    才固化），此前第 2 步末~第 3 步末之间 `text_llm` 为 None，渲染时动态内联——
    第 3 步的 3_1/3_2 也要看到已消解的指代（3_2 判断"这段在讲什么"就靠它）。
    回退 `text` 仍是兜底：第 2 步之前 annotation 尚无「指代」，动态内联退化为
    纯文本，等价于旧实现的 `text_llm or text`。
    """
    if sub.text_llm:
        return sub.text_llm
    return _render_dynamic_inlined(sub)


def row_type_of(sub: SubtitleRow) -> str:
    """取该行的行号标识（`重点`/`无关`/`噪音`）；无注释的行默认 `重点`。"""
    ann = sub.annotation
    if not isinstance(ann, dict):
        return "重点"
    typ = ann.get("类型")
    return typ if typ in ("重点", *_NON_ANALYSIS_TYPES) else "重点"


def _row_type_column(sub: SubtitleRow) -> str:
    """第三列完整内容：行号标识 +（有不确定时）不确定错词/指代说明。

    确定的信息已物化进文本（错词替换、指代内联），本列只承载「写不进文本」的
    不确定说明——2_3 后仍无法判断的错词、仍困惑的指代。形态
    `重点` 或 `重点|错词:通价→铜价?；指代:电梯→胜宏科技?`。
    """
    typ = row_type_of(sub)
    ann = sub.annotation
    uncertain = ann.get("不确定") if isinstance(ann, dict) else None
    if not uncertain:
        return typ
    notes: list[str] = []
    for u in uncertain:
        if not isinstance(u, dict) or not u.get("词"):
            continue
        word = u["词"]
        cand = u.get("候选")
        kind = (
            "嘴瓢" if u.get("类型") == "嘴瓢"
            else "错词" if u.get("类型") == "错词"
            else "指代"
        )
        notes.append(f"{kind}:{word}→{cand}?" if cand else f"{kind}:{word}?")
    return f"{typ}|{'；'.join(notes)}" if notes else typ


def _keyword_entry_payload(entry: dict) -> dict:
    """关键词条目紧凑载荷（[ANNOTATION] 用）。

    只透传 `词`/`正式名`/`类型` 三个字段。`依据`/`困惑`/`困惑原因` 已随 3_1
    输出一起移除——指代与黑话解析结果已物化进 [DATA]，无需再区分依据与困惑。
    """
    return {
        "词": entry["词"],
        "正式名": entry["正式名"],
        "类型": entry.get("类型") or "",
    }


def _row_in_scope(row_id: int | str, row_scope: set[int]) -> bool:
    try:
        return int(row_id) in row_scope
    except (TypeError, ValueError):
        return False


def _is_jargon_keyword(entry: dict) -> bool:
    """关键词条目是否要进 `keywords_only_jargon` 口径的 [ANNOTATION]。

    该口径只渲染「承载消歧信息、且 [DATA] 文本里看不到」的条目：
    - 字面条目（词=正式名）已在第 3 步物化为 `词（类型）`，模型可直接从 [DATA]
      看到，跳过；
    - 已解析且 ≥2 字的黑话（词→正式名）已在第 3 步物化为 `词（正式名，类型）`，
      模型同样能从 [DATA] 直接看到，跳过——同一事实不许既内联又挂注释；
    - 未解析黑话（类型 `个股黑话`/`板块黑话`）与单字别名（无法安全内联）仍保留。
    """
    word = (entry.get("词") or "").strip()
    if entry.get("类型") in ("个股黑话", "板块黑话"):
        return True
    return len(word) < 2 and word != (entry.get("正式名") or "").strip()


def _render_data_line(sub: SubtitleRow, with_type: bool) -> str | None:
    """渲染 `[DATA]` 的一行；空文本返回 None（由调用方跳过）。

    两列 `rowID\\t文本`，`with_type=True` 时三列 `rowID\\t文本\\t行级注释`
    （行号标识 + 不确定错词/指代说明）。不标注说话人——多人对话 / 一人念弹幕
    由模型按内容自行推断。
    """
    text = llm_text_of(sub)
    if not text:
        return None
    cols = [str(sub.rowID), text]
    if with_type:
        cols.append(_row_type_column(sub))
    return "\t".join(cols)


def subtitle_dict_to_llm_text(
    subtitle_l: list[SubtitleRow], with_type: bool = False,
) -> str:
    """把字幕字典列表渲染成 LLM 输入文本。

    无关/噪音行同样保留（作上下文），由行号标识列（`with_type=True`，步骤 4 起）
    或 `[ANNOTATION]` 剪枝数组（步骤 1-3）标出，prompt 约束 + 下游引用校验
    双重保证它们不被引用。空文本行跳过。
    """
    lines = []
    for sub in subtitle_l:
        line = _render_data_line(sub, with_type)
        if line is not None:
            lines.append(line)
    return "\n".join(lines)


def render_full_subtitle(
    subtitle_l: list[SubtitleRow], with_type: bool = False,
) -> str:
    """渲染送进 `[DATA]` 的完整字幕（所有 LLM 调用的统一 [DATA] 底稿）。

    同一阶段的所有并行调用必须逐字节共用这一份文本，完整字幕才能在前缀缓存
    里只存一份。`with_type` 必须在同一阶段内保持一致（步骤 4 起恒 True）。
    """
    return subtitle_dict_to_llm_text(subtitle_l, with_type=with_type)


def count_occurrences(text: str, word: str) -> int:
    """word 在 text 中互不重叠的出现次数（自左向右、每次匹配后跳过该词）。

    是「出现序号」语义的唯一权威计数：3_1 校验与内联替换必须用同一口径。
    """
    count = 0
    start = 0
    while True:
        idx = text.find(word, start)
        if idx < 0:
            return count
        count += 1
        start = idx + len(word)


def _referent_names(raw: Any) -> list[str]:
    """把指代的「指代」字段归一化成非空实体名字列表（一对多）。

    接受字符串数组，或历史/误判的单字符串（拆成单元素）。去重保序。
    """
    if isinstance(raw, str):
        candidates: list = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if not isinstance(c, str):
            continue
        s = c.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_inlined_text(sub: SubtitleRow, referents: list[dict] | None = None) -> str:
    """把已确认（困惑=false）的指代内联进文本，产出 `别名（正式名）`/`原词（指代）`。

    是渲染期动态内联（`_render_dynamic_inlined`）与 2_x 预览物化
    （`materialize_llm_text`）共用的底层：读 `text` + 指代条目，按「出现序号」
    定位替换。第 2 步末起 3_1/3_2 渲染 `[DATA]` 时经 `llm_text_of` 动态内联，
    3_2 话题重置判断"这段到底在讲什么"正需要它。

    Args:
        sub: 字幕行（读 `text` 与 `annotation.指代`）
        referents: 显式指代条目列表；None 时取 `sub.annotation["指代"]`。
            第 2 步的注释尚未组装完成，故允许直接传入。

    按 `出现序号`（缺省 1）定位「原先的词」第 N 次出现，替换为 `原词（指代）`
    （全角括号）。序号超界 / 区间重叠 → WARNING 丢弃。无有效指代条目时原样
    返回 `text`。
    """
    text = sub.text
    if not text:
        return ""
    if referents is None:
        ann = sub.annotation
        if not isinstance(ann, dict):
            return text
        referents = ann.get("指代") or []
    chosen: dict[tuple[str, int], list[str]] = {}
    for r in referents:
        if not isinstance(r, dict) or not r.get("原先的词"):
            continue
        names = _referent_names(r.get("指代"))
        if not names:
            continue
        if r.get("困惑") is True:
            continue
        word: str = r["原先的词"]
        try:
            occ = int(r.get("出现序号") or 1)
        except (TypeError, ValueError):
            logger.warning("指代出现序号非法：rowID %s「%s」%r，丢弃", sub.rowID, word, r.get("出现序号"))
            continue
        if occ < 1:
            logger.warning("指代出现序号小于 1：rowID %s「%s」%d，丢弃", sub.rowID, word, occ)
            continue
        if (word, occ) in chosen:
            logger.warning("同词同序号重复指代：rowID %s「%s」第%d次，取先入者", sub.rowID, word, occ)
            continue
        chosen[(word, occ)] = names

    if not chosen:
        return text
    replacements: list[tuple[int, int, str]] = []
    for (word, occ), names in chosen.items():
        count = count_occurrences(text, word)
        if occ > count:
            logger.warning(
                "指代出现序号超界：rowID %s「%s」第%d次 > 实际%d次，丢弃",
                sub.rowID, word, occ, count,
            )
            continue
        idx = -1
        for _ in range(occ):
            idx = text.find(word, idx + 1)
        replacements.append((idx, idx + len(word), f"{word}（{'、'.join(names)}）"))

    # 从后往前应用替换，天然免疫长度偏移；区间重叠者跳过（罕见，如「这个」与「这个票」）
    replacements.sort(key=lambda t: t[0], reverse=True)
    applied: list[tuple[int, int]] = []
    out = text
    for start, end, repl in replacements:
        if any(not (end <= s or start >= e) for s, e in applied):
            logger.warning("内联区间重叠，跳过：rowID %s「%s」", sub.rowID, repl)
            continue
        out = out[:start] + repl + out[end:]
        applied.append((start, end))
    return out


def _render_dynamic_inlined(sub: SubtitleRow) -> str:
    """渲染期懒加载：读 `text` + `annotation` 动态内联（指代/黑话 + 段落指代）。

    `text_llm` 只在第 3 步末一次性物化，此前（第 2 步末~第 3 步末）本函数是
    `llm_text_of` 的动态内联路径：指代/黑话走 `build_inlined_text`（读
    `annotation["指代"]`），段落指代按 `annotation["段落指代"]` 追加。产物逐字节
    确定，但不落盘——同一份动态文本在 3_1/3_2 各次渲染间保持一致。
    """
    text = build_inlined_text(sub)
    return _append_paragraph_referents(
        text, (sub.annotation or {}).get("段落指代") or [],
    )


def _append_paragraph_referents(text: str, paragraph_referents: list[dict]) -> str:
    """把段落指代追加到文本末尾：` {段落指代：{...}}`（不含理由，紧凑 JSON）。"""
    for pr in paragraph_referents:
        if not isinstance(pr, dict) or not pr.get("rowID"):
            continue
        marker = json.dumps(
            {"rowID": pr["rowID"], "真实含义": pr["真实含义"]},
            ensure_ascii=False, separators=(",", ":"),
        )
        text = f"{text} {{段落指代：{marker}}}"
    return text


def strip_paragraph_referents(text: str) -> str:
    """去掉 text 末尾追加的段落指代标记（` {段落指代：{...}}`），供引用原文回填用。

    段落指代由 `_append_paragraph_referents` 恒追加在行尾，是给 LLM 理解跨行指代的
    段级注释；写进 `llm_analysis.json` 的引用 `原文` 只要带内联注释的字幕本身，
    不该下发段级注释。`{段落指代：` 是合成标记，字幕原文不会自然出现，故从首次
    出现处截断即可。
    """
    idx = text.find("{段落指代：")
    return text[:idx].rstrip() if idx >= 0 else text


def _paragraph_referents_by_row(
    paragraph_referents: list[dict],
) -> dict[int, list[dict]]:
    """段落指代按「段落最后一行（max(rowID)）」分组，供写 annotation 与物化共用。"""
    by_row: dict[int, list[dict]] = {}
    for pr in paragraph_referents:
        if not isinstance(pr, dict):
            continue
        rows = pr.get("rowID") or []
        if not rows:
            continue
        by_row.setdefault(max(rows), []).append(pr)
    return by_row


def _windowed_rows(
    subtitle_l: list[SubtitleRow],
    scope_row_ids: Iterable[Any],
    pad: int,
    extra_span_ids: Iterable[Any] | None = None,
) -> tuple[list[tuple[int, SubtitleRow]], set[int]] | None:
    """算窗口化 [DATA] 的保留下标集：返回 (过滤后行列表, 保留下标集)。

    退化情形（scope 为空 / scope 行号全不存在）返回 None，由调用方回退完整字幕。
    `render_windowed_subtitle` 与 `window_row_ids` 共用本函数，保证「渲染出的行」
    与「声明为窗口内的行」逐行一致——否则 [ANNOTATION] 会为 [DATA] 里不存在的
    行给注释（或反之漏给上下文行的注释）。
    """
    # 只有非空文本行进 [DATA]（与 subtitle_dict_to_llm_text 同口径），
    # 下标空间必须建在同一份过滤后列表上，pad 才是「实际给出的行数」
    rows: list[tuple[int, SubtitleRow]] = []
    for sub in subtitle_l:
        row_id = sub.rowID
        if row_id is None or not sub.text:
            continue
        try:
            rows.append((int(row_id), sub))
        except (TypeError, ValueError):
            continue

    scope = scope_id_set(scope_row_ids)
    if not scope:
        logger.warning("窗口化 [DATA] 的工作范围为空，退化为完整字幕")
        return None

    idx_by_id = {rid: i for i, (rid, _) in enumerate(rows)}
    keep: set[int] = set()
    for rid in scope:
        i = idx_by_id.get(rid)
        if i is None:
            continue
        keep.update(range(max(0, i - pad), min(len(rows), i + pad + 1)))
    for rid in scope_id_set(extra_span_ids):
        i = idx_by_id.get(rid)
        if i is not None:
            keep.add(i)

    if not keep:
        logger.warning(
            "窗口化 [DATA]：%d 个范围行号在字幕中均不存在，退化为完整字幕", len(scope),
        )
        return None
    return rows, keep


def window_row_ids(
    subtitle_l: list[SubtitleRow],
    scope_row_ids: Iterable[Any],
    pad: int = N_REDUNDANT_ROWS,
    extra_span_ids: Iterable[Any] | None = None,
) -> set[int]:
    """窗口化 [DATA] 实际会渲染出的 rowID 集合（含 ±pad 上下文行）。

    用于把 [ANNOTATION] 切到与 [DATA] 完全相同的行集：注释块只应描述 [DATA]
    里出现过的行。退化情形（同 render_windowed_subtitle）返回全部非空文本行。
    """
    computed = _windowed_rows(subtitle_l, scope_row_ids, pad, extra_span_ids)
    if computed is None:
        return scope_id_set(
            s.rowID for s in subtitle_l if s.text
        )
    rows, keep = computed
    return {rows[i][0] for i in keep}


def render_windowed_subtitle(
    subtitle_l: list[SubtitleRow],
    scope_row_ids: Iterable[Any],
    pad: int = N_REDUNDANT_ROWS,
    with_type: bool = False,
    extra_span_ids: Iterable[Any] | None = None,
) -> str:
    """渲染窗口化 `[DATA]`：工作范围行 ± pad 行的并集，按原始顺序。

    取代「全量 [DATA] + [ROWID] 声明范围」：作用域直接由本函数过滤表达，
    模型看到的就是它要处理的行及其上下文。

    Args:
        subtitle_l: 完整字幕行列表（唯一真源，按 rowID 升序）
        scope_row_ids: 本次工作范围的 rowID（窗口中心）
        pad: 上下文半径，按**列表位置**扩展而非 rowID 数值——字幕 rowID 有
            缺号（空文本行跳过），`rid ± pad` 会少给行
        with_type: True 时追加第三列行号标识（步骤 4 起恒 True，见模块 docstring）
        extra_span_ids: 需整段纳入的额外行号（不受 pad 限制）。1_2 片段合并
            判定要读两段之间的完整间隔（间隔上限 50 行 > pad），把间隔行号
            从这里传入。

    Returns:
        `rowID\\t文本[\\t行号标识]` 行文本。不连续的窗口之间插入独占一行的
        `……（此处省略 N 行）`——窗口化打破了「相邻 rowID 时间连续」这一
        前提，断裂处必须显式告知，否则模型会把跨窗口的两行当相邻语句误读。
        scope 为空时退化为完整字幕（防空 [DATA]）。
    """
    computed = _windowed_rows(subtitle_l, scope_row_ids, pad, extra_span_ids)
    if computed is None:
        return render_full_subtitle(subtitle_l, with_type=with_type)
    rows, keep = computed

    lines: list[str] = []
    prev: int | None = None
    for i in sorted(keep):
        if prev is not None and i > prev + 1:
            lines.append(f"……（此处省略 {i - prev - 1} 行）")
        line = _render_data_line(rows[i][1], with_type)
        if line is not None:
            lines.append(line)
        prev = i
    return "\n".join(lines)


def render_scoped_subtitle(
    subtitle_l: list[SubtitleRow],
    row_scope: set[int] | None = None,
    with_type: bool = False,
) -> str:
    """按行号集合切片渲染 `[DATA]`（不带 ±pad 冗余，范围即切片）。

    `row_scope` 非 None 时只渲染该集合内的行：按主题分析这类"工作范围本就只有
    自己那几段"的阶段，给整场字幕会让模型在无关内容里找依据，也让 prompt 白白
    变长。缺省 None = 整场（4_x 等全局阶段逐字节共用同一份以命中前缀缓存）。

    取代旧的 `render_full_subtitle_inline`：内联已在第 2 步物化进 `text_llm`，
    "inline 版渲染器"这个概念本身已不存在，本函数只管**范围**这一个维度。
    """
    lines = []
    for sub in subtitle_l:
        if row_scope is not None:
            rid = sub.rowID
            if rid is None or int(rid) not in row_scope:
                continue
        line = _render_data_line(sub, with_type)
        if line is not None:
            lines.append(line)
    return "\n".join(lines)


def render_annotation_dict(
    annotations_by_row: dict[int | str, dict],
    include_keywords: bool = False,
    row_scope: set[int] | None = None,
    keywords_only_jargon: bool = False,
    include_prunes: bool = True,
    pinyin_candidates: dict[int, list[dict]] | None = None,
    row_readings: dict[int, str] | None = None,
    corrections: list[dict] | None = None,
    include_suspect_candidates: bool = True,
) -> str:
    """渲染 `[ANNOTATION]` 块内容：按类型为键的紧凑 JSON。

    新形态：`{"无关行号":[...], "噪音行号":[...], "关键词":{"3":[...]},
    "疑似黑话候选":{"3":[{"词","候选"}]}}`——
    `无关行号`/`噪音行号` 为剪枝行号升序数组；`关键词` 为 rowID 字符串 →
    条目数组（词/正式名/类型/依据?/困惑?/困惑原因?）；`疑似黑话候选` 为 2_3
    后仍未解决、但带候选的疑似点（供 3_1 参考）。指代不再出现在本块（已内联
    进 [DATA]）。未出现在任何键中的行 = 普通重点行。各键全空时返回空串，
    模板据此整块省略。

    Args:
        include_keywords: 是否渲染「关键词」子对象。默认关——全量逐行关键词
            体量约为 [DATA] 的数倍，只有明确需要逐行关键词的调用才开。
        row_scope: 非 None 时，关键词只在 rowID ∈ scope 时携带，用于按调用
            作用域瘦身。剪枝行号是否携带取决于 `include_prunes`。
        keywords_only_jargon: 只渲染承载消歧信息、且 [DATA] 文本里看不到的关键词
            条目（见 `_is_jargon_keyword`：字面条目与已内联的已确证黑话被跳过）。
        include_prunes: 是否渲染 `无关行号` / `噪音行号`。
            - True（默认）：第 2 步内部用——那时 `[DATA]` 还是两列，剪枝初判
              只能靠本块传给 2_2/2_3/3_1。
            - False：步骤 4 起用（`render_annotations` 固定传 False）。剪枝事实
              已由 `[DATA]` 第三列行号标识随行携带，本块再给一份必然不一致。
        pinyin_candidates: `{rowID: [{"片段","读音","音近",...}]}`，确定性拼音候选
            （词级阶段 2_1/2_2 用）。空/None 时该键整键不出现。
        row_readings: `{rowID: 整行拼音}`（2_2 用，只给疑似点所在行）。同上。
        include_suspect_candidates: 是否渲染 `疑似黑话候选`。
            - True（默认）：3_1 直调用——它 `[DATA]` 仍两列，`不确定` 只靠本块携带。
            - False：步骤 4 起用（`render_annotations` 固定传 False）。`不确定` 已随
              `[DATA]` 第三列行号标识逐行给出，本块再给一份必然重复。
    """
    无关行号: list[int] = []
    噪音行号: list[int] = []
    关键词: dict[str, list[dict]] = {}
    for row_id, ann in annotations_by_row.items():
        if not isinstance(ann, dict):
            continue
        typ = ann.get("类型") or "重点"
        try:
            rid = int(row_id)
        except (TypeError, ValueError):
            continue
        if typ in _NON_ANALYSIS_TYPES:
            if not include_prunes:
                continue
            (无关行号 if typ == "无关" else 噪音行号).append(rid)
            continue
        if not include_keywords:
            continue
        if row_scope is not None and not _row_in_scope(row_id, row_scope):
            continue
        entries: list[dict] = []
        for k in ann.get("关键词") or []:
            if not isinstance(k, dict) or not k.get("词") or not k.get("正式名"):
                continue
            if keywords_only_jargon and not _is_jargon_keyword(k):
                continue
            entries.append(_keyword_entry_payload(k))
        if entries:
            关键词[str(row_id)] = entries

    # 疑似黑话候选：2_3 后仍未解决、但带候选的疑似点（供 3_1 关键词提取参考）。
    # 候选是「疑似、未确证」，只进本块供参考，不物化进 [DATA]。
    # 步骤 4 起 `不确定` 已随 [DATA] 第三列行号标识逐行给出，故由
    # include_suspect_candidates 关闭（render_annotations 传 False），只留 3_1
    # 直调（默认 True）携带。
    疑似黑话候选: dict[str, list[dict]] = {}
    if include_suspect_candidates:
        for row_id, ann in annotations_by_row.items():
            if not isinstance(ann, dict):
                continue
            uncertain = ann.get("不确定")
            if not uncertain:
                continue
            entries: list[dict] = []
            for u in uncertain:
                if not isinstance(u, dict):
                    continue
                word = (u.get("词") or "").strip()
                cand = (u.get("候选") or "").strip()
                if word and cand:
                    entries.append({"词": word, "候选": cand})
            if entries:
                try:
                    rid = int(row_id)
                except (TypeError, ValueError):
                    continue
                if row_scope is not None and not _row_in_scope(row_id, row_scope):
                    continue
                疑似黑话候选[str(rid)] = entries

    # 纠错：2_1/2_2 确定性改字的痕迹（原字消失，[ANNOTATION] 补记「修改前→修改后」）
    纠错: dict[str, list[dict]] = {}
    for c in corrections or []:
        if not isinstance(c, dict):
            continue
        rid = c.get("rowID")
        before = (c.get("原先的词") or "").strip()
        after = (c.get("修改的词") or "").strip()
        if rid is None or not before or not after:
            continue
        纠错.setdefault(str(rid), []).append({"修改前": before, "修改后": after})

    # 拼音辅助信息：确定性产出，按 row_scope 收敛（不在 [DATA] 里的行不该出现）
    拼音候选: dict[str, list[dict]] = {}
    for rid, entries in (pinyin_candidates or {}).items():
        if not entries:
            continue
        if row_scope is not None and not _row_in_scope(rid, row_scope):
            continue
        拼音候选[str(rid)] = entries
    行读音: dict[str, str] = {}
    for rid, reading in (row_readings or {}).items():
        if not reading:
            continue
        if row_scope is not None and not _row_in_scope(rid, row_scope):
            continue
        行读音[str(rid)] = reading

    if (
        not 无关行号 and not 噪音行号 and not 关键词
        and not 拼音候选 and not 行读音 and not 疑似黑话候选
        and not 纠错
    ):
        return ""
    payload: dict = {}
    if 无关行号:
        payload["无关行号"] = sorted(无关行号)
    if 噪音行号:
        payload["噪音行号"] = sorted(噪音行号)
    if 关键词:
        payload["关键词"] = 关键词
    if 拼音候选:
        payload["拼音候选"] = 拼音候选
    if 行读音:
        payload["行读音"] = 行读音
    if 疑似黑话候选:
        payload["疑似黑话候选"] = 疑似黑话候选
    if 纠错:
        payload["纠错"] = 纠错
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_annotations(
    subtitle_l: list[SubtitleRow],
    include_keywords: bool = False,
    row_scope: set[int] | None = None,
    keywords_only_jargon: bool = False,
) -> str:
    """渲染 `[ANNOTATION]` 块内容（步骤 4 起：**只有关键词**）。

    剪枝行号**不再由本块携带**：3_1 剪枝终判之后 `[DATA]` 自带第三列行号标识
    （`with_type=True`），同一事实两处写必然打架——曾经的 `scope_prunes` 参数
    正是为"切片调用别标出 [DATA] 里不存在的行号"打的补丁，行号标识随行走之后
    这个问题从根上消失了。故本函数不再输出 `无关行号`/`噪音行号`，
    `scope_prunes` 参数一并删除。

    无任何有效关键词时返回空串，模板据此整块省略（此时该阶段的 [ANNOTATION]
    整块不出现，符合块协议）。第 2 步内部传剪枝初判仍走 `render_annotation_dict`。
    """
    annotations_by_row: dict[int | str, dict] = {}
    for sub in subtitle_l:
        row_id = sub.rowID
        if row_id is None:
            continue
        ann = sub.annotation
        if not isinstance(ann, dict):
            continue
        annotations_by_row[row_id] = ann
    return render_annotation_dict(
        annotations_by_row,
        include_keywords=include_keywords,
        row_scope=row_scope,
        keywords_only_jargon=keywords_only_jargon,
        include_prunes=False,
        include_suspect_candidates=False,
    )


def mark_unimportant_segments(
    subtitle_l: list[SubtitleRow],
    segments: list[TopicSegment],
) -> list[SubtitleRow]:
    """把「不重要」话题段的行打上 `{"类型": "无关"}` 合成注释（组剪枝）。

    1_1 话题分割判定整段纯无关（感谢礼物/无关闲聊/互动闲聊/小助理/卖课/带货）的段
    不再参与纠错、关键词提取与总结——这里借道逐行注释协议把剪枝传给所有下游阶段：
    无关行保留在 [DATA] 里作上下文，由 [ANNOTATION] 块标出，模型不得引用。
    已有注释的行不动（幂等）；其余段一律不动（默认重要）。
    """
    pruned_ids: set[int] = set()
    for seg in segments or []:
        if seg.是否重要 is not False:
            continue
        pruned_ids.update(range(int(seg.开始行号), int(seg.结束行号) + 1))
    if not pruned_ids:
        return list(subtitle_l)

    out: list[SubtitleRow] = []
    for sub in subtitle_l:
        row_id = sub.rowID
        if (
            row_id is not None
            and int(row_id) in pruned_ids
            and not isinstance(sub.annotation, dict)
        ):
            out.append(sub.model_copy(update={ANNOTATION_KEY: {"类型": "无关"}}))
        else:
            out.append(sub)
    return out


def collect_non_analysis_row_ids(subtitle_l: Iterable[SubtitleRow]) -> set[int]:
    """收集注释类型为 无关/噪音 的行号，供下游阶段的引用校验排除。"""
    blocked: set[int] = set()
    for sub in subtitle_l or []:
        if row_is_analyzable(sub):
            continue
        row_id = sub.rowID
        if row_id is None:
            continue
        try:
            blocked.add(int(row_id))
        except (TypeError, ValueError):
            continue
    return blocked


def row_is_analyzable(sub: SubtitleRow) -> bool:
    """行是否可作分析/引用对象：无注释的行默认可分析，注释类型为 无关/噪音 的不可。

    [DATA] 渲染时会保留无关/噪音行（作上下文），但引用校验与原文回填
    不允许落在这些被 [ANNOTATION] 标出剪枝的行上。
    """
    ann = sub.annotation
    return not isinstance(ann, dict) or ann.get("类型") not in _NON_ANALYSIS_TYPES


def materialize_llm_text(
    subtitle_l: list[SubtitleRow], referents: list[dict],
) -> list[SubtitleRow]:
    """把已确认指代物化进每行的 `text_llm`（**2_x 内部预览用**）。

    供 2_1→2_2→2_3 链内把「错词改字 + 指代内联」局部应用到字幕副本，让下一层
    2_2/2_3 的 `[DATA]` 看到已消解指代。最终收尾不再走这里——第 2 步末改为
    `_structurize_annotation` 结构化写 annotation、第 3 步末 `materialize_once`
    一次性物化。

    **`text` 一律不动**：它保持纯纠错文本，作为 `text_llm` 物化的基础，
    内联注释绝不能进上传到 B 站的字幕。两个字段的分工是硬性的，不要合并。
    """
    by_row: dict[int, list[dict]] = {}
    for r in referents or []:
        if not isinstance(r, dict):
            continue
        rid = r.get("rowID")
        if rid is None:
            continue
        try:
            by_row.setdefault(int(rid), []).append(r)
        except (TypeError, ValueError):
            continue

    out: list[SubtitleRow] = []
    inlined_rows = 0
    for sub in subtitle_l:
        rid = sub.rowID
        refs = by_row.get(int(rid)) if rid is not None else None
        text_llm = build_inlined_text(sub, referents=refs or [])
        if refs and text_llm != sub.text:
            inlined_rows += 1
        out.append(sub.model_copy(update={LLM_TEXT_KEY: text_llm}))
    logger.info(
        "指代内联物化：%d 行写入 text_llm（其中 %d 行有实际内联）",
        len(out), inlined_rows,
    )
    return out


def build_terminals_inlined_text(
    sub: SubtitleRow, terminals: list[dict], tag: str,
) -> str:
    """把嘴瓢（<slip>）/事件描述（<event>）片段物化为 `<tag>片段</tag>` 内联进文本。

    与 `build_inlined_text` 同机制：按「出现序号」定位片段第 N 次出现，替换为标签
    包裹形式。原字保留（标签只包裹不删字），空间局部性好；下游 `[DATA]` 就地读到。
    """
    text = llm_text_of(sub)
    if not text:
        return ""
    chosen: dict[tuple[str, int], str] = {}
    for t in terminals:
        if not isinstance(t, dict) or not t.get("原先的片段"):
            continue
        word: str = t["原先的片段"]
        try:
            occ = int(t.get("出现序号") or 1)
        except (TypeError, ValueError):
            continue
        if occ < 1:
            continue
        if (word, occ) in chosen:
            continue
        chosen[(word, occ)] = word
    if not chosen:
        return text
    replacements: list[tuple[int, int, str]] = []
    for (word, occ), _ in chosen.items():
        count = count_occurrences(text, word)
        if occ > count:
            logger.warning(
                "终结标签出现序号超界：rowID %s「%s」第%d次 > 实际%d次，丢弃",
                sub.rowID, word, occ, count,
            )
            continue
        idx = -1
        for _ in range(occ):
            idx = text.find(word, idx + 1)
        replacements.append((idx, idx + len(word), f"<{tag}>{word}</{tag}>"))
    replacements.sort(key=lambda t: t[0], reverse=True)
    applied: list[tuple[int, int]] = []
    out = text
    for start, end, repl in replacements:
        if any(not (end <= s or start >= e) for s, e in applied):
            logger.warning("终结标签内联区间重叠，跳过：rowID %s「%s」", sub.rowID, repl)
            continue
        out = out[:start] + repl + out[end:]
        applied.append((start, end))
    return out


def materialize_terminals(
    subtitle_l: list[SubtitleRow], terminals: list[dict], tag: str,
) -> list[SubtitleRow]:
    """把嘴瓢/事件描述标签内联进 `text_llm`（与 `materialize_llm_text` 同路）。

    `terminals` 为 `[{rowID, 原先的片段, 出现序号}]`；`tag` 为 `slip` 或 `event`。
    """
    by_row: dict[int, list[dict]] = {}
    for t in terminals or []:
        if not isinstance(t, dict):
            continue
        rid = t.get("rowID")
        if rid is None:
            continue
        try:
            by_row.setdefault(int(rid), []).append(t)
        except (TypeError, ValueError):
            continue
    out: list[SubtitleRow] = []
    inlined_rows = 0
    for sub in subtitle_l:
        rid = sub.rowID
        terms = by_row.get(int(rid)) if rid is not None else None
        text_llm = build_terminals_inlined_text(sub, terms or [], tag)
        if terms and text_llm != sub.text:
            inlined_rows += 1
        out.append(sub.model_copy(update={LLM_TEXT_KEY: text_llm}))
    logger.info(
        "终结标签（%s）内联物化：%d 行写入 text_llm（其中 %d 行有实际内联）",
        tag, len(out), inlined_rows,
    )
    return out


def materialize_once(subtitle_l: list[SubtitleRow]) -> list[SubtitleRow]:
    """第 3 步末一次性物化：指代（代词）+ 黑话 + 关键词类型 + 段落指代 → text_llm。

    从纯 `text` 出发，同次生成时把黑话与关键词类型合并为 `词（正式名，类型）`，
    避免「先物化括号、再往括号里补类型」的二次匹配——那是嵌套重复 bug 的根源。
    此后第 4~7 步渲染 `text_llm`（`llm_text_of` 直接返回），逐字节稳定。

    `text` 一律不动：它保持纯纠错文本，是 B 站上传字幕的基础，内联注释绝不进去。
    """
    out: list[SubtitleRow] = []
    inlined_rows = 0
    for sub in subtitle_l:
        text_llm = _build_final_inlined_text(sub)
        if text_llm != sub.text:
            inlined_rows += 1
        out.append(sub.model_copy(update={LLM_TEXT_KEY: text_llm}))
    logger.info(
        "最终物化：%d 行写入 text_llm（其中 %d 行有内联）",
        len(out), inlined_rows,
    )
    return out


def _build_final_inlined_text(sub: SubtitleRow) -> str:
    """最终物化单行：从纯 `text` 出发，一次性内联指代 + 黑话 + 关键词类型。

    黑话（`依据`∈{数据库,联网}）与关键词按「词」合并：词在关键词里就用关键词
    （`词（正式名，类型）` / 字面条目 `词（类型）`）；词只在黑话里则兜底
    `词（正式名）`（无类型）；代词指代（`依据`=上下文）恒替换为 `原词（指代）`。
    全部替换基于纯 `text` 定位、一次生成，长 span 优先、区间重叠跳过，防「半导体」
    误伤「半导体设备」的前缀子串。
    """
    text = sub.text
    if not text:
        return ""
    ann = sub.annotation
    if not isinstance(ann, dict):
        return text

    # ---- 关键词：词 → (正式名列表, 类型集合)，只收真实类型 + ≥2 字 ----
    kw_groups: dict[str, dict] = {}
    for kw in ann.get("关键词") or []:
        if not isinstance(kw, dict):
            continue
        word = (kw.get("词") or "").strip()
        formal = (kw.get("正式名") or "").strip()
        typ = kw.get("类型") or ""
        if not word or not formal:
            continue
        if typ not in _INLINABLE_KEYWORD_TYPES or len(word) < 2:
            continue
        g = kw_groups.setdefault(word, {"formal": [], "types": set()})
        if formal not in g["formal"]:
            g["formal"].append(formal)
        if typ:
            g["types"].add(typ)

    # ---- 指代：词 → [(出现序号, 正式名列表, 依据)] ----
    ref_entries: dict[str, list[tuple[int, list[str], str]]] = {}
    for r in ann.get("指代") or []:
        if not isinstance(r, dict):
            continue
        word = (r.get("原先的词") or "").strip()
        names = _referent_names(r.get("指代"))
        if not word or not names:
            continue
        if r.get("困惑") is True:
            continue
        try:
            occ = int(r.get("出现序号") or 1)
        except (TypeError, ValueError):
            continue
        if occ < 1:
            continue
        ref_entries.setdefault(word, []).append((occ, names, r.get("依据") or ""))

    replacements: list[tuple[int, int, str]] = []

    # 1) 关键词替换（所有出现；黑话的类型合并在此一步完成）
    for word, g in kw_groups.items():
        typ = select_topic_type(g["types"])
        formal_str = "、".join(g["formal"])
        is_literal = len(g["formal"]) == 1 and g["formal"][0] == word
        repl = f"{word}（{typ}）" if is_literal else f"{word}（{formal_str}，{typ}）"
        start = 0
        while True:
            idx = text.find(word, start)
            if idx < 0:
                break
            end = idx + len(word)
            replacements.append((idx, end, repl))
            start = end

    # 2) 指代替换（第 N 次出现；黑话若已由关键词覆盖则跳过，代词恒替换）
    for word, entries in ref_entries.items():
        for occ, names, basis in entries:
            if basis in ("数据库", "联网") and word in kw_groups:
                continue
            count = count_occurrences(text, word)
            if occ > count:
                logger.warning(
                    "最终物化指代出现序号超界：rowID %s「%s」第%d次 > 实际%d次，丢弃",
                    sub.rowID, word, occ, count,
                )
                continue
            idx = -1
            for _ in range(occ):
                idx = text.find(word, idx + 1)
            replacements.append((idx, idx + len(word), f"{word}（{'、'.join(names)}）"))

    if not replacements:
        return _append_paragraph_referents(text, ann.get("段落指代") or [])

    # 3) 应用替换：长 span 优先 + 区间重叠跳过（坐标基于原 text，从后往前应用）
    replacements.sort(key=lambda t: (t[0], t[1]), reverse=True)
    applied: list[tuple[int, int]] = []
    out = text
    for start, end, repl in replacements:
        if any(not (end <= s or start >= e) for s, e in applied):
            logger.warning("最终物化内联区间重叠，跳过：rowID %s「%s」", sub.rowID, repl)
            continue
        out = out[:start] + repl + out[end:]
        applied.append((start, end))

    # 4) 段落指代追加
    return _append_paragraph_referents(out, ann.get("段落指代") or [])


def save_subtitle_tsv(
    subtitle_l: list[SubtitleRow], output_path: Path, text_key: str = "text",
) -> None:
    """把字幕行列表原子写为四列 TSV 字幕文件（start_time\\tend_time\\tSpeakerX\\t文本）。

    先写同目录临时文件再 `os.replace` 原子替换，保证并发读者（如 B 站上传的
    SRT 转换）只会看到旧文件或新文件的完整内容，不会读到半个文件。

    Args:
        text_key: 取哪个字段当文本列。**两个取值对应两种不同产物**：

            - `"text"`（默认）：纯纠错文本（不含指代内联）。
            - `"text_llm"`：`llm_text_of`——`text_llm` 已物化则取它，否则渲染期
              动态内联（纠错 + 指代内联），只用于落盘查看
              （`llm_analysis/2_clean_subtitle.txt`）。**这个产物禁止进上传链路。**
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for sub in subtitle_l:
                start = sub.start_time
                end = sub.end_time
                speaker = format_speaker_label(sub.speaker)
                if text_key == LLM_TEXT_KEY:
                    text = llm_text_of(sub)
                else:
                    text = getattr(sub, text_key) or sub.text
                f.write(f"{start}\t{end}\t{speaker}\t{text}\n")
        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info(
        "字幕已原子覆盖: %s（共 %d 行，文本列取 %s）",
        output_path, len(subtitle_l), text_key,
    )


def build_topic_units(
    subtitle_l: list[SubtitleRow], segments: list[TopicSegment],
) -> list[TopicUnit]:
    """把字幕按话题段切成工作单元（按话题并行阶段共用）。

    剪枝口径：
    - 「不重要」话题段（组剪枝：`是否重要=false`）整体跳过，不建工作单元；
    - 注释类型为 无关/噪音 的行（组剪枝合成注释或清洗阶段标注）不作为工作对象，
      只留在 [DATA] 里作上下文——它们不会进入「未分类兜底」单元。

    Returns:
        `[TopicUnit]`；`rows` 为 `(row_id, sub)` 且 text 非空、
        可分析（重点/无注释）。未落入任何话题段的可分析行会追加一个
        "未分类兜底"单元，保证每条可分析行都被处理。
    """
    valid_items = [
        (int(sub.rowID), sub)
        for sub in subtitle_l
        if sub.text and row_is_analyzable(sub)
    ]

    units: list[TopicUnit] = []
    covered: set[int] = set()
    for seg_idx, seg in enumerate(segments):
        if seg.是否重要 is False:
            continue
        start = int(seg.开始行号)
        end = int(seg.结束行号)
        rows = [(rid, sub) for rid, sub in valid_items if start <= rid <= end]
        if not rows:
            continue
        units.append(TopicUnit(
            idx=seg_idx,
            label=str(seg.话题标签).strip() or "未命名",
            事件=str(seg.事件).strip(),
            rows=rows,
            row_ids=[rid for rid, _ in rows],
        ))
        covered.update(rid for rid, _ in rows)

    orphan_rows = [(rid, sub) for rid, sub in valid_items if rid not in covered]
    if orphan_rows:
        orphan_idx = max((u.idx for u in units), default=-1) + 1
        logger.warning(
            "%d 行未落入任何话题段，追加「未分类兜底」单元",
            len(orphan_rows),
        )
        units.append(TopicUnit(
            idx=orphan_idx,
            label="未分类兜底",
            rows=orphan_rows,
            row_ids=[rid for rid, _ in orphan_rows],
        ))

    return units
