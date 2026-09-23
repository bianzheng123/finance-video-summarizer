"""词级阶段的公共部件：词表/拼音索引加载、修改应用与校验、逐行注释组装。

这里的内容从旧三阶段实现（2_1/2_2/2_3 的 `subtitle_cleaner_corrector.py`）中保留下来——
它们与阶段划分无关，是踩过坑攒下来的确定性逻辑：

- `HotwordResources`：拼音索引（严格/宽松/吞音）+ 三源黑话表，供预清洗与 2_2/2_3/3_1 复用
- `apply_mods_to_rows` / `filter_invalid_mods`：修改回写前的锚定校验（修正前词组必须是
  当前行原文子串，否则丢弃，防止 `str.replace` 静默落空）
- `build_row_annotations`：逐行注释组装（类型 + 纠错 + 指代 + 关键词），第 3 步收尾调用
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .pinyin_matcher import (
    _build_hotword_pinyin_index,
    _build_truncation_index,
)
from .vocab_hints import build_jargon_map
from ..schemas import SubtitleRow
from ...llm_infra.clients import AkshareMarketClient

logger = logging.getLogger(__name__)

_DATABASE_DIR = Path(__file__).parent.parent.parent.parent / "database"
_HOTWORD_ENHANCE_PATH = _DATABASE_DIR / "hotword_stock_enhance.json"


class HotwordResources:
    """热词/黑话/拼音索引资源包（进程内构建一次，2_x / 3_1 共用）。

    - `jargon_map`：三源合并黑话表（sector + stock + hotspot），供 [DATABASE] 与门控扫描
    - `pinyin_index` / `truncation_index`：同音 / 吞音候选索引，2_2/2_3 的拼音候选用
    """

    def __init__(self) -> None:
        try:
            with open(_HOTWORD_ENHANCE_PATH, encoding="utf-8") as f:
                hotwords_by_weight: dict = json.load(f)
        except Exception as e:  # noqa: BLE001 —— 词表缺失降级，不中断纠错
            logger.warning("加载 hotword_stock_enhance.json 失败: %s", e)
            hotwords_by_weight = {}

        self.jargon_map: dict[str, list[str]] = build_jargon_map()
        # 候选索引门槛 10：它只喂 `[ANNOTATION]` 提示，有 LLM 复核 + 写盘门禁兜底，
        # 覆盖面比精确度重要。门槛 100 时「昇腾」「铜价」这类权重 10 的常见词
        # 根本不在索引里——判据放宽多少都扫不到，因为比对目标不存在。
        self.pinyin_index = _build_hotword_pinyin_index(
            hotwords_by_weight, self.jargon_map,
            min_weight=10, min_word_len=2, max_word_len=4,
        )
        self.truncation_index = _build_truncation_index(
            hotwords_by_weight, self.jargon_map, min_weight=100, min_word_len=3,
        )
        # 市场快照客户端惰性创建：只有 2_2 真正需要新闻上下文时才拉取，
        # 避免断点命中 / 跳过 2_2 的场景仍触发 AKShare 网络请求。
        self._market_client: AkshareMarketClient | None = None
        self._market_news_cache: list[dict] | None = None

    def get_market_news(self, limit: int = 10) -> list[dict]:
        """当日财经快讯（惰性拉取 + 进程内缓存，失败返回空列表）。

        供 2_2 词表判定的 [DATABASE] 块注入：黑话消歧有时需要"当天市场在炒什么"
        作背景。网络请求失败只 warning 并返回空——热点是增强非刚需，不能阻塞纠错。
        """
        if self._market_news_cache is None:
            try:
                if self._market_client is None:
                    self._market_client = AkshareMarketClient()
                self._market_news_cache = self._market_client.get_market_news(limit)
            except Exception as e:  # noqa: BLE001 —— 网络请求，失败降级为空
                logger.warning("获取当日市场快讯失败，本次 2_2 不带新闻上下文：%s", e)
                self._market_news_cache = []
        return self._market_news_cache


def apply_mods_to_rows(
    rows: list[tuple[int, SubtitleRow]], mods: list[dict],
) -> dict[int, SubtitleRow]:
    """把修改记录按行应用到话题行上，返回 {rowID: 纠错后行}。

    修改记录按 rowID 归组后依次 `.replace`（同行多处的顺序即记录顺序）。
    行为浅拷贝（``model_copy``），不改动入参。
    """
    mods_by_line: dict[int, list[dict]] = {}
    for mod in mods:
        row_id = mod.get("rowID")
        if row_id:
            mods_by_line.setdefault(int(row_id), []).append(mod)

    corrected: dict[int, SubtitleRow] = {}
    for row_id, sub in rows:
        sub_copy = sub.model_copy()
        for mod in mods_by_line.get(row_id, []):
            before = mod.get("修正前词组")
            after = mod.get("修正后词组")
            if before and after:
                sub_copy.text = sub_copy.text.replace(before, after)
        corrected[row_id] = sub_copy
    return corrected


def _batch_tag(rid: int, row_to_batch: dict[int, int] | None) -> str:
    """批次后缀「（批次 bN）」；行号不在映射内（如非工作单元行）时返回空串。"""
    if not row_to_batch or rid not in row_to_batch:
        return ""
    return f"（批次 b{row_to_batch[rid]}）"


def filter_invalid_mods(
    rows: list[tuple[int, SubtitleRow]],
    mods: list[dict],
    row_to_batch: dict[int, int] | None = None,
) -> list[dict]:
    """按记录顺序重放修改，丢弃无效项。

    只丢弃「修正前词组不在当前行文本中」的（模型不守约时 `str.replace` 会静默
    落空）。稳定口语表达保护与同片段跨行统一已移除——纠错判定全权交给 LLM，
    代码侧只留「锚定校验」这一道防改坏的硬门禁。

    row_to_batch 为 {rowID: batch_idx} 映射，仅用于丢弃 warning 标注发生批次。
    """
    text_by_id: dict[int, str] = {
        row_id: sub.text for row_id, sub in rows
    }
    kept: list[dict] = []
    for mod in mods:
        row_id = mod.get("rowID")
        before = (mod.get("修正前词组") or "").strip()
        after = (mod.get("修正后词组") or "").strip()
        if not row_id or not before or not after or before == after:
            continue
        rid = int(row_id)
        batch_tag = _batch_tag(rid, row_to_batch)
        text = text_by_id.get(rid, "")
        if before not in text:
            logger.warning(
                "丢弃锚定失败的修改：rowID=%s 「%s」不在当前行文本中%s",
                rid, before, batch_tag,
            )
            continue
        text_by_id[rid] = text.replace(before, after)
        kept.append(mod)
    return kept


def _referents_by_row(referents: list[dict]) -> dict[int, list[dict]]:
    """referents → {rowID: [精简指代条目]}（annotation["指代"] 与动态内联共用格式）。

    精简条目只保留 `build_inlined_text` 动态内联需要的字段：`原先的词`/`指代`/
    `依据`/`困惑`/`出现序号`（>1 才保留）。困惑条目允许空指代（待后续阶段消歧）。
    """
    out: dict[int, list[dict]] = {}
    for ref in referents:
        if not isinstance(ref, dict):
            continue
        row_id = ref.get("rowID")
        if row_id is None:
            continue
        entry = {k: ref[k] for k in ("原先的词", "指代", "依据") if ref.get(k)}
        if not entry.get("原先的词"):
            continue
        if not entry.get("指代") and ref.get("困惑") is not True:
            continue
        if ref.get("困惑") is True:
            entry["困惑"] = True
        if (occ := ref.get("出现序号")) and occ > 1:
            entry["出现序号"] = occ
        out.setdefault(int(row_id), []).append(entry)
    return out


def build_row_annotations(
    corrected_rows_by_id: dict[int, SubtitleRow],
    mods: list[dict],
    exceptions: dict[int, str],
    referents: list[dict],
    keywords: list[dict] | None = None,
) -> dict[int, dict]:
    """组装逐行注释：类型 + 纠错（按行归组去重）+ 指代 + 关键词。

    纠错列表用注释格式的键名「修改前/修改后」（修改记录里是「修正前词组/修正后
    词组」）。无关/噪音行不纠错、不指代、不标关键词，三个列表为空。
    keywords 为 None 时不写关键词字段（尚未做关键词提取的调用点）。
    """
    corrections_by_row: dict[int, list[dict]] = {}
    for mod in mods:
        row_id = mod.get("rowID")
        before = (mod.get("修正前词组") or "").strip()
        after = (mod.get("修正后词组") or "").strip()
        if not row_id or not before or not after or before == after:
            continue
        rid = int(row_id)
        bucket = corrections_by_row.setdefault(rid, [])
        if not any(c["修改前"] == before and c["修改后"] == after for c in bucket):
            bucket.append({"修改前": before, "修改后": after})

    referents_by_row = _referents_by_row(referents)

    keywords_by_row: dict[int, list[dict]] = {}
    for kw in keywords or []:
        if not isinstance(kw, dict):
            continue
        row_id = kw.get("rowID")
        if row_id is None:
            continue
        word = (kw.get("词") or "").strip()
        formal = (kw.get("正式名") or "").strip()
        if not word or not formal:
            continue
        entry = {
            "词": word,
            "正式名": formal,
            "类型": kw.get("类型") or "",
        }
        keywords_by_row.setdefault(int(row_id), []).append(entry)

    annotations: dict[int, dict] = {}
    for row_id in corrected_rows_by_id:
        row_type = exceptions.get(row_id, "重点")
        annotations[row_id] = {
            "类型": row_type,
            "纠错": corrections_by_row.get(row_id, []) if row_type == "重点" else [],
            "指代": referents_by_row.get(row_id, []) if row_type == "重点" else [],
        }
        if keywords is not None:
            annotations[row_id]["关键词"] = (
                keywords_by_row.get(row_id, []) if row_type == "重点" else []
            )
    return annotations
