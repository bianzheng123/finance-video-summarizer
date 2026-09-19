"""第 3 步各阶段的单阶段执行器：渲染 → 调用 → 语义校验。

- `run_3_1`：关键词提取 + 剪枝终判。它面对**已纠错且指代/黑话已内联**的字幕
  （`原词（指代）`/`别名（正式名）`），关键词正式名映射更准（「赵毅」→「兆易」
  后才可能映射到兆易创新），剪枝也不再被 ASR 噪声干扰。输出只含关键词与
  增量改判，最终无关/噪音集合由「第 2 步初判 + 改判说明」重建。
- `run_3_2`：话题重置（新阶段），对 1_1 自报困惑的段重新判定分割方式。

本模块只负责"发一次调用并把结果校验干净"，批量分组/并行编排/断点由
`refine_step.py` 承担。词表资源与拼音索引复用 `subtitle_cleaning` 的
`HotwordResources` / `build_jargon_hints`（供 3_2 用）——那是与阶段划分无关的
确定性资产，不复制一份。
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ..subtitle_cleaning.common import HotwordResources
from ..subtitle_cleaning.vocab_hints import build_jargon_hints
from ..schemas import TopicUnit
from ..utils.llm_text_utils import load_prompt
from ..utils.rowid_scope import format_scope, scope_id_set

logger = logging.getLogger(__name__)

# 本步骤的 LLM 落盘目录编号（llm_output_3）
STEP_NUMBER = 3

_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent)),
    keep_trailing_newline=True,
)
# `| tojson` 不转义非 ASCII：中文若被转成 \uXXXX，模型读到的与 [DATA] 里的
# 字面不一致，逐字校验与引用锚定都会对不上（与 topic_segmentation 的 env 同口径）
_TEMPLATE_ENV.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
_STAGES = ("3_1_关键词与剪枝", "3_2_话题重置")

# 市场状态/技术形态词：不是金融资产/板块/事件，不应作为关键词/主题提取，
# 由第 5 步宏观分析（大盘情绪/后市观点）覆盖。prompt 已排除，这里做确定性兜底，
# 防止 prompt 漏网后仍进入主题列表。
_MARKET_STATE_TERMS: frozenset[str] = frozenset({
    "双底", "牛市", "熊市", "慢牛", "快牛", "震荡市", "单边市",
    "技术性牛市", "技术性熊市",
})


class RefineStages:
    """3_1 / 3_2 的调用执行器（无状态，资源在构造时装配）。"""

    def __init__(self, resources: HotwordResources | None = None) -> None:
        self._res = resources or HotwordResources()
        self._llm = LLMClient()
        self._prompts = {n: load_prompt(f"{n}.txt", __file__) for n in _STAGES}
        self._schemas = {n: _load_schema(n, __file__) for n in _STAGES}
        self._templates = {
            n: _TEMPLATE_ENV.get_template(f"{n}_user.jinja2") for n in _STAGES
        }

    # ==================== 3_1：关键词 + 剪枝终判 ====================

    def run_3_1(
        self,
        topics: list[TopicUnit],
        subtitle_text: str,
        annotation_text: str,
        pruned: dict,
        base_dir: Path | None,
        batch_idx: int,
        log_suffix: str = "",
    ) -> dict:
        """3_1 关键词提取 + 剪枝终判（在**已纠错 + 指代/黑话已内联**字幕上跑）。

        指代与黑话的解析结果已由第 2 步物化进 `[DATA]`（`原词（指代）`/`别名（正式名）`），
        本阶段不再吃任何数据库输入，LLM 线性扫描即可。`pruned` 是第 2 步的剪枝初判，
        作为最终剪枝集合的基线，与 `改判说明` 增量合并得出终判。
        """
        all_row_ids = [rid for t in topics for rid in t.row_ids]
        user_prompt = self._templates["3_1_关键词与剪枝"].render(
            task_rules=self._prompts["3_1_关键词与剪枝"],
            topics=[
                {"idx": t.idx, "label": t.label or "",
                 "scope": format_scope(t.row_ids)}
                for t in topics
            ],
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
            rowid_scope=format_scope(all_row_ids),
        )
        log_dir = (
            get_step_output_dir(base_dir, STEP_NUMBER) if base_dir is not None else None
        )
        result = self._llm.call(
            user_prompt=user_prompt,
            temperature=0.0,
            schema=self._schemas["3_1_关键词与剪枝"],
            log_dir=log_dir,
            log_name=f"3_1_关键词与剪枝_b{batch_idx}{log_suffix}",
            step_name="3_1_关键词与剪枝",
        )
        return self._validate_3_1(result, all_row_ids, subtitle_text, pruned)

    def _validate_3_1(
        self,
        result: dict,
        row_ids: list[int],
        subtitle_text: str,
        pruned: dict,
    ) -> dict:
        """3_1 校验：行号范围、词为子串、剪枝增量重建、保护性校验。"""
        scope = scope_id_set(row_ids)
        text_by_id = text_by_row(subtitle_text)

        keywords: list[dict] = []
        seen: set[tuple[int, str, str]] = set()
        for kw in result.get("关键词列表") or []:
            rid = kw.get("rowID")
            if rid is None or int(rid) not in scope:
                logger.warning("3_1 关键词行号 %s 越界，丢弃", rid)
                continue
            rid = int(rid)
            word = (kw.get("词") or "").strip()
            formal = (kw.get("正式名") or "").strip()
            if not word or not formal:
                continue
            if formal in _MARKET_STATE_TERMS:
                logger.info(
                    "3_1 关键词「%s」（正式名「%s」）是市场状态词，丢弃",
                    word, formal,
                )
                continue
            if word not in text_by_id.get(rid, ""):
                logger.warning("3_1 关键词「%s」不是 row%d 原文子串，丢弃", word, rid)
                continue
            key = (rid, word, formal)
            if key in seen:
                continue
            seen.add(key)
            keywords.append({
                "rowID": rid, "词": word, "正式名": formal,
                "类型": kw.get("类型") or "",
            })

        # 剪枝增量重建：基线（第 2 步初判）+ 改判说明增量 → 最终无关/噪音集合
        irrelevant, noise = _apply_pruning_deltas(
            pruned, scope, result.get("改判说明"),
        )

        # 保护性校验：有金融实体关键词的行不该被剪掉
        entity_rows = {kw["rowID"] for kw in keywords}
        rescued = (irrelevant | noise) & entity_rows
        if rescued:
            logger.warning(
                "3_1 有 %d 行含金融实体却被判剪枝，保留为重点：%s",
                len(rescued), sorted(rescued)[:5],
            )
            irrelevant -= rescued
            noise -= rescued

        reclass = [
            r for r in (result.get("改判说明") or [])
            if isinstance(r, dict) and r.get("rowID") is not None
            and int(r["rowID"]) in scope
        ]
        return {
            "关键词列表": keywords,
            "无关行号": sorted(irrelevant),
            "噪音行号": sorted(noise),
            "改判说明": reclass,
        }

    # ==================== 3_2：话题重置 ====================

    def run_3_2(
        self,
        unit_idx: int,
        segments: list[dict],
        unit_scope: list[dict],
        subtitle_text: str,
        annotation_text: str,
        confirmed_jargon: list[dict],
        base_dir: Path | None,
        log_suffix: str = "",
        retry_hint: str = "",
    ) -> dict:
        """3_2 话题重置：对含困惑段的一个单元重新输出完整话题分割。

        Args:
            segments: 本单元的段列表（困惑段 + 上下文段，供 prompt 作参考）
            unit_scope: 本单元负责区间（`format_scope` 形态）
            confirmed_jargon: 第 2 步已确证黑话（`困惑=true` 者为未解析），已按
                本单元行号范围过滤。渲染进 [DATABASE]，让 3_2 带着"1_1 当初读
                不懂的词现在是什么"的答案重新判段。
            retry_hint: 打回重跑时携带的上轮校验错误说明（空串表示首发）。

        Returns: schema 校验后的 result dict（语义校验由 `topic_reset.py` 承担——
            它需要 rowID 全集才能判"完全覆盖"，属编排侧信息）。
        """
        # 三类黑话知识进 [DATABASE]：词表别名（本窗口命中+门控）、已确证、未解析
        jargon_hints = build_jargon_hints(self._res.jargon_map, subtitle_text)
        resolved_jargon = [c for c in confirmed_jargon if not c.get("困惑")]
        unresolved_jargon = [c for c in confirmed_jargon if c.get("困惑")]
        user_prompt = self._templates["3_2_话题重置"].render(
            task_rules=self._prompts["3_2_话题重置"],
            segments=segments,
            unit_scope=unit_scope,
            subtitle_text=subtitle_text,
            annotation_text=annotation_text,
            jargon_hints=jargon_hints,
            resolved_jargon=resolved_jargon,
            unresolved_jargon=unresolved_jargon,
            retry_hint=retry_hint,
        )
        log_dir = (
            get_step_output_dir(base_dir, STEP_NUMBER) if base_dir is not None else None
        )
        return self._llm.call(
            user_prompt=user_prompt,
            temperature=0.0,
            schema=self._schemas["3_2_话题重置"],
            log_dir=log_dir,
            log_name=f"3_2_话题重置_u{unit_idx}{log_suffix}",
            step_name="3_2_话题重置",
        )


# ==================== 模块级辅助 ====================


def text_by_row(subtitle_text: str) -> dict[int, str]:
    """从渲染后的 [DATA] 文本反解 `{rowID: 行文本}`。

    `[DATA]` 在本步骤是**两列**（`rowID\\t文本`）——行号标识列从步骤 4 起才加。
    但仍按"首列 rowID、**第二列**文本"解析而非 `parts[-1]`：三列形态下
    `parts[-1]` 会取到行号标识（`重点`/`无关`），让所有"词是否为原文子串"的
    校验全数失败。取 `parts[1]` 对两列三列都正确。
    """
    out: dict[int, str] = {}
    for line in subtitle_text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            out[int(parts[0])] = parts[1]
        except ValueError:
            continue
    return out


def _apply_pruning_deltas(
    pruned: dict,
    scope: set[int],
    reclass: list | None,
) -> tuple[set[int], set[int]]:
    """基线（第 2 步剪枝初判）+ 改判说明增量 → 最终无关/噪音集合。

    3_1 不再输出无关/噪音全量行号，只输出「改变初判」的改判说明。故最终剪枝集合
    以第 2 步初判（按 scope 过滤）为基线，逐条应用改判增量：
    - 原判 ∈ {无关, 噪音}、终判 = 重点 → 从基线移除（纠错后语义变完整，被救回）；
    - 原判 = 重点、终判 ∈ {无关, 噪音} → 加入对应集合；
    - 无关 ↔ 噪音 互换无意义（prompt 已禁止），但此处仍幂等处理。
    同一行同时属无关与噪音时判为无关（保留既有兜底）。
    """
    irrelevant: set[int] = set(pruned.get("无关") or ()) & scope
    noise: set[int] = set(pruned.get("噪音") or ()) & scope
    for r in reclass or []:
        if not isinstance(r, dict) or r.get("rowID") is None:
            continue
        try:
            rid = int(r["rowID"])
        except (TypeError, ValueError):
            continue
        if rid not in scope:
            continue
        original, final = r.get("原判"), r.get("终判")
        if original == "无关":
            irrelevant.discard(rid)
        elif original == "噪音":
            noise.discard(rid)
        if final == "无关":
            irrelevant.add(rid)
        elif final == "噪音":
            noise.add(rid)
    both = irrelevant & noise
    if both:
        logger.warning("3_1 有 %d 行同时标为无关与噪音，判为无关", len(both))
        noise -= both
    return irrelevant, noise
