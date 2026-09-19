"""主题语义等价合并（4_1 向量粗筛 + 4_2 LLM 二元结论+困惑度判定）。

替代旧实现「4_1 分块 LLM 去重 + 4_2 串行全局 LLM 去重」——旧方案把全部主题
（约百来个）交给 LLM 做 O(N²) 自比，reasoning 预算打满、慢且贵。新方案两级：

- **4_1 向量主题去重**：对所有主题名调用 embedding（腾讯云 TokenHub
  `kinfra-text-embedding-4b`，2560 维），本地算两两余弦相似度，相似度 ≥
  `threshold`（`parameter_config.json` 的 `embedding_dedup.threshold`，默认 0.6）
  的主题对作为「候选等价对」产出。**不调用 LLM、不合并**——embedding 相似度无法
  区分「等价」与「父子/上下游」（实测两者分数重叠），粗筛只做召回、砍掉大量
  明显无关的两两比较。
- **4_2 关键词合并**：对 4_1 的候选等价对做**批量并行** LLM 判定（每个候选对带
  相似度分数作线索），输出「结论（等价/不等价）+ 困惑度（bool）+ 保留」——
  结论=等价且困惑度=false 的合并（保留指定方向）、结论=不等价且困惑度=false 的
  丢弃、困惑度=true 的导出给 4_3 精确去重。

本阶段调用不发送 system prompt、不携带 [DATA]/[ANNOTATION]——user prompt 只含
[TASK] 规则与【候选等价对】（主题名 + 类型 + 相似度）。4_2 的 user_id 由 client
按 stage 自动派生为 `{base_uid}_4_2`，与其它阶段天然隔离。
"""

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ...log_config import step_logger
from ...llm_infra.batch import (
    batch_units,
    run_batch_with_fallback,
    split_batch_result,
)
from ...llm_infra.io import read_json, write_integrated
from ...llm_infra.clients import EmbeddingClient
from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ...llm_infra.config import get_batch_size, get_embedding_threshold
from ...llm_infra.resource import fanout
from ..utils.llm_text_utils import load_prompt
from ..schemas import TopicData
from .event_topics import is_keyword_event, is_segment_event

logger = logging.getLogger(__name__)

# 4_2 结论枚举（与 schema 的 enum 对齐，防模型编造）；「困惑」不在此列——
# 它由独立的 `困惑度` bool 表达。
_CONCLUSIONS = ("等价", "不等价")


def _cosine(a: list[float], b: list[float]) -> float:
    """两向量的余弦相似度（零向量返回 0.0，避免除零）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class TopicDeduper:
    """4_1 向量主题去重（embedding 粗筛）+ 4_2 关键词合并（结论+困惑度判定）。"""

    def __init__(self) -> None:
        self._prompt = load_prompt("4_2_关键词合并.txt", __file__)
        self._schema = _load_schema("4_2_关键词合并", __file__)

        _jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._tmpl = _jinja_env.get_template("4_2_关键词合并_user.jinja2")

        self._llm_client = LLMClient()
        self._threshold = get_embedding_threshold()
        # embedding 客户端惰性创建：只有 4_1 真正需要时才读 key，避免断点命中
        # （4_1 跳过）时仍要求配好 TENCENT_LLM_API_KEY。
        self._embedding_client: EmbeddingClient | None = None

    # ==================== 4_1 向量主题去重 ====================

    def dedup_topics_vector(
        self,
        keyword_data: dict[str, TopicData],
        base_dir: Path | None,
    ) -> tuple[dict[str, TopicData], list[dict]]:
        """4_1 向量主题去重：embedding + 余弦相似度粗筛候选等价对。

        返回 `(keyword_data, 候选对)`——keyword_data 不在此步合并，原样返回；
        候选对每项形如 `{对ID, 关键词1, 关键词2, 类型1, 类型2, 相似度}`。
        整合落盘 `4_1_embedding_dedup.json`（含 threshold、候选对、统计）。
        """
        with step_logger("第4_1步-向量主题去重"):
            if base_dir is not None:
                dedup_path = base_dir / "4_1_embedding_dedup.json"
                if dedup_path.exists():
                    cached = read_json(dedup_path)
                    if isinstance(cached, dict) and isinstance(cached.get("结果"), dict):
                        pairs = cached.get("候选对")
                        logger.info("检测到 4_1_embedding_dedup.json，跳过向量主题去重")
                        return (
                            {n: TopicData.model_validate(td) for n, td in cached["结果"].items()},
                            pairs if isinstance(pairs, list) else [],
                        )

            names = sorted(keyword_data)
            if len(names) < 2:
                return keyword_data, []

            vectors = self._get_embedding_client().embed(names)
            pairs: list[dict] = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    sim = _cosine(vectors[i], vectors[j])
                    if sim < self._threshold:
                        continue
                    pairs.append({
                        "对ID": f"p{len(pairs):04d}",
                        "关键词1": names[i],
                        "关键词2": names[j],
                        "类型1": keyword_data[names[i]].类型,
                        "类型2": keyword_data[names[j]].类型,
                        "相似度": round(sim, 4),
                    })

            logger.info(
                "4_1 向量主题去重：%d 个主题 → %d 个候选等价对（threshold=%.2f）",
                len(names), len(pairs), self._threshold,
            )

            if base_dir is not None:
                write_integrated(
                    base_dir / "4_1_embedding_dedup.json",
                    {
                        "embedding": {
                            "模型": self._get_embedding_client()._MODEL,
                            "维度": len(vectors[0]) if vectors else 0,
                        }
                    },
                    {n: td.model_dump() for n, td in keyword_data.items()},
                    extra={
                        "threshold": self._threshold,
                        "候选对": pairs,
                        "统计": {
                            "主题数": len(names),
                            "两两对数": len(names) * (len(names) - 1) // 2,
                            "候选对数": len(pairs),
                        },
                    },
                )
            return keyword_data, pairs

    # ==================== 4_2 关键词合并 ====================

    def dedup_topics_merge(
        self,
        candidate_pairs: list[dict],
        keyword_data: dict[str, TopicData],
        base_dir: Path | None,
    ) -> tuple[dict[str, TopicData], list[dict], list[dict]]:
        """4_2 关键词合并：对候选等价对批量并行判定（结论 + 困惑度）。

        返回 `(合并后 keyword_data, 困惑对)`；困惑对每项形如
        `{关键词1, 关键词2, 理由}`。整合落盘 `4_2_topic_merge.json`（困惑对存进
        额外字段「困惑对」）。
        """
        with step_logger("第4_2步-关键词合并"):
            if base_dir is not None:
                merge_path = base_dir / "4_2_topic_merge.json"
                if merge_path.exists():
                    cached = read_json(merge_path)
                    if isinstance(cached, dict) and isinstance(cached.get("结果"), dict):
                        uncertain = cached.get("困惑对")
                        merge_pairs = cached.get("合并对")
                        logger.info("检测到 4_2_topic_merge.json，跳过关键词合并")
                        return (
                            {n: TopicData.model_validate(td) for n, td in cached["结果"].items()},
                            uncertain if isinstance(uncertain, list) else [],
                            merge_pairs if isinstance(merge_pairs, list) else [],
                        )

            if not candidate_pairs:
                logger.info("4_2 无候选等价对，跳过关键词合并")
                return keyword_data, [], []

            batch_size = get_batch_size("4_2_关键词合并")
            batches = batch_units(
                candidate_pairs, batch_size, sort_key=lambda p: p["对ID"],
            )
            log_dir = get_step_output_dir(base_dir, 4) if base_dir is not None else None
            logger.info(
                "4_2 关键词合并：%d 个候选对 → %d 批（每批 ≤ %d）",
                len(candidate_pairs), len(batches), batch_size,
            )

            futures = fanout("4_2", [
                (self._run_batch_worker, (batch, log_dir, batch_idx), {})
                for batch_idx, batch in enumerate(batches)
            ])

            llm_outputs: dict[str, dict] = {}
            all_results: dict[str, dict] = {}
            for future, batch_idx in zip(futures, range(len(batches))):
                try:
                    batch_outputs, batch_results = future.result()
                except Exception as e:  # noqa: BLE001 —— 单批失败降级，不拖垮整步
                    logger.warning("4_2 第 %d 批整体失败：%s", batch_idx, e)
                    continue
                llm_outputs.update(batch_outputs)
                all_results.update(batch_results)

            # 统一应用判定：迭代合并处理传递等价，级联（主题已被合并）静默跳过。
            keyword_data, uncertain_pairs, merge_pairs = self._apply_decisions(keyword_data, all_results)

            if base_dir is not None:
                write_integrated(
                    base_dir / "4_2_topic_merge.json", llm_outputs,
                    {n: td.model_dump() for n, td in keyword_data.items()},
                    extra={"困惑对": uncertain_pairs, "合并对": merge_pairs},
                )
            return keyword_data, uncertain_pairs, merge_pairs

    def _run_batch_worker(
        self,
        batch: list[dict],
        log_dir: Path | None,
        batch_idx: int,
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """单批判定 worker（fanout 并行提交）：整批一次调用，失败拆单跑。

        返回 `(batch_outputs, results)`：batch_outputs 是 `{log_name: LLM result}`
        供整合落盘；results 是 `{对ID: 判定 payload}` 供应用判定。
        """
        outputs: dict[str, dict] = {}

        def run_batch(units: list[dict]) -> dict[str, Any]:
            result = self._call_batch(units, log_dir, f"4_2_关键词合并_b{batch_idx}")
            if isinstance(result, dict):
                outputs[f"4_2_关键词合并_b{batch_idx}"] = result
            mapping, missing = split_batch_result(
                result, "判定", "对ID", [u["对ID"] for u in units],
            )
            logger.info(
                "4_2 批次 %d：%d/%d 个候选对完成判定%s",
                batch_idx, len(mapping), len(units),
                f"（缺失 {missing}，转单跑）" if missing else "",
            )
            return mapping

        def run_unit(pair: dict) -> Any:
            result = self._call_batch([pair], log_dir, f"4_2_关键词合并_s{pair['对ID']}")
            if isinstance(result, dict):
                outputs[f"4_2_关键词合并_s{pair['对ID']}"] = result
            mapping, _ = split_batch_result(result, "判定", "对ID", [pair["对ID"]])
            return mapping.get(pair["对ID"])

        def validate_unit(pair: dict, payload: Any) -> bool:
            return (
                isinstance(payload, dict)
                and payload.get("结论") in _CONCLUSIONS
                and isinstance(payload.get("困惑度"), bool)
            )

        results, failures = run_batch_with_fallback(
            batch, len(batch),
            unit_key=lambda p: p["对ID"],
            run_batch=run_batch,
            run_unit=run_unit,
            validate_unit=validate_unit,
            batch_label="4_2 关键词合并",
        )
        for pid, exc in failures.items():
            logger.warning("4_2 候选对 %s 判定失败：%s（保持不合并）", pid, exc)
        return outputs, results

    def _call_batch(
        self,
        pairs: list[dict],
        log_dir: Path | None,
        log_name: str,
    ) -> dict | str:
        """渲染 4_2 候选对模板并调用 LLM（无 system prompt）。"""
        user_prompt = self._tmpl.render(task_rules=self._prompt, pairs=pairs)
        return self._llm_client.call(
            user_prompt=user_prompt,
            system_prompt="",
            temperature=0.0,
            schema=self._schema,
            log_dir=log_dir,
            log_name=log_name,
            step_name="4_2_关键词合并",
        )

    # ==================== 判定应用 ====================

    def _apply_decisions(
        self,
        keyword_data: dict[str, TopicData],
        results: dict[str, dict],
    ) -> tuple[dict[str, TopicData], list[dict], list[dict]]:
        """把 4_2 的判定统一应用到 keyword_data。

        - `困惑度=true`：收集为困惑对（`{关键词1, 关键词2, 理由}`）导出待 4_3，
          不合并（无论结论为何）；合并完成后用 rename_map 回写其中的主题名。
        - `结论=等价` 且 `困惑度=false`：作为等价边收集，按「出现频率高者保留」
          确定合并方向（频率相等用模型「保留」字段兜底），迭代合并处理传递等价。
        - `结论=不等价` 且 `困惑度=false`：丢弃。

        合并过程中维护 rename_map（被合并名 → 最终存活名），合并完成后用它回写
        困惑对，避免「主题已被合并、困惑对仍引用旧名」的悬空引用。
        """
        # 先分类：等价边（含保留字段）/ 困惑对（不等价直接丢弃）。
        equiv_edges: list[tuple[str, str, str]] = []  # (关键词1, 关键词2, 保留)
        uncertain_pairs: list[dict] = []
        for pid, payload in results.items():
            if not isinstance(payload, dict):
                continue
            conclusion = payload.get("结论")
            confused = payload.get("困惑度")
            k1 = (payload.get("关键词1") or "").strip()
            k2 = (payload.get("关键词2") or "").strip()
            keep = (payload.get("保留") or "").strip()
            reason = payload.get("理由") or ""
            if not k1 or not k2 or k1 == k2:
                logger.warning("4_2 候选对 %s 关键词非法，跳过", pid)
                continue

            if confused is True:
                uncertain_pairs.append({"关键词1": k1, "关键词2": k2, "理由": reason})
            elif conclusion == "等价":
                if keep not in (k1, k2):
                    logger.warning(
                        "4_2 候选对 %s 判「等价」但保留字段非法（%r），默认保留「关键词1」",
                        pid, keep,
                    )
                    keep = k1
                equiv_edges.append((k1, k2, keep))
            # 结论=不等价：丢弃

        # 频率统计：每个关键词在所有等价边端点中的出现次数。出现次数 > 1 的
        # 关键词是多对一合并的锚点（合并后的保留名）。
        freq: dict[str, int] = {}
        for k1, k2, _ in equiv_edges:
            freq[k1] = freq.get(k1, 0) + 1
            freq[k2] = freq.get(k2, 0) + 1

        # 逐边确定合并方向：段落事件只能与段落事件或关键词事件合并（与板块/个股/
        # 指数等非事件类型不合并）；段落事件 vs 关键词事件时强制保留段落事件；其余
        # 频率高者保留，频率相等（含纯一对一）用模型「保留」字段兜底（关键词事件
        # 视作普通项，可与任意关键词合并）。
        directed: list[tuple[str, str]] = []  # (保留名, 被合并名)
        for k1, k2, keep in equiv_edges:
            k1_seg = self._is_segment_event(keyword_data, k1)
            k2_seg = self._is_segment_event(keyword_data, k2)
            k1_kw = is_keyword_event(keyword_data.get(k1))
            k2_kw = is_keyword_event(keyword_data.get(k2))
            # 段落事件不能与非事件（既非段落事件也非关键词事件）合并。
            if (k1_seg and not (k2_seg or k2_kw)) or (k2_seg and not (k1_seg or k1_kw)):
                logger.warning(
                    "4_2 段落事件与非事件不合并：「%s」↔「%s」，跳过（宁缺勿滥）",
                    k1, k2,
                )
                continue
            # 段落事件 vs 关键词事件：强制保留段落事件（关键词事件作为被合并方）。
            if k1_seg and not k2_seg:
                keep_name, merge_name = k1, k2
            elif k2_seg and not k1_seg:
                keep_name, merge_name = k2, k1
            elif freq[k1] > freq[k2]:
                keep_name, merge_name = k1, k2
            elif freq[k2] > freq[k1]:
                keep_name, merge_name = k2, k1
            else:
                keep_name, merge_name = (k2, k1) if keep == k2 else (k1, k2)
            directed.append((keep_name, merge_name))

        # 迭代合并：一个关键词合并后，引用它的其他边会级联失效（静默跳过），
        # 反复扫描直到无变化，从而覆盖 A→B、B→C 这类链式等价。同时维护
        # rename_map 的传递闭包，保证映射始终指向最终存活名。
        rename_map: dict[str, str] = {}
        merge_pairs: list[dict[str, str]] = []
        merged = True
        while merged:
            merged = False
            for keep_name, merge_name in directed:
                if keep_name not in keyword_data or merge_name not in keyword_data:
                    continue  # 级联：主题已被前面的合并吸收
                keyword_data = self._rename_topic(keyword_data, merge_name, keep_name, "")
                rename_map[merge_name] = keep_name
                # 传递闭包：已映射到 merge_name 的旧名改指到 keep_name。
                for old, cur in rename_map.items():
                    if cur == merge_name:
                        rename_map[old] = keep_name
                merge_pairs.append({"被合并名": merge_name, "保留名": keep_name})
                merged = True

        if merge_pairs:
            logger.info("4_2 合并 %d 对：%s", len(merge_pairs), merge_pairs)
        else:
            logger.info("4_2 未发现可合并的等价对")

        uncertain_pairs = self._rewrite_uncertain_pairs(uncertain_pairs, rename_map)
        if uncertain_pairs:
            logger.info("4_2 导出困惑对 %d 个，待 4_3 精确去重", len(uncertain_pairs))
        return keyword_data, uncertain_pairs, merge_pairs

    @staticmethod
    def _rewrite_uncertain_pairs(
        uncertain_pairs: list[dict],
        rename_map: dict[str, str],
    ) -> list[dict]:
        """用 rename_map 回写困惑对主题名：替换旧名 → 丢弃同名对 → 无序去重。"""
        rewritten: list[dict] = []
        seen: set[tuple[str, str]] = set()
        dropped_same = 0
        dropped_dup = 0
        for p in uncertain_pairs:
            k1 = rename_map.get(p["关键词1"], p["关键词1"])
            k2 = rename_map.get(p["关键词2"], p["关键词2"])
            if k1 == k2:
                dropped_same += 1
                continue
            key = (k1, k2) if k1 <= k2 else (k2, k1)
            if key in seen:
                dropped_dup += 1
                continue
            seen.add(key)
            rewritten.append({"关键词1": k1, "关键词2": k2, "理由": p.get("理由") or ""})
        if dropped_same or dropped_dup:
            logger.info(
                "4_2 困惑对回写：%d → %d（丢弃同名 %d、去重 %d）",
                len(uncertain_pairs), len(rewritten), dropped_same, dropped_dup,
            )
        return rewritten

    @staticmethod
    def _is_segment_event(keyword_data: dict[str, TopicData], name: str) -> bool:
        """主题是否为段落事件（一等事件主题）。关键词事件不在此列，视作普通项。"""
        return is_segment_event(keyword_data.get(name))

    @staticmethod
    def _rename_topic(
        keyword_data: dict[str, TopicData], old_name: str, new_name: str, new_type: str,
    ) -> dict[str, TopicData]:
        """把 old_name 主题改名为 new_name 并改类型；若 new_name 已存在则合并。"""
        if old_name not in keyword_data:
            return keyword_data
        old_data = keyword_data[old_name]

        if new_name in keyword_data:
            existing = keyword_data[new_name]
            old_refs = old_data.引用 or []
            new_refs = existing.引用 or []
            if new_refs and isinstance(new_refs[0], int):
                merged_set: set[int] = set()
                for r in old_refs + new_refs:
                    try:
                        merged_set.add(int(r))
                    except (TypeError, ValueError):
                        continue
                existing.引用 = sorted(merged_set)
            else:
                existing.引用 = list(new_refs) + list(old_refs)
            # 只在 existing 缺类型时才覆盖；existing 已有的类型来自 3_1 标注判断，更可信
            # （场景：'CUDA' 已存在为 '板块'，'酷大' 黑话合并进来时不应被改写为 '股票'）
            if not existing.类型 and new_type:
                existing.类型 = new_type
            # 段落事件类型不可降级：被合并方是段落事件（来自 1_1/1_2 段标注，比关键词
            # 类型更权威）时，合并结果必须仍是段落事件——否则它会以「概念」身份参与 4_4
            # 拿到板块父，在 4_5 被销毁式合并掉，整条事件弧退化成板块子项列表里的一个名字。
            # 关键词事件不享受此保护：它本就是只能作子的事件，合并时继承目标类型。
            if is_segment_event(old_data) and existing.类型 != "段落事件":
                logger.info(
                    "合并保留段落事件类型：'%s'（原 %s）← '%s'",
                    new_name, existing.类型, old_name,
                )
                existing.类型 = "段落事件"
            del keyword_data[old_name]
        else:
            # new_type 为空串表示"沿用原类型"（4_1/4_2 的调用口径）——不得覆盖成空串，
            # 否则改名即丢类型（事件被改名后会以无类型身份参与 4_4 并被销毁）
            if new_type:
                old_data.类型 = new_type
            keyword_data[new_name] = old_data
            del keyword_data[old_name]
        return keyword_data

    # ==================== helper ====================

    def _get_embedding_client(self) -> EmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient()
        return self._embedding_client
