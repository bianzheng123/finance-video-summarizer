"""步骤 1 话题分割编排器。

在（预清洗后的）完整字幕上做 1_1 话题分割 + 1_2 片段合并，产出话题段列表，
后续清洗纠错 / 指代解析 / 聚类都以话题段为工作单元并行执行。

断点续跑：
- `1_1_topic_segmentation.json`：1_1 各 chunk LLM 输出整合 + 原始段列表，
  存在即复用；
- `1_2_segment_merging.json`：1_2 各 pair LLM 输出整合 + 合并后段列表，
  存在即跳过整个话题分割阶段。

返回 `(合并后 segments, 1_1 原始 segments)`：原始 segments 供组剪枝使用——
1_2 桥接合并会把「不重要」段吸收进合并组（不再出现在合并后列表里），
剪枝标记必须按合并前的段列表打，互动行才不会混入分析。
"""

import logging
from pathlib import Path

from .segment_merger import SegmentMerger
from .topic_segmenter import TopicSegmenter
from ..schemas import SubtitleRow, TopicSegment
from ...llm_infra.io import read_json, write_integrated
from ...log_config import step_logger

logger = logging.getLogger(__name__)


class TopicSegmentationStep:
    """步骤 1 编排器：话题分割 + 片段合并。"""

    def __init__(self) -> None:
        self._segmenter = TopicSegmenter()
        self._segment_merger = SegmentMerger()

    def segment(
        self,
        subtitle_l: list[SubtitleRow],
        base_dir: Path | None = None,
    ) -> tuple[list[TopicSegment], list[TopicSegment], list[dict]]:
        """话题分割 + 片段合并，返回 `(合并后 segments, 1_1 原始 segments, 字幕知识)`。"""
        merged_path = (
            base_dir / "1_2_segment_merging.json" if base_dir is not None else None
        )
        raw_path = (
            base_dir / "1_1_topic_segmentation.json" if base_dir is not None else None
        )
        if merged_path is not None and merged_path.exists():
            merged = self._load_result_list(merged_path)
            if merged is not None:
                raw = self._load_raw_segments(raw_path, merged)
                knowledge = self._load_knowledge(merged_path)
                with step_logger("第1_1步-话题分割"):
                    logger.info("检测到 1_2_segment_merging.json，跳过话题分割阶段")
                return merged, raw, knowledge
            logger.info("1_2_segment_merging.json 无效，重跑话题分割")

        knowledge_1_1: list[dict] = []
        with step_logger("第1_1步-话题分割"):
            segments = self._load_raw_segments(raw_path, None)
            chunk_outputs: dict[str, dict] = {}
            if segments is not None:
                logger.info(
                    "检测到 1_1_topic_segmentation.json，复用话题分割（%d 段）",
                    len(segments),
                )
                knowledge_1_1 = self._load_knowledge(raw_path)
            else:
                segments, chunk_outputs, knowledge_1_1 = self._segmenter.segment(
                    subtitle_l, base_dir,
                )
                if raw_path is not None:
                    write_integrated(
                        raw_path, chunk_outputs,
                        [s.model_dump() for s in segments],
                        extra={"字幕知识": knowledge_1_1},
                    )
                logger.info("%d 个话题段", len(segments))
            raw_segments = list(segments)

        with step_logger("第1_2步-片段合并"):
            segments, pair_outputs, knowledge_1_2 = self._segment_merger.merge(
                segments, subtitle_l, base_dir,
            )
            subtitle_knowledge = _merge_knowledge(knowledge_1_1, knowledge_1_2)
            if merged_path is not None:
                write_integrated(
                    merged_path, pair_outputs,
                    [s.model_dump() for s in segments],
                    extra={"字幕知识": subtitle_knowledge},
                )
            logger.info(
                "合并后 %d 个话题段，字幕知识 %d 条",
                len(segments), len(subtitle_knowledge),
            )
        return segments, raw_segments, subtitle_knowledge

    @staticmethod
    def _load_result_list(path: Path) -> list[TopicSegment] | None:
        """读整合文件的「结果」节（须为非空 segments 列表），回读为模型。"""
        data = read_json(path)
        if (
            isinstance(data, dict)
            and isinstance(data.get("结果"), list)
            and data["结果"]
        ):
            return [TopicSegment.model_validate(seg) for seg in data["结果"]]
        return None

    @classmethod
    def _load_raw_segments(
        cls,
        raw_path: Path | None,
        fallback: list[TopicSegment] | None,
    ) -> list[TopicSegment] | None:
        """读 1_1 原始段列表；缺失/无效时回退 fallback（剪枝标记降级为用合并后段）。"""
        if raw_path is not None and raw_path.exists():
            raw = cls._load_result_list(raw_path)
            if raw is not None:
                return raw
            logger.warning("1_1_topic_segmentation.json 缺失/无效")
        if fallback is not None:
            logger.warning(
                "无 1_1 原始段列表，组剪枝降级为按合并后段列表标注"
                "（桥接吸收的互动行将不会被剪枝）"
            )
            return list(fallback)
        return None

    @staticmethod
    def _load_knowledge(path: Path | None) -> list[dict]:
        """从检查点恢复「字幕知识」（顶层「字幕知识」键，缺省空列表）。"""
        data = read_json(path) if path is not None else None
        if isinstance(data, dict):
            k = data.get("字幕知识")
            if isinstance(k, list):
                return [e for e in k if isinstance(e, dict)]
        return []


def _merge_knowledge(a: list[dict], b: list[dict]) -> list[dict]:
    """合并 1_1 与 1_2 的字幕知识，按「词」去重（先到先得，1_1 优先）。"""
    merged: list[dict] = []
    seen: set[str] = set()
    for e in (*a, *b):
        if not isinstance(e, dict):
            continue
        word = (e.get("词") or "").strip()
        if not word or word in seen:
            continue
        seen.add(word)
        merged.append(e)
    return merged
