"""LLM分析主模块 - 使用重构后的各个子模块完成字幕分析"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from .llm_summarize import (
    TopicSegmentationStep,
    WordLevelStep,
    RefineStep,
    ClusteringStep,
    MacroAnalysisStep,
    TradingSensitivityStep,
    TopicAnalysisStep,
)
from .llm_infra.resource import RESOURCE_MANAGER
from .llm_summarize.analysis_common import dedup_position_advice_against_skills
from .llm_summarize.schemas import SubtitleRow
from .llm_summarize.utils.llm_text_utils import mark_unimportant_segments
from .log_config import step_logger
from .utils.llm_user_id import user_id_scope


_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """字幕分析主类 - 协调各个子模块完成完整的分析流程"""

    def __init__(self) -> None:
        self._topic_segmentation_step = TopicSegmentationStep()
        self._word_level_step = WordLevelStep()
        self._refine_step = RefineStep()
        self._clustering_step = ClusteringStep()
        self._macro_analysis_step = MacroAnalysisStep()
        self._trading_sensitivity_step = TradingSensitivityStep()
        self._topic_analysis_step = TopicAnalysisStep()

    def summarize(
        self,
        output_dir: Path,
        subtitle_l: list[SubtitleRow],
        *,
        economic: bool = False,
    ) -> dict:
        """
        主入口 - 话题分割 -> 词级三阶段 -> 剪枝终判∥话题重置 -> 聚类
                -> 宏观分析 -> 交易技巧与敏感信号 -> 主题分析

        经济版（economic=True）：跳过第 1/2/3/4 步预处理与聚类、第 6/7 步板块分析，
        直接用原始字幕跑宏观分析（仅四个非板块字段组），落盘到 llm_analysis_economic。

        Args:
            output_dir: 输出目录
            subtitle_l: 字幕字典列表
        """
        # 全程包在 user_id_scope 内：拿视频级基础 user_id（断点续跑从 metadata.json 复用）。
        # 各阶段的 user_id 由 client 按 stage 派生 `{base_uid}_{stage}` 做阶段级隔离。
        with user_id_scope(output_dir):
            analysis_dir_name = "llm_analysis_economic" if economic else "llm_analysis"
            llm_analysis_dir = output_dir / analysis_dir_name
            cached_analysis_path = llm_analysis_dir / "llm_analysis.json"
            if cached_analysis_path.exists():
                logger.info("检测到缓存 %s，跳过 LLM 分析直接返回结果", cached_analysis_path)
                with open(cached_analysis_path, encoding="utf-8") as f:
                    return json.load(f)

            self._validate_subtitle_l(subtitle_l)
            logger.info("使用内存中的字幕数据，共 %d 条（过滤后）", len(subtitle_l))

            llm_analysis_dir.mkdir(parents=True, exist_ok=True)

            if economic:
                # 经济版：跳过第 1/2/3/4 步预处理与聚类、第 6/7 步板块分析，
                # 直接用原始字幕跑宏观分析（四个非板块字段组）。
                logger.info(
                    "经济版模式：跳过第 1/2/3/4 步预处理与聚类、第 6/7 步板块分析，"
                    "直接以原始字幕执行宏观分析"
                )
                analysis = self._macro_analysis_step.analyze(
                    {}, llm_analysis_dir, subtitle_l, economic=True,
                )
                with open(cached_analysis_path, "w", encoding="utf-8") as f:
                    json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
                logger.info(
                    "llm_analysis.json 已写入（经济版，%d 个顶层 section）", len(analysis),
                )
                return analysis

            # 步骤 1：话题分割
            with step_logger("第1步-话题分割"):
                segments, raw_segments, subtitle_knowledge = (
                    self._topic_segmentation_step.segment(
                        subtitle_l, llm_analysis_dir,
                    )
                )
                logger.info("话题分割完成，共 %d 个话题段", len(segments))
                if subtitle_knowledge:
                    logger.info(
                        "字幕知识库：%d 条（黑话/概念 → 字幕内解释）",
                        len(subtitle_knowledge),
                    )

                # 组剪枝：1_1 判定「不重要」的段（纯感谢/闲聊/小助理/卖课/带货）整体跳过——
                # 给其行打上「无关」合成注释，下游纠错/关键词不建工作单元，5_x 不引用。
                # 必须用 1_2 合并**前**的段列表：桥接合并会把这些段吸收进合并组，
                # 用合并后列表会漏标互动行
                subtitle_l = mark_unimportant_segments(subtitle_l, raw_segments)
                pruned_count = sum(
                    1 for seg in raw_segments if seg.是否重要 is False
                )
                if pruned_count:
                    logger.info("组剪枝：%d 个「不重要」话题段跳过，后续不再处理", pruned_count)

            # 步骤 2：词级三阶段（2_1 粗筛 / 2_2 词表判定 / 2_3 联网确认）。
            # 合并了旧的「演讲稿清洗」与「逐行标注」两步——「是黑话还是错字」本是同一个
            # 判断，拆开会让前一步在无词表时先猜、猜错方向后浪费大量联网搜索。
            # batch 间并行、batch 内 2_1→[2_2→2_3] 串行；收尾做两件确定性的事：
            # 确证错字回写 `text`、已确认指代物化进 `text_llm`（第 3 步起的 [DATA] 底稿）。
            with step_logger("第2步-词级三阶段"):
                corrected_subtitle_l, step2_conclusions = self._word_level_step.process(
                    subtitle_l, segments, llm_analysis_dir, output_dir,
                    subtitle_knowledge=subtitle_knowledge,
                )
                logger.info("词级三阶段完成，保留 %d 条字幕", len(corrected_subtitle_l))

            # 步骤 3：3_1 剪枝与关键词终判 ∥ 3_2 话题重置（**两者并行**）。
            # 都在已纠错、指代已内联的字幕上重做判定，但改的维度不同且互不依赖：
            # 3_1 改逐行类型与关键词（按 rowID 键控），3_2 改话题段分割（按段区间键控）。
            # 3_1 的分批按 3_2 之前的段做——段列表只决定"怎么分批"，不决定"处理哪些行"。
            with step_logger("第3步-剪枝终判与话题重置"):
                processed_subtitle_l, segments = self._refine_step.process(
                    corrected_subtitle_l, segments, step2_conclusions, llm_analysis_dir,
                )
                logger.info(
                    "第 3 步完成：%d 条字幕带最终注释，话题段 %d 个",
                    len(processed_subtitle_l), len(segments),
                )

            # 步骤 4：聚类（注释聚合 / 主题去重 4_1~4_3 / 候选生成 4_4~4_7 /
            # 候选剪枝 4_8 / 两层压缩 4_9 / 引用扩展 4_10 / 行号转时间）
            with step_logger("第4步-聚类"):
                keyword_data = self._clustering_step.run(
                    processed_subtitle_l, segments, llm_analysis_dir,
                )
                logger.info("聚类完成，共 %d 个金融主题", len(keyword_data))

            # 步骤 5/6/7 并行：三步互不依赖（敏感信号判定/聚合已归步骤 6，与主题分析
            # 无依赖）。各阶段 user_id 由 client 按 stage 自动派生（每阶段独立
            # 命名空间），无需在此显式隔离。三路完成即各自落盘 `N_analysis_result.json`；
            # 最终 `llm_analysis.json` 由本处 join 后合并写（避免三步并发写同一文件）。
            futures = {
                "5": RESOURCE_MANAGER.submit(
                    self._macro_analysis_step.analyze,
                    keyword_data, llm_analysis_dir, processed_subtitle_l,
                ),
                "6": RESOURCE_MANAGER.submit(
                    self._trading_sensitivity_step.analyze,
                    keyword_data, llm_analysis_dir, processed_subtitle_l,
                ),
                "7": RESOURCE_MANAGER.submit(
                    self._topic_analysis_step.analyze,
                    keyword_data, llm_analysis_dir, processed_subtitle_l,
                ),
            }
            step_results: dict[str, dict] = {}
            for step, future in futures.items():
                step_results[step] = future.result()  # 任一失败整体上抛（其余已落盘）
            logger.info("步骤 5/6/7 并行完成，合并三路 section")

            analysis = {
                **step_results["5"],
                **step_results["6"],
                **step_results["7"],
            }

            # 跨步骤归属仲裁：三路并行时彼此看不到对方产物，5_3/7_2 的引用校验也只判
            # 「引用是否支撑观点」、不判「该不该属于这个字段」。此处是三路产物第一次
            # 同时可见的地方，删掉宏观「仓位建议」里其实属于「交易技巧」的条目。
            dedup_position_advice_against_skills(analysis)

            # 最终产物：整管线缓存标记，下游渲染/聚合/评测/目录同步的唯一事实源
            with open(cached_analysis_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
            logger.info("llm_analysis.json 已写入（合并 %d 个顶层 section）", len(analysis))

            return analysis

    def _validate_subtitle_l(self, subtitle_l: list[SubtitleRow]) -> None:
        if subtitle_l is None:
            raise ValueError("subtitle_l参数不能为None")

        if not isinstance(subtitle_l, list):
            raise TypeError(f"subtitle_l参数必须是列表类型，实际类型: {type(subtitle_l)}")

        if len(subtitle_l) == 0:
            raise ValueError("subtitle_l参数不能为空列表")

        # 验证并过滤无效条目
        valid_count = 0
        expected_row_id = 1
        for item in subtitle_l:
            if not isinstance(item, SubtitleRow):
                continue
            row_id = item.rowID
            if row_id != expected_row_id:
                raise ValueError(f"subtitle_l中rowID不是从1开始连续的，期望{expected_row_id}，实际{row_id}")
            text = item.text
            if text and str(text).strip():
                valid_count += 1
            expected_row_id += 1

        if valid_count == 0:
            raise ValueError("subtitle_l中没有有效字幕数据，所有条目都无效")

