"""经济版分析入口：跳过预处理/聚类/主题，直接以原始字幕跑宏观分析 + 板块分析。

与完整版 ``llm_summarize`` 完全独立（均为刻意设计）：
- 不跑第 1/2/3/4 步预处理与聚类、第 6/7 步交易技巧/主题分析；
- 宏观分析（step1，`step1_macro/`）+ 板块分析（step2，`step2_sector/`）都是经济版
  **独立步骤**（不复用完整版 `step5_macro`），直接对整场字幕按字段组分析，prompt 结构
  为 `[DATA]` 前（完整字幕成为 KV 缓存稳定前缀）；
- 生成阶段 7 个字段组（宏观 4 + 板块 3）共享同一 user_id（`{base_uid}_economic`），
  完整字幕只预热一次，后续字段组命中「system + 完整字幕」前缀；
- 绑定经济版独立 system prompt（本包根目录 `shared_system.txt`，`[DATA]` 前协议）。

产物落盘到 ``llm_analysis_economic/llm_analysis.json``，渲染层据此切到
``rendering_economic/``。
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from ..llm_infra.resource import fanout
from ..llm_summarize.schemas import SubtitleRow, validate_subtitle_l
from ..utils.llm_user_id import user_id_scope
from ..utils.llm_system_prompt import load_system_prompt_file, system_prompt_scope
from .step1_macro import MacroAnalysisStep
from .step2_sector import SectorAnalysisStep

_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)

# 经济版 system prompt：位于本包根目录（与完整版 llm_summarize/shared_system.txt 分离）。
_ECONOMIC_SYSTEM_PROMPT_PATH = Path(__file__).parent / "shared_system.txt"


class EconomicAnalyzer:
    """经济版：跳过预处理/聚类/主题，直接以原始字幕跑宏观分析（step1）+ 板块分析（step2）。"""

    def __init__(self) -> None:
        self._macro_analysis_step = MacroAnalysisStep()
        self._sector_analysis_step = SectorAnalysisStep()

    def summarize(self, output_dir: Path, subtitle_l: list[SubtitleRow]) -> dict:
        """生成经济版宏观分析 + 板块分析，落盘 ``llm_analysis_economic/llm_analysis.json``。

        Args:
            output_dir: 视频输出目录。
            subtitle_l: ASR 原始字幕（未经纠错/指代内联）。
        """
        with system_prompt_scope(load_system_prompt_file(_ECONOMIC_SYSTEM_PROMPT_PATH)), user_id_scope(output_dir):
            llm_analysis_dir = output_dir / "llm_analysis_economic"
            cached_analysis_path = llm_analysis_dir / "llm_analysis.json"
            if cached_analysis_path.exists():
                logger.info("检测到缓存 %s，跳过 LLM 分析直接返回结果", cached_analysis_path)
                with open(cached_analysis_path, encoding="utf-8") as f:
                    return json.load(f)

            validate_subtitle_l(subtitle_l)
            logger.info("使用内存中的字幕数据，共 %d 条（过滤后）", len(subtitle_l))

            llm_analysis_dir.mkdir(parents=True, exist_ok=True)

            # 宏观分析（step1）与板块分析（step2）并行：输入同为整场字幕、
            # 输出 section 独立、落盘文件独立（1_*/2_*），互不依赖。二者各自
            # 内部字段组仍由 fanout 并行；顶层 2 个父任务共享全局线程池。
            futures = fanout("e", [
                (self._macro_analysis_step.analyze, (llm_analysis_dir, subtitle_l), {}),
                (self._sector_analysis_step.analyze, (llm_analysis_dir, subtitle_l), {}),
            ])
            macro_analysis = futures[0].result()
            sector_analysis = futures[1].result()
            analysis = {**macro_analysis, **sector_analysis}
            with open(cached_analysis_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
            logger.info(
                "llm_analysis.json 已写入（经济版，%d 个顶层 section）", len(analysis),
            )
            return analysis
