"""敏感信号分析 - 包装 sensitivity_detector，供步骤 6 编排器调用。

实际工作仍委托给 sensitivity_detector.detect_and_annotate：
对 keyword_data 各主题原地挂上 `敏感提示` 字段，供后续 SensitivityAggregator 聚合。
analyze() 返回的 (name, {}) 是占位形态，调用方只取挂载副作用、忽略返回值。
"""

from pathlib import Path

from ..schemas import SubtitleRow, TopicData
from .sensitivity_detector import detect_and_annotate


class SensitivityAnalyzer:
    """敏感信号判定的薄包装，供步骤 6 编排器调用。"""

    SECTION_NAME = "敏感信号检测"

    @staticmethod
    def analyze(
        keyword_data: dict[str, TopicData],
        subtitle_l: list[SubtitleRow],
        base_dir: Path,
        result_sink: dict[str, dict] | None = None,
    ) -> tuple[str, dict]:
        detect_and_annotate(
            keyword_data, subtitle_l, base_dir=base_dir, result_sink=result_sink,
        )
        return SensitivityAnalyzer.SECTION_NAME, {}
