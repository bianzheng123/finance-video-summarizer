"""6_1 交易技巧 - LLM 调用封装"""

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ...llm_infra.client import LLMClient
from ...llm_infra.schema import get_step_output_dir
from ...log_config import step_logger

logger = logging.getLogger(__name__)

_COMMON_DIR = Path(__file__).parent.parent / "analysis_common"


def _sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "_").replace("\\", "_").replace(":", "_")
        .replace("*", "_").replace("?", "_").replace('"', "_")
        .replace("<", "_").replace(">", "_").replace("|", "_")
    )


class TradingSkillAnalyzer:
    """6_1 交易技巧。

    输入完整字幕，输出主讲人提到的交易技巧 / 操作策略 / 看盘要点等结构化字段。

    与其它阶段不同，本阶段**保留整场完整字幕**作为 `[DATA]`：技巧类内容散布全场、
    且往往是对某段行情的复盘，切片会让模型看不到被复盘的那段行情。
    """

    SECTION_NAME = "交易技巧"
    STEP_NUMBER_STR = "6_1"
    STEP_NUMBER = 6

    def __init__(self, llm_client: LLMClient, prompt: str, schema: dict) -> None:
        self._llm_client = llm_client
        self._prompt = prompt
        self._schema = schema
        _jinja_env = Environment(
            loader=FileSystemLoader([str(Path(__file__).parent), str(_COMMON_DIR)])
        )
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("6_1_交易技巧_user.jinja2")

    def analyze(
        self,
        subtitle_text: str,
        base_dir: Path,
        annotation_text: str = "",
        result_sink: dict[str, dict] | None = None,
    ) -> tuple[str, dict]:
        safe_section_name = _sanitize_filename(self.SECTION_NAME)
        # 阶段标签在本方法内自打：本方法会被丢进线程池执行，
        # 提交点的 with 块管不到 worker 线程
        with step_logger(f"第{self.STEP_NUMBER_STR}步-{self.SECTION_NAME}"):
            user_prompt = self._user_template.render(
                task_rules=self._prompt,
                subtitle_text=subtitle_text,
                annotation_text=annotation_text,
            )

            log_dir = (
                get_step_output_dir(base_dir, self.STEP_NUMBER)
                if base_dir is not None else None
            )
            log_name = f"{self.STEP_NUMBER_STR}_{safe_section_name}"
            result = self._llm_client.call(
                user_prompt=user_prompt,
                temperature=0.0,
                schema=self._schema,
                log_dir=log_dir,
                log_name=log_name,
                step_name=f"{self.STEP_NUMBER_STR}_{self.SECTION_NAME}",
            )
            if result_sink is not None:
                result_sink[log_name] = result
            return self.SECTION_NAME, result
