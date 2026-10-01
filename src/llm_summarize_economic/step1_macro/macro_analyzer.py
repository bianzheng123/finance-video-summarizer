"""e1_1 宏观分析：按字段组分析（经济版，整场字幕）+ 条目级修复。

经济版专用独立步骤（与完整版 ``step5_macro`` 完全独立）：无选段（经济版无
keyword_data / 话题段），直接对整场字幕按四个字段组（大盘情绪 / 宏观事件 /
宏观观点 / 仓位建议）分析，每个字段都是分块列表（`{标题/观点/引用}`）。

组 schema 由 `e1_1_宏观分析.json` 按字段裁剪派生（单份 schema 为唯一真源）。prompt
片段与字段注册表复用完整版 `src.llm_summarize.step5_macro.macro_prompt`，仅 `[DATA]`
语义由本包 `_macro_data_econ.txt` 提供，每次 e1_1 调用只组装当前字段组的片段。
"""

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ...log_config import step_logger
from ...utils.llm_user_id import economic_user_id
from ...llm_summarize.analysis_common import build_repair_items
from ...llm_summarize.step5_macro.macro_prompt import (
    MACRO_FIELD_GROUPS,
    MacroFieldGroup,
    _sanitize_filename,
    build_field_prompt,
    group_of_field,
)

logger = logging.getLogger(__name__)

_COMMON_DIR = Path(__file__).parent.parent.parent / "llm_summarize" / "analysis_common"


class MacroAnalyzer:
    """e1_1 宏观分析：单字段组分析（整场字幕 `[DATA]`）+ 条目级修复。"""

    SECTION_NAME = "宏观分析"
    STEP_NUMBER_STR = "e1_1"

    def __init__(
        self,
        llm_client: LLMClient,
        schema: dict,
        data_semantics: str,
    ) -> None:
        self._llm_client = llm_client
        self._schema = schema
        # 版本专属 [DATA] 语义片段（经济版：整场字幕，无选段）
        self._data_semantics = data_semantics
        # 修复模式只输出失败条目的局部 JSON，不再重新生成整个字段
        self._repair_schema = _load_schema("e1_1_宏观分析_repair", __file__)
        _jinja_env = Environment(
            loader=FileSystemLoader([str(Path(__file__).parent), str(_COMMON_DIR)])
        )
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("e1_1_宏观分析_user.jinja2")

    # ==================== schema 派生 ====================

    @staticmethod
    def group_schema(full_schema: dict, group: MacroFieldGroup) -> dict:
        """从整份 e1_1 schema 裁剪出单字段组的 schema（唯一真源，防漂移）。"""
        props = full_schema.get("properties") or {}
        required = set(full_schema.get("required") or [])
        return {
            "type": "object",
            "properties": {k: props[k] for k in group.字段 if k in props},
            "required": [k for k in group.字段 if k in required],
        }

    # ==================== e1_1 分析 ====================

    def analyze_group(
        self,
        group: MacroFieldGroup,
        subtitle_text: str,
        base_dir: Path,
        result_sink: dict[str, dict] | None = None,
    ) -> tuple[str, dict]:
        """分析单个字段组，返回 (组名, 该组字段的局部结果)。

        `[DATA]` 是整场字幕（经济版无选段）。阶段标签在本方法内自打——本方法会被
        丢进线程池，提交点的 with 块管不到 worker 线程。
        """
        safe_name = _sanitize_filename(group.名称)
        with step_logger(f"第{self.STEP_NUMBER_STR}步-{group.名称}"):
            user_prompt = self._user_template.render(
                task_rules=build_field_prompt(group.字段[0], self._data_semantics),
                subtitle_text=subtitle_text,
                repair_items=[],
            )
            log_dir = get_step_output_dir(base_dir, 1) if base_dir is not None else None
            log_name = f"{self.STEP_NUMBER_STR}_宏观分析_{safe_name}"
            result = self._llm_client.call(
                user_prompt=user_prompt,
                temperature=0.0,
                schema=self.group_schema(self._schema, group),
                log_dir=log_dir,
                log_name=log_name,
                step_name=f"{self.STEP_NUMBER_STR}_宏观分析_{group.名称}",
                user_id=economic_user_id(),
                warmup_shared=True,
            )
            if result_sink is not None:
                result_sink[log_name] = result
            logger.info(
                "宏观字段组「%s」生成完成：%s",
                group.名称,
                {k: (len(v) if isinstance(v, list) else "✓") for k, v in result.items()},
            )
            return group.名称, result

    # ==================== 条目级修复 ====================

    def repair_item(
        self,
        task: dict,
        current_item: dict,
        failure_reason: str,
        subtitle_text: str,
        base_dir: Path,
        valid_row_ids: set[int],
        result_sink: dict[str, dict] | None = None,
    ) -> dict | None:
        """只重写失败的那一条宏观条目，返回修复后的条目；不可用返回 None。

        写回前做确定性校验（引用非空且落在合法行号集合内），不通过返回 None
        （由调用方退回重写路径），杜绝"修复无效无人发现"。
        """
        sub_key = task.get("sub_key")
        group = group_of_field(sub_key) if sub_key else None
        if group is None:
            logger.warning("宏观修复：字段 %r 不属于任何字段组，放弃", sub_key)
            return None
        name_key = "标题"
        ident = str(current_item.get(name_key) or "").strip()
        if not ident:
            logger.warning("宏观修复：待修条目缺标识字段 %s，放弃", name_key)
            return None

        item_index = task.get("item_index")
        safe_ident = _sanitize_filename(ident)[:24]
        with step_logger(f"第{self.STEP_NUMBER_STR}步-{group.名称}修复"):
            user_prompt = self._user_template.render(
                task_rules=build_field_prompt(group.字段[0], self._data_semantics),
                subtitle_text=subtitle_text,
                repair_items=build_repair_items([{
                    "字段": sub_key,
                    "失败原因": failure_reason,
                    "原始内容": current_item,
                }]),
            )
            log_name = (
                f"{self.STEP_NUMBER_STR}_宏观分析_{_sanitize_filename(group.名称)}"
                f"_repair_{safe_ident}"
            )
            repaired = self._llm_client.call(
                user_prompt=user_prompt,
                temperature=0.0,
                schema=self._repair_schema,
                log_dir=get_step_output_dir(base_dir, 1) if base_dir is not None else None,
                log_name=log_name,
                step_name=f"{self.STEP_NUMBER_STR}_宏观分析_{group.名称}_修复",
            )
            if result_sink is not None:
                result_sink[log_name] = repaired
            if not isinstance(repaired, dict):
                return None
            payload = repaired.get(sub_key)
            if payload is None:
                logger.warning(
                    "宏观修复 %s：产物缺字段 %s，放弃（退回重写路径）", ident, sub_key,
                )
                return None
            # 列表字段：prompt 要求只输出被修的那一条 → 取首条；
            # 对象字段：产物本身就是该条目
            new_item = payload[0] if isinstance(payload, list) else payload
            if not isinstance(new_item, dict):
                logger.warning("宏观修复 %s：产物形态非对象，放弃", ident)
                return None
            if str(new_item.get(name_key) or "").strip() != ident:
                logger.warning(
                    "宏观修复 %s：产物标识为 %r，与待修条目不一致，放弃",
                    ident, new_item.get(name_key),
                )
                return None
            if item_index is None and not isinstance(payload, dict):
                logger.warning("宏观修复 %s：顶层子段应产出对象，放弃", ident)
                return None
            if not self._repair_deterministic_ok(new_item, valid_row_ids, ident):
                return None
            logger.info("宏观修复「%s」完成并重生成条目：%s", ident, sub_key)
            return new_item

    @staticmethod
    def _repair_deterministic_ok(
        item: dict, valid_row_ids: set[int], ident: str,
    ) -> bool:
        """修复产物的确定性校验：不通过就不写回（杜绝"修复无效无人发现"）。

        检查：
        - 引用必须是非空整数数组且全部落在合法行号集合内（越界引用会在行号→时间
          转换时被静默吸附到邻近真实行，反而更难发现）。
        """
        citations = item.get("引用")
        if not isinstance(citations, list) or not citations:
            logger.warning("宏观修复 %s：修复产物无引用，放弃", ident)
            return False
        bad = [
            r for r in citations
            if not isinstance(r, int) or isinstance(r, bool) or r not in valid_row_ids
        ]
        if bad:
            logger.warning(
                "宏观修复 %s：修复产物引用越界 %s，放弃", ident, bad[:10],
            )
            return False
        return True
