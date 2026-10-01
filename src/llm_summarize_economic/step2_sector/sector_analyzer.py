"""e2_1 板块分析：按字段组分析（经济版，整场字幕）+ 条目级修复。

经济版专用独立步骤（与宏观分析经济版 step1 完全独立）：无选段（经济版无 keyword_data /
话题段），直接对整场字幕按三个字段组（板块情绪 / 板块事件 / 板块观点）分析：
- 板块情绪 / 板块事件：分块列表（`{标题/观点/引用}`）；
- 板块观点：富结构（`总览` + `板块清单[板块名称/预测倾向/时间窗口/条件·前置事件/引用]`）。

组 schema 由 `e2_1_板块分析.json` 按字段裁剪派生（单份 schema 为唯一真源）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ...utils.analysis_cleanup import _DROP_KEY
from ...log_config import step_logger
from ...utils.llm_user_id import economic_user_id
from ...llm_summarize.analysis_common import build_repair_items

logger = logging.getLogger(__name__)

_COMMON_DIR = Path(__file__).parent.parent.parent / "llm_summarize" / "analysis_common"


def _sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "_").replace("\\", "_").replace(":", "_")
        .replace("*", "_").replace("?", "_").replace('"', "_")
        .replace("<", "_").replace(">", "_").replace("|", "_")
    )


def _extract_rich_item(payload: object, name_key: str, ident: str) -> dict | None:
    """从「板块观点」富结构（总览 + 板块清单[]）修复产物提取被修条目。

    从板块清单里找标识匹配的条目；找不到说明 LLM 判定该板块应合并进别的板块
    或删除，返回丢弃 sentinel（由 clean_analysis 删除该条目）。
    """
    if not isinstance(payload, dict):
        return None
    board = payload.get("板块清单")
    if not isinstance(board, list):
        return None
    for item in board:
        if isinstance(item, dict) and str(item.get(name_key) or "").strip() == ident:
            return item
    return {_DROP_KEY: f"板块修复：{ident} 应合并进别的板块或删除"}


def _extract_flat_item(
    payload: object, name_key: str, ident: str, item_index: object, label: str,
) -> dict | None:
    """从扁平列表字段（板块情绪/板块事件）修复产物提取被修条目：取首条并校验标识。"""
    new_item = payload[0] if isinstance(payload, list) else payload
    if not isinstance(new_item, dict):
        logger.warning("%s：产物形态非对象，放弃", label)
        return None
    if str(new_item.get(name_key) or "").strip() != ident:
        logger.warning(
            "%s：产物标识为 %r，与待修条目不一致，放弃", label, new_item.get(name_key),
        )
        return None
    if item_index is None and not isinstance(payload, dict):
        logger.warning("%s：顶层子段应产出对象，放弃", label)
        return None
    return new_item


def _extract_repaired_item(
    payload: object, group: SectorFieldGroup, name_key: str, ident: str, item_index: object,
) -> dict | None:
    """从修复产物 payload 提取被修条目，按字段结构分派（富结构 vs 扁平列表）。"""
    if len(group.条目路径) > 1:
        return _extract_rich_item(payload, name_key, ident)
    return _extract_flat_item(payload, name_key, ident, item_index, f"板块修复 {ident}")


@dataclass(frozen=True)
class SectorFieldGroup:
    """板块分析的一个字段组：一次整场字幕分析。

    除顶层字段名外，还描述「条目」的结构与定位，供修复匹配、同标识去重等
    消费方统一取用（避免散落 if 分支重复表达）：
    - 条目路径：从 section 顶层到条目列表的路径；一层 = 扁平列表，两层 = 富结构。
    - 标识键：条目用于标识自身的字段名（板块情绪/事件用「标题」，板块观点用「板块名称」）。
    """

    名称: str
    字段: tuple[str, ...]
    条目路径: tuple[str, ...]
    标识键: str


SECTOR_FIELD_GROUPS: tuple[SectorFieldGroup, ...] = (
    SectorFieldGroup(名称="板块情绪", 字段=("板块情绪",), 条目路径=("板块情绪",), 标识键="标题"),
    SectorFieldGroup(名称="板块事件", 字段=("板块事件",), 条目路径=("板块事件",), 标识键="标题"),
    SectorFieldGroup(名称="板块观点", 字段=("板块观点",), 条目路径=("板块观点", "板块清单"), 标识键="板块名称"),
)

# 字段 → 所属字段组（修复时按 sub_key 反查）
_FIELD_TO_GROUP: dict[str, SectorFieldGroup] = {
    field: group for group in SECTOR_FIELD_GROUPS for field in group.字段
}


def group_of_field(field: str) -> SectorFieldGroup | None:
    """按顶层字段名反查所属字段组；未知字段返回 None。"""
    return _FIELD_TO_GROUP.get(field)


class SectorAnalyzer:
    """e2_1 板块分析：单字段组分析（整场字幕 `[DATA]`）+ 条目级修复。"""

    SECTION_NAME = "板块分析"
    STEP_NUMBER_STR = "e2_1"

    def __init__(
        self,
        llm_client: LLMClient,
        prompt: str,
        schema: dict,
    ) -> None:
        self._llm_client = llm_client
        self._prompt = prompt
        self._schema = schema
        # 修复模式只输出失败条目的局部 JSON，不再重新生成整个字段
        self._repair_schema = _load_schema("e2_1_板块分析_repair", __file__)
        _jinja_env = Environment(
            loader=FileSystemLoader([str(Path(__file__).parent), str(_COMMON_DIR)])
        )
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("e2_1_板块分析_user.jinja2")

    # ==================== schema 派生 ====================

    @staticmethod
    def group_schema(full_schema: dict, group: SectorFieldGroup) -> dict:
        """从整份 e2_1 schema 裁剪出单字段组的 schema（唯一真源，防漂移）。"""
        props = full_schema.get("properties") or {}
        required = set(full_schema.get("required") or [])
        return {
            "type": "object",
            "properties": {k: props[k] for k in group.字段 if k in props},
            "required": [k for k in group.字段 if k in required],
        }

    # ==================== e2_1 分析 ====================

    def analyze_group(
        self,
        group: SectorFieldGroup,
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
                task_rules=self._prompt,
                subtitle_text=subtitle_text,
                field_group=group,
                repair_items=[],
            )
            log_dir = get_step_output_dir(base_dir, 2) if base_dir is not None else None
            log_name = f"{self.STEP_NUMBER_STR}_板块分析_{safe_name}"
            result = self._llm_client.call(
                user_prompt=user_prompt,
                temperature=0.0,
                schema=self.group_schema(self._schema, group),
                log_dir=log_dir,
                log_name=log_name,
                step_name=f"{self.STEP_NUMBER_STR}_板块分析_{group.名称}",
                user_id=economic_user_id(),
                warmup_shared=True,
            )
            if result_sink is not None:
                result_sink[log_name] = result
            logger.info(
                "板块字段组「%s」生成完成：%s",
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
        """只重写失败的那一条板块条目，返回修复后的条目；不可用返回 None。

        写回前做确定性校验（引用非空且落在合法行号集合内），不通过返回 None
        （由调用方退回重写路径），杜绝"修复无效无人发现"。
        """
        sub_key = task.get("sub_key")
        group = group_of_field(sub_key) if sub_key else None
        if group is None:
            logger.warning("板块修复：字段 %r 不属于任何字段组，放弃", sub_key)
            return None
        name_key = group.标识键
        ident = str(current_item.get(name_key) or "").strip()
        if not ident:
            logger.warning("板块修复：待修条目缺标识字段 %s，放弃", name_key)
            return None

        item_index = task.get("item_index")
        safe_ident = _sanitize_filename(ident)[:24]
        with step_logger(f"第{self.STEP_NUMBER_STR}步-{group.名称}修复"):
            user_prompt = self._user_template.render(
                task_rules=self._prompt,
                subtitle_text=subtitle_text,
                field_group=group,
                repair_items=build_repair_items([{
                    "字段": sub_key,
                    "失败原因": failure_reason,
                    "原始内容": current_item,
                }]),
            )
            log_name = (
                f"{self.STEP_NUMBER_STR}_板块分析_{_sanitize_filename(group.名称)}"
                f"_repair_{safe_ident}"
            )
            repaired = self._llm_client.call(
                user_prompt=user_prompt,
                temperature=0.0,
                schema=self._repair_schema,
                log_dir=get_step_output_dir(base_dir, 2) if base_dir is not None else None,
                log_name=log_name,
                step_name=f"{self.STEP_NUMBER_STR}_板块分析_{group.名称}_修复",
            )
            if result_sink is not None:
                result_sink[log_name] = repaired
            if not isinstance(repaired, dict):
                return None
            payload = repaired.get(sub_key)
            if payload is None:
                logger.warning(
                    "板块修复 %s：产物缺字段 %s，放弃（退回重写路径）", ident, sub_key,
                )
                return None
            new_item = _extract_repaired_item(payload, group, name_key, ident, item_index)
            if new_item is None:
                return None
            if _DROP_KEY in new_item:
                logger.info("板块修复「%s」：合并/删除，标记丢弃", ident)
                return new_item
            if not self._repair_deterministic_ok(new_item, valid_row_ids, ident):
                return None
            logger.info("板块修复「%s」完成并重生成条目：%s", ident, sub_key)
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
            logger.warning("板块修复 %s：修复产物无引用，放弃", ident)
            return False
        bad = [
            r for r in citations
            if not isinstance(r, int) or isinstance(r, bool) or r not in valid_row_ids
        ]
        if bad:
            logger.warning(
                "板块修复 %s：修复产物引用越界 %s，放弃", ident, bad[:10],
            )
            return False
        return True
