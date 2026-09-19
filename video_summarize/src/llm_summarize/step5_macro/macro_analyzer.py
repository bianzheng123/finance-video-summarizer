"""5_2 宏观分析：按字段组渐进式披露（选段 → 段全文分析）+ 条目级修复。

**为什么拆字段组**：旧实现一次调用把整场字幕喂进去生成六个字段，输出 token 与
失效重试成本都压在一次调用上，且任一字段引用不支撑就要重跑**整个**字段（重新
生成 20~36 条列表却只取走 1 条）。现在按五个字段组各自独立：
- 5_1 先只看「话题段索引」（行号范围 + 首行片段，纯代码拼、零 LLM）挑选相关段；
- 5_2 只把选中段的全文作为 `[DATA]` 分析该组字段。

`[DATA]` 由「整场完整字幕」改为「自选段全文」是 `[DATA]` 的新例外（见
LLM调用层.md）：宏观的作用域本就需要通篇，但通篇的代价由**选段**承担——索引
阶段极便宜，分析阶段只看选中的段，两端相加远低于整场。

字段组定义是 `MACRO_FIELD_GROUPS`，组 schema 由 `5_2_宏观分析.json` 按字段裁剪
派生（单份 schema 为唯一真源，杜绝五份 schema 漂移）。

**修复粒度**：校验不支撑时只重写**失败的那一条**（`repair_item`），log_name 用
可读的「字段组 + 条目标识」而非随机 uuid——旧实现的 `5_1_宏观分析_repair_{字段}_{uuid6}`
目录爆炸（实测 13 个）正源于此。
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ...llm_infra.client import LLMClient
from ...llm_infra.schema import _load_schema, get_step_output_dir
from ...log_config import step_logger
from ..analysis_common import build_repair_items, inline_single_field

logger = logging.getLogger(__name__)

_COMMON_DIR = Path(__file__).parent.parent / "analysis_common"


def _sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "_").replace("\\", "_").replace(":", "_")
        .replace("*", "_").replace("?", "_").replace('"', "_")
        .replace("<", "_").replace(">", "_").replace("|", "_")
    )


@dataclass(frozen=True)
class MacroFieldGroup:
    """宏观分析的一个字段组：一次 5_1 选段 + 一次 5_2 分析。"""

    名称: str
    字段: tuple[str, ...]
    # 5_1 选段时告诉模型「这一部分要找什么样的内容」
    选段指引: str


MACRO_FIELD_GROUPS: tuple[MacroFieldGroup, ...] = (
    MacroFieldGroup(
        名称="大盘情绪与状态判断",
        字段=("大盘情绪与状态判断",),
        选段指引=(
            "主讲人描述当日盘面**已经发生的现象**：整体涨跌、上涨/下跌家数、资金进出"
            "方向、量能变化、关键指数表现。只要段里在讲「今天市场是什么样子」就选，"
            "不要因为段里同时提到原因而排除（分析阶段会自行只取现象部分）。"
        ),
    ),
    MacroFieldGroup(
        名称="宏观经济与事件影响分析",
        字段=("宏观经济与事件影响分析",),
        选段指引=(
            "主讲人在讲外部事件（政策、数据、海外、产业链事件）**如何传导到市场**："
            "有「事件 → 影响」因果链条的段都选。纯盘面描述、纯后市预测的段不选。"
        ),
    ),
    MacroFieldGroup(
        名称="后市观点",
        字段=("后市观点",),
        选段指引=(
            "主讲人对未来走势给出**明确预判**的段（如「预计会反弹」「关键看能否突破」）"
            "。只讲已经发生了什么、或只讲理由而不落结论的段不选。"
        ),
    ),
    MacroFieldGroup(
        名称="仓位建议",
        字段=("仓位建议",),
        选段指引=(
            "主讲人讲**总仓位/整体敞口**该怎么摆的段：几成仓、加仓/减仓/调仓/观望、"
            "卸杠杆。判断口径是「这段有没有回答**手里总共该拿多少**」。"
            "只讲某个标的的买卖时机或持有周期、只讲通用交易纪律（如年线怎么用、"
            "持仓周期该多长）、只讲行情判断的段**不选**——那些由别的阶段负责。"
        ),
    ),
    MacroFieldGroup(
        名称="板块",
        字段=("热点板块预测", "板块分析"),
        选段指引=(
            "提到**具体板块/行业/概念**的段全选：既包括当日涨跌与资金流向，也包括"
            "对未来热点的前瞻（含条件性、不确定表述）。漏选会直接导致总结缺板块。"
        ),
    ),
)

# 字段 → 所属字段组（修复时按 sub_key 反查）
_FIELD_TO_GROUP: dict[str, MacroFieldGroup] = {
    field: group for group in MACRO_FIELD_GROUPS for field in group.字段
}

# 各字段组内条目的标识键（修复后按它在新产物里匹配目标条目）
_ITEM_NAME_KEY: dict[str, str] = {
    "板块分析": "板块名称",
    "热点板块预测": "板块名称",
}


def group_of_field(field: str) -> MacroFieldGroup | None:
    """按顶层字段名反查所属字段组；未知字段返回 None。"""
    return _FIELD_TO_GROUP.get(field)


class MacroAnalyzer:
    """5_2 宏观分析：单字段组分析（段全文 `[DATA]`）+ 条目级修复。"""

    SECTION_NAME = "宏观分析"
    STEP_NUMBER_STR = "5_2"

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
        self._repair_schema = _load_schema("5_2_宏观分析_repair", __file__)
        _jinja_env = Environment(
            loader=FileSystemLoader([str(Path(__file__).parent), str(_COMMON_DIR)])
        )
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("5_2_宏观分析_user.jinja2")

    # ==================== schema 派生 ====================

    @staticmethod
    def group_schema(full_schema: dict, group: MacroFieldGroup) -> dict:
        """从整份 5_2 schema 裁剪出单字段组的 schema（唯一真源，防漂移）。"""
        props = full_schema.get("properties") or {}
        required = set(full_schema.get("required") or [])
        return {
            "type": "object",
            "properties": {k: props[k] for k in group.字段 if k in props},
            "required": [k for k in group.字段 if k in required],
        }

    # ==================== 5_2 分析 ====================

    def analyze_group(
        self,
        group: MacroFieldGroup,
        subtitle_text: str,
        base_dir: Path,
        all_sectors_text: str = "",
        indices_text: str = "",
        styles_text: str = "",
        annotation_text: str = "",
        result_sink: dict[str, dict] | None = None,
    ) -> tuple[str, dict]:
        """分析单个字段组，返回 (组名, 该组字段的局部结果)。

        `[DATA]` 是 5_1 选中的话题段全文（不是整场字幕），见模块 docstring。
        阶段标签在本方法内自打——本方法会被丢进线程池，提交点的 with 块管不到
        worker 线程。
        """
        safe_name = _sanitize_filename(group.名称)
        with step_logger(f"第{self.STEP_NUMBER_STR}步-{group.名称}"):
            user_prompt = self._user_template.render(
                task_rules=self._prompt,
                subtitle_text=subtitle_text,
                annotation_text=annotation_text,
                all_sectors_text=all_sectors_text,
                indices_text=indices_text,
                styles_text=styles_text,
                field_group=group,
                repair_items=[],
            )
            log_dir = get_step_output_dir(base_dir, 5) if base_dir is not None else None
            log_name = f"{self.STEP_NUMBER_STR}_宏观分析_{safe_name}"
            result = self._llm_client.call(
                user_prompt=user_prompt,
                temperature=0.0,
                schema=self.group_schema(self._schema, group),
                log_dir=log_dir,
                log_name=log_name,
                step_name=f"{self.STEP_NUMBER_STR}_宏观分析_{group.名称}",
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
        all_sectors_text: str = "",
        indices_text: str = "",
        styles_text: str = "",
        annotation_text: str = "",
        result_sink: dict[str, dict] | None = None,
    ) -> dict | None:
        """只重写失败的那一条宏观条目，返回修复后的条目；不可用返回 None。

        与旧实现（整字段重跑再按名称取一条）的差别：
        - `原始内容` 只给失败条目本身，prompt 明确要求**只输出这一条**；
        - log_name 是「字段组 + 条目标识」的可读名，不再带随机 uuid；
        - 写回前做确定性校验（引用非空且落在合法行号集合内），不通过返回 None
          （由调用方退回重写路径），杜绝"修复无效无人发现"。
        """
        sub_key = task.get("sub_key")
        group = group_of_field(sub_key) if sub_key else None
        if group is None:
            logger.warning("宏观修复：字段 %r 不属于任何字段组，放弃", sub_key)
            return None
        name_key = _ITEM_NAME_KEY.get(sub_key, "标题")
        ident = str(current_item.get(name_key) or "").strip()
        if not ident:
            logger.warning("宏观修复：待修条目缺标识字段 %s，放弃", name_key)
            return None

        item_index = task.get("item_index")
        safe_ident = _sanitize_filename(ident)[:24]
        with step_logger(f"第{self.STEP_NUMBER_STR}步-{group.名称}修复"):
            user_prompt = self._user_template.render(
                task_rules=self._prompt,
                subtitle_text=subtitle_text,
                annotation_text=annotation_text,
                all_sectors_text=all_sectors_text,
                indices_text=indices_text,
                styles_text=styles_text,
                field_group=group,
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
                log_dir=get_step_output_dir(base_dir, 5) if base_dir is not None else None,
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
            # 对象字段（如 资金流向）：产物本身就是该条目
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

    # ==================== 黑话注入 ====================

    @staticmethod
    def inject_jargon(macro_data: dict, used_jargons: dict[str, list[str]]) -> dict:
        """把字幕里实际出现过的黑话挂到宏观分析的板块名字段：
        - 板块分析.板块涨跌[].板块名称
        - 热点板块预测.板块清单[].板块名称
        """
        if not isinstance(macro_data, dict) or not used_jargons:
            return macro_data
        ban = macro_data.get("板块分析")
        if isinstance(ban, dict):
            for entry in ban.get("板块涨跌", []) or []:
                inline_single_field(entry, "板块名称", used_jargons)
        hot = macro_data.get("热点板块预测")
        if isinstance(hot, dict):
            for entry in hot.get("板块清单", []) or []:
                inline_single_field(entry, "板块名称", used_jargons)
        return macro_data
