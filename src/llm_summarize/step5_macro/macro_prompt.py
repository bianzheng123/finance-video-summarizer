"""宏观分析 prompt 片段与字段注册表（完整版 step5_macro 与经济版 step1_macro 共享）。

把宏观分析 prompt 从「一份 txt 塞满四个字段」拆成「公共头 + 版本专属 [DATA]/[DATABASE]
语义 + 公共任务说明 + 单字段片段 + 公共尾」：每次 5_2 / e1_1 生成或修复调用只组装当前
字段组的片段，其余字段的 JSON 结构与提取规则不再进入 prompt，消除四分之三冗余。

字段注册表（`MacroFieldGroup` / `MACRO_FIELD_GROUPS`）与字段名清理
（`_sanitize_filename`）也在此统一，完整版与经济版不再各维护一份。

版本专属的 `[DATA]` / `[DATABASE]` 语义由调用方（各自的 macro_analyzer）读各自包内的
片段文件后作为 `data_semantics` 传入 `build_field_prompt`，不在本模块硬编码。
"""

from dataclasses import dataclass
from pathlib import Path

_FRAGMENT_DIR = Path(__file__).parent

# 公共片段文件名（与代码共置）。字段片段按字段名映射到对应文件。
_HEADER_FILE = "_macro_header.txt"
_TASK_INTRO_FILE = "_macro_task_intro.txt"
_FOOTER_FILE = "_macro_footer.txt"
_FIELD_FILES: dict[str, str] = {
    "大盘情绪": "_macro_field_大盘情绪.txt",
    "宏观事件": "_macro_field_宏观事件.txt",
    "宏观观点": "_macro_field_宏观观点.txt",
    "仓位建议": "_macro_field_仓位建议.txt",
}


def _load(name: str) -> str:
    return (_FRAGMENT_DIR / name).read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class MacroFieldGroup:
    """宏观分析的一个字段组：一次 5_1 选段（仅完整版）+ 一次 5_2/e1_1 分析。"""

    名称: str
    字段: tuple[str, ...]
    # 5_1 选段时告诉模型「这一部分要找什么样的内容」（仅完整版使用，经济版留空）
    选段指引: str = ""


MACRO_FIELD_GROUPS: tuple[MacroFieldGroup, ...] = (
    MacroFieldGroup(
        名称="大盘情绪",
        字段=("大盘情绪",),
        选段指引=(
            "主讲人描述当日盘面**已经发生的现象**：整体涨跌、上涨/下跌家数、资金进出"
            "方向、量能变化、关键指数表现。只要段里在讲「今天市场是什么样子」就选，"
            "不要因为段里同时提到原因而排除（分析阶段会自行只取现象部分）。"
        ),
    ),
    MacroFieldGroup(
        名称="宏观事件",
        字段=("宏观事件",),
        选段指引=(
            "主讲人在讲**宏观事件**（政策、数据、海外、监管、影响整体市场的事件）"
            "**如何传导到市场**：有「事件 → 市场影响」因果链条的段都选。"
            "板块级/产业级事件（如某板块的供给需求、某厂商停产）不选——由板块分析/"
            "主题分析负责。纯盘面描述、纯后市预测的段不选。"
        ),
    ),
    MacroFieldGroup(
        名称="宏观观点",
        字段=("宏观观点",),
        选段指引=(
            "主讲人对**大盘/整体市场**未来走势给出**明确预判**的段（如「预计会反弹」"
            "「关键看能否突破」）。只讲已经发生了什么、或只讲理由而不落结论的段不选；"
            "对某个板块/标的的预判不选——由板块分析/主题分析负责。"
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
)

# 字段 → 所属字段组（修复时按 sub_key 反查）
_FIELD_TO_GROUP: dict[str, MacroFieldGroup] = {
    field: group for group in MACRO_FIELD_GROUPS for field in group.字段
}


def group_of_field(field: str) -> MacroFieldGroup | None:
    """按顶层字段名反查所属字段组；未知字段返回 None。"""
    return _FIELD_TO_GROUP.get(field)


def _sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "_").replace("\\", "_").replace(":", "_")
        .replace("*", "_").replace("?", "_").replace('"', "_")
        .replace("<", "_").replace(">", "_").replace("|", "_")
    )


# 公共片段在模块加载时一次性读入并缓存，build_field_prompt 仅做字符串拼接，无额外 IO。
_HEADER = _load(_HEADER_FILE)
_TASK_INTRO = _load(_TASK_INTRO_FILE)
_FOOTER = _load(_FOOTER_FILE)
_FIELD_FRAGMENTS: dict[str, str] = {f: _load(file) for f, file in _FIELD_FILES.items()}


def field_requirements(field: str) -> str:
    """返回单字段完整要求片段（JSON 结构 + 字段提取规则），供 5_1 选段参考。"""
    fragment = _FIELD_FRAGMENTS.get(field)
    if fragment is None:
        raise KeyError(f"宏观分析字段 {field!r} 没有对应的 prompt 片段")
    return fragment


def build_field_prompt(field: str, data_semantics: str) -> str:
    """组装单字段 task_rules：公共头 + 版本专属 [DATA]/[DATABASE] 语义 + 任务说明 + 字段片段 + 公共尾。

    Args:
        field: 字段名（`大盘情绪` / `宏观事件` / `宏观观点` / `仓位建议`）。
        data_semantics: 版本专属的 `[DATA]`（完整版另含 `[DATABASE]`）语义片段文本。
    """
    fragment = field_requirements(field)
    parts = (
        _HEADER,
        data_semantics.strip(),
        _TASK_INTRO,
        fragment,
        _FOOTER,
    )
    return "\n\n".join(part for part in parts if part)
