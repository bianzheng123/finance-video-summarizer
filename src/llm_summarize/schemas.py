"""七步管线核心传递结构的 pydantic 模型定义。

把贯穿各步骤的模糊 ``list[dict]`` / ``dict`` 升级为字段明确的模型类：

- :class:`SubtitleRow` —— 字幕行（``subtitle_l``），贯穿全部 7 步；
- :class:`TopicSegment` —— 话题段（``segments``），步骤 1 产出、2/3/4 消费；
- :class:`TopicUnit` —— 工作单元（``units``），``build_topic_units`` 产出、步骤 2/3 内部消费；
- :class:`TopicData` —— 聚类结果（``keyword_data``）中单个主题的值，步骤 4 产出、5/6/7 消费。

设计要点：

- **字段名沿用现有中英文键名**（``rowID`` / ``text`` / ``开始行号`` / ``子项`` 等），
  ``model_dump()`` 输出的 JSON 结构与既有检查点完全一致，``model_validate()`` 可
  直接回读存量 JSON，断点续跑天然兼容。
- **``extra="allow"``**：项目字段集随管线演进（如话题段先后新增 ``事件`` / ``困惑`` /
  ``困惑原因``），允许临时携带额外字段而不抛校验错误；核心字段名与类型在此显式声明。
- **动态补齐字段用 Optional/默认值**：``text_llm``（第 2 步才物化）与 ``annotation``
  （第 2/3 步才写）默认 ``None``，容纳「字段在不同阶段逐步出现」的现状。
- ``annotation`` 内部（纠错/指代/关键词/不确定）是 LLM 中间产物、字段随 prompt 演进，
  **保持 ``dict`` 不深挖**——本次重构只约束四类核心结构的顶层字段与容器类型。
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SubtitleRow(BaseModel):
    """字幕行：贯穿七步管线传递的核心数据单元。

    ``text`` 为纯纠错文本，``text_llm`` 为「纠错 + 指代内联」的下游底稿（第 2 步收尾
    物化，此前为 ``None``）。``annotation`` 是逐行注释（类型/纠错/指代/关键词/不确定），
    第 2/3 步才写入，内部为 LLM 中间产物故保持 dict。
    """

    model_config = ConfigDict(extra="allow")

    rowID: int
    text: str = ""
    text_llm: str | None = None
    start_time: str = ""
    end_time: str = ""
    speaker: str | int | None = None
    annotation: dict | None = None


class TopicSegment(BaseModel):
    """话题段：步骤 1 话题分割产出，步骤 2/3/4 消费。

    字段集由 ``topic_segmentation/topic_segmenter.py::_normalize_segment`` 唯一构造入口
    决定；新增字段只需改该处与这里。
    """

    model_config = ConfigDict(extra="allow")

    开始行号: int
    结束行号: int
    话题标签: str = "未命名"
    事件: str = ""
    是否重要: bool = True
    困惑: bool = False
    困惑原因: str = ""


class TopicData(BaseModel):
    """聚类结果 ``keyword_data`` 中单个主题的值（``keyword_data`` 为 ``dict[str, TopicData]``）。

    ``引用`` 形态随阶段演进：步骤 4 内部先为行号数组（``list[int]``），经引用扩展与
    行号转时间后变为时间区间数组（``list[dict]``），故用 ``list[Any]`` 容纳两种形态。
    ``子项`` 引用其它主题名，形成主题树/图。
    ``辅助`` 标识「可折叠销毁」的缺席桶/标准桶/数据库 sector（非文本提取的组织性节点）。
    """

    model_config = ConfigDict(extra="allow")

    类型: str = ""
    引用: list[Any] = Field(default_factory=list)
    子项: list[str] = Field(default_factory=list)
    辅助: bool = False


class TopicUnit(BaseModel):
    """工作单元：``build_topic_units`` 产出，步骤 2/3 内部按话题并行的工作单元。

    ``rows`` 为 ``(row_id, SubtitleRow)`` 元组列表（text 非空、可分析的行），
    ``row_ids`` 为对应行号升序列表。纯内存对象，不落盘。
    """

    model_config = ConfigDict(extra="allow")

    idx: int
    label: str
    事件: str = ""
    rows: list[tuple[int, SubtitleRow]]
    row_ids: list[int]
