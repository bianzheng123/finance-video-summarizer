"""JSON Schema 相关工具：动态加载、按 schema 精确转换数字、步骤输出目录。

各 LLM 阶段的 prompt / schema 成对与调用方源码共置（`*.txt` / `*.json`），
本模块负责按调用方 `__file__` 同目录定位并加载 schema，以及在校验前做安全的数字类型转换。
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 已加载 schema 的缓存，键为解析后的绝对路径，避免重复读盘。
_SCHEMA_CACHE: dict[Path, dict] = {}


def _load_schema(schema_name: str, caller_file: str) -> dict | None:
    """动态加载 JSON Schema 文件。

    Args:
        schema_name: Schema 文件名（不含扩展名），如 "2_1_清洗与纠错"
        caller_file: 调用方的 __file__，schema 必须与之同目录

    Returns:
        加载的 Schema 字典，如果文件不存在则返回 None
    """
    schema_file = (Path(caller_file).parent / f"{schema_name}.json").resolve()
    if schema_file in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[schema_file]

    if not schema_file.exists():
        logger.warning("Schema 文件不存在: %s", schema_file)
        return None

    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)
    _SCHEMA_CACHE[schema_file] = schema
    logger.debug("成功加载 Schema: %s", schema_file)
    return schema


def _coerce_numeric_by_schema(value: Any, schema: dict | None) -> Any:
    """按 schema 路径精确转换 LLM 输出的字符串数字。

    LLM 偶尔会把 schema 声明为 integer/number 的字段输出成字符串（如 `"rowID": "1741"`）；
    但同时会有一些 string 字段的值长得像数字（如 ETF 代码 `"510300"`、港股代码 `"00700"`）。
    本函数沿着 schema 递归走，**只对** schema 声明类型集合包含 integer/number 的标量做转换，
    其他位置的字符串原样保留——避免把"510300"误转成整数 510300 把后续 schema 校验弄崩。
    """
    if schema is None or not isinstance(schema, dict):
        return value

    if any(k in schema for k in ("oneOf", "anyOf", "allOf")):
        return value

    schema_type = schema.get("type")
    type_set: set[str] = set()
    if isinstance(schema_type, str):
        type_set = {schema_type}
    elif isinstance(schema_type, list):
        type_set = {t for t in schema_type if isinstance(t, str)}

    if isinstance(value, dict):
        if "object" in type_set or not type_set:
            properties = schema.get("properties") or {}
            additional = schema.get("additionalProperties")
            out = {}
            for k, v in value.items():
                if k in properties:
                    out[k] = _coerce_numeric_by_schema(v, properties[k])
                elif isinstance(additional, dict):
                    out[k] = _coerce_numeric_by_schema(v, additional)
                else:
                    out[k] = v
            return out
        return value

    if isinstance(value, list):
        if "array" in type_set or not type_set:
            items_schema = schema.get("items")
            if isinstance(items_schema, dict):
                return [_coerce_numeric_by_schema(item, items_schema) for item in value]
        return value

    if isinstance(value, str):
        if "integer" in type_set:
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        if "number" in type_set:
            try:
                if "." in value or "e" in value or "E" in value:
                    return float(value)
                return int(value)
            except (TypeError, ValueError):
                return value
        return value

    return value


def get_step_output_dir(base_dir: Path, step_number: int) -> Path:
    """获取步骤对应的输出目录（不存在则创建）。"""
    step_dir = base_dir / f"llm_output_{step_number}"
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir
