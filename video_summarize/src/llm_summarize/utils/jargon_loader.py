"""本会话黑话增量加载——TopicAnalyzer 与 PostProcessor 共用。

来源文件：`<base_dir>/../incremental_database/jargon_incremental.json`
由第 2 步词级三阶段收尾（`subtitle_cleaning/word_level_step.py::_save_incremental_database`）
写入，结构：
{
  "个股黑话": {"<黑话>": {"正式名": "<formal>", ...}},
  "板块黑话": {"<黑话>": {"正式板块名": "<formal>", ...}},
  "未解析": [{"黑话": "<黑话>", "类型": "个股黑话|板块黑话", "状态": "未解析"}]
}
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_incremental_jargon(
    base_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """返回 (个股黑话 map, 板块黑话 map)，格式：{黑话: [正式名]}。"""
    incremental_path = base_dir.parent / "incremental_database" / "jargon_incremental.json"
    if not incremental_path.exists():
        return {}, {}
    try:
        data = json.loads(incremental_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("加载 jargon_incremental.json 失败：%s", e)
        return {}, {}

    stock_map: dict[str, list[str]] = {}
    sector_map: dict[str, list[str]] = {}
    for jargon, info in (data.get("个股黑话") or {}).items():
        if not isinstance(info, dict):
            continue
        formal = info.get("正式名")
        if isinstance(formal, str) and formal.strip():
            stock_map[jargon] = [formal.strip()]
    for jargon, info in (data.get("板块黑话") or {}).items():
        if not isinstance(info, dict):
            continue
        formal = info.get("正式板块名")
        if isinstance(formal, str) and formal.strip():
            sector_map[jargon] = [formal.strip()]
    return stock_map, sector_map
