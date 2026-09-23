"""引用校验状态台账落盘（步骤 5 与 7 各自一份；步骤 6 不做校验，无台账）。

校验已内联进各生成步骤，台账也随步骤走：`{步骤号}_validation_status.json`。
每个待校验条目一条记录（通过 / 丢弃 / 异常保留 / 未处理），丢弃带失败原因。
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def write_validation_status(path: Path, statuses: list[dict]) -> None:
    """写状态台账：`{"metadata": {时间, 统计}, "条目": [...]}`。"""
    counts: dict[str, int] = {}
    for s in statuses:
        key = str(s.get("状态") or "未知")
        counts[key] = counts.get(key, 0) + 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "process_timestamp": datetime.now().isoformat(),
                    "统计": counts,
                },
                "条目": statuses,
            },
            f, ensure_ascii=False, indent=2, default=str,
        )
    logger.info("状态台账已落盘 %s：%s", path.name, counts)
