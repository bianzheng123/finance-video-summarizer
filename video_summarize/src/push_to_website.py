"""主动推送：本地总结完成后，把新总结 POST 到云端网站接收端点。

云端地址取仓库根 .env 的 SUMMARIZE_WEBSITE_URL，认证用 SUMMARIZE_SERVER_TOKEN
（与 summarize_website 的 SUMMARIZE_SERVER_TOKEN 一致）。
推送失败只告警不阻断管线，由云端定时拉差异兜底恢复。
"""

import logging
import os
from pathlib import Path

import requests

from .summary_catalog import payload_for_dir

logger = logging.getLogger(__name__)


def push_summary(save_path: Path, mode: str = "full") -> bool:
    """把单个总结目录推送到云端 /api/push/summary；返回是否成功。

    mode: full / economic，决定读 rendering 还是 rendering_economic 产物。
    """
    website_url = os.getenv("SUMMARIZE_WEBSITE_URL", "").strip().rstrip("/")
    if not website_url:
        logger.info("未配置 SUMMARIZE_WEBSITE_URL，跳过主动推送")
        return False

    payload = payload_for_dir(save_path, mode=mode)
    if payload is None:
        logger.warning("推送失败：未找到总结产物 %s（mode=%s）", save_path, mode)
        return False

    token = os.getenv("SUMMARIZE_SERVER_TOKEN", "")
    headers = {"X-Auth-Token": token} if token else {}
    try:
        resp = requests.post(f"{website_url}/api/push/summary", json=payload, headers=headers, timeout=60)
    except requests.RequestException as e:
        logger.warning("主动推送失败（网络异常，交由定时拉差异兜底）: %s", e)
        return False

    if resp.status_code >= 400:
        logger.warning("主动推送失败（HTTP %d，交由定时拉差异兜底）: %s", resp.status_code, resp.text[:200])
        return False

    logger.info("已推送总结到云端: %s (%s)", payload.get("title"), payload.get("id"))
    return True
