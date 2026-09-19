"""summary_data 目录扫描（共享模块）。

供 `summarize_server.py`（提供差异/详情）与 `src/push_to_website.py`（主动推送）
复用，保证目录解析口径一致。约定 cwd = video_summarize/。

索引标记是 rendering/summary.md（最终产物）；metadata.json 缺失时走目录名降级链。
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

BASE = Path("summary_data")

# 排除段：rel.parts 任一命中即跳过（测试残留 / 聚合产物）
_DENY_SEGMENTS = {"_react_test", "_replay", "benchmark", "test", "overview", "unknown"}

_LIVE_SPLIT = "直播录像-"
# 日期正则链（依次尝试，归一化为 YYYY-MM-DD）
_DATE_RES = [
    re.compile(r"(\d{4}-\d{1,2}-\d{1,2})"),
    re.compile(r"(\d{4}\.\d{1,2}\.\d{1,2})"),
    re.compile(r"(\d{4}\d{2}\d{2})"),
]

_cache: tuple[float, list[dict]] | None = None


def summary_id(rel_path: Path) -> str:
    """rel_path 的 sha1 前 16 位：URL 安全、跨机稳定、目录名含换行/特殊字符也无碍。"""
    return hashlib.sha1(rel_path.as_posix().encode()).hexdigest()[:16]


def invalidate_cache() -> None:
    global _cache
    _cache = None


def _read_metadata(folder: Path) -> dict:
    """读目录下 metadata.json；不存在/损坏返回空 dict（不阻断全量扫描）。"""
    p = folder / "metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("读取 metadata.json 失败: %s", p)
        return {}


def _extract_date(text: str) -> str | None:
    for pattern in _DATE_RES:
        m = pattern.search(text)
        if m:
            raw = m.group(1)
            if "-" in raw:
                y, mo, d = raw.split("-")
            elif "." in raw:
                y, mo, d = raw.split(".")
            else:
                y, mo, d = raw[:4], raw[4:6], raw[6:8]
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return None


def _parse(folder: Path, rel: Path, meta: dict) -> tuple[str, str, str | None, str, str]:
    """解析 (author, title, publish_date, kind, session)，metadata 缺失时逐级降级。

    kind: live（直播录像）/ replay（录播再上传）/ video（短视频）。
    """
    name = folder.name
    if _LIVE_SPLIT in name:
        blogger, rest = name.split(_LIVE_SPLIT, 1)
        d = _extract_date(rest)
        session = rest[len(d):].lstrip("-").strip() if d else ""
        return blogger.strip() or "未知", name, d, "live", session

    title = meta.get("title") or name
    author = str(meta.get("author") or (rel.parts[1] if len(rel.parts) > 1 else "") or "未知")
    d = _extract_date(str(meta.get("publish_date") or "")) or _extract_date(name)
    kind = "replay" if ("直播回放" in str(title) or "直播回放" in name) else "video"
    return author, title, d, kind, ""


def scan_summaries(base: Path = BASE, hide_test: bool = False) -> list[dict]:
    """扫描全部总结目录，返回 entry 列表（publish_date 倒序，无日期在最后）。"""
    global _cache
    now = time.time()
    ttl = float(os.getenv("SCAN_TTL_SECONDS", "60"))
    if _cache and now - _cache[0] < ttl:
        return _cache[1]

    entries = []
    for rendering_name, mode in (("rendering", "full"), ("rendering_economic", "economic")):
        for summary_md in base.glob(f"**/{rendering_name}/summary.md"):
            folder = summary_md.parent.parent
            rel = folder.relative_to(base)
            if any(seg in _DENY_SEGMENTS for seg in rel.parts):
                continue

            meta = _read_metadata(folder)
            author, title, publish_date, kind, session = _parse(folder, rel, meta)
            if hide_test and "测试" in author:
                continue

            entries.append({
                "id": summary_id(rel),
                "platform": rel.parts[0],
                "author": author,
                "title": title,
                "kind": kind,
                "session": session,
                "publish_date": publish_date,
                "duration": meta.get("duration"),
                "view_count": meta.get("view_count"),
                "url": meta.get("url"),
                "rel_path": rel.as_posix(),
                "updated_at": int(summary_md.stat().st_mtime),
                "mode": mode,
            })

    entries.sort(key=lambda e: (e["publish_date"] or "", e["platform"], e["author"]), reverse=True)
    _cache = (now, entries)
    logger.info("目录扫描完成：%d 条总结", len(entries))
    return entries


def _read_json_file(path: Path) -> dict | None:
    """读 JSON 文件；不存在/损坏返回 None。"""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("读取 JSON 失败: %s", path)
        return None


def get_summary(base: Path, sid: str) -> dict | None:
    """按 id 查 entry，附 summary_md 全文 + metadata.json + llm_analysis.json（云端入库用）。"""
    for entry in scan_summaries(base):
        if entry["id"] == sid:
            folder = base / entry["rel_path"]
            mode = entry.get("mode", "full")
            rendering_name = "rendering_economic" if mode == "economic" else "rendering"
            analysis_name = "llm_analysis_economic" if mode == "economic" else "llm_analysis"
            return {
                **entry,
                "summary_md": read_summary_md(folder, rendering_name=rendering_name),
                "metadata": _read_json_file(folder / "metadata.json"),
                "analysis": _read_json_file(folder / analysis_name / "llm_analysis.json"),
            }
    return None


def read_summary_md(folder: Path, rendering_name: str = "rendering") -> str:
    """读 rendering[_economic]/summary.md（utf-8-sig，自动剥 BOM）。"""
    return (folder / rendering_name / "summary.md").read_text(encoding="utf-8-sig")


def payload_for_dir(folder: Path, mode: str = "full") -> dict | None:
    """直接按目录构造推送 payload（与 get_summary 输出契约一致，不依赖 scan 缓存）。

    供主动推送使用：总结刚渲染完成、目录扫描缓存尚未刷新时也能准确取到产物。
    mode: full / economic，决定读 rendering 还是 rendering_economic。
    folder 可能是绝对路径（record_live）或相对路径（url_video / monitor），统一 resolve
    后再 relative_to，保证 rel_path / id 与 scan_summaries 口径一致。
    """
    folder = folder.resolve()
    rel = folder.relative_to(BASE.resolve())
    rendering_name = "rendering_economic" if mode == "economic" else "rendering"
    analysis_name = "llm_analysis_economic" if mode == "economic" else "llm_analysis"
    summary_file = folder / rendering_name / "summary.md"
    if not summary_file.exists():
        return None

    meta = _read_metadata(folder)
    author, title, publish_date, kind, session = _parse(folder, rel, meta)
    return {
        "id": summary_id(rel),
        "platform": rel.parts[0],
        "author": author,
        "title": title,
        "kind": kind,
        "session": session,
        "publish_date": publish_date,
        "duration": meta.get("duration"),
        "view_count": meta.get("view_count"),
        "url": meta.get("url"),
        "rel_path": rel.as_posix(),
        "updated_at": int(summary_file.stat().st_mtime),
        "mode": mode,
        "summary_md": read_summary_md(folder, rendering_name=rendering_name),
        "metadata": meta,
        "analysis": _read_json_file(folder / analysis_name / "llm_analysis.json"),
    }
