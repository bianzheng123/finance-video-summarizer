"""标定桌面 GUI 的 ETA 时长系数：扫描历史 summary_data 日志，按版本拟合「视频时长 → 处理耗时」。

用法（在 finance-video-summarizer 目录下）::

    uv run python script/calibrate_eta.py [summary_data根目录]

默认扫描 ``../video_summarize/summary_data``（主仓库的历史运行产物），可传入其它
目录覆盖。脚本只读日志、不写任何文件，输出各版本的比例分布与建议系数，供人工把
结果回填到 ``src/desktop/progress.py`` 的 ``FULL_TIME_RATIO`` / ``ECONOMIC_TIME_RATIO``。

口径说明：
- 每个视频目录含 ``metadata.json``（有 ``duration`` 秒）与 ``workflow.log``（完整版）/
  ``workflow_economic.log``（经济版）；耗时取日志首行与末行时间戳之差。
- 比例 = 耗时 / 时长；完整版耗时受字幕行数、LLM 修复回路、API 负载影响，方差很大，
  单看比例不够，建议取中位数并略偏保守（预估宁可偏长，别让用户白等后才发现低估）。
- 样本里混有「跳过 ASR 的重跑」（日志含「检测到已有字幕，跳过 API 请求」），重跑省了
  ASR，但 LLM 分析仍是绝对大头，标定时不做硬性剔除，仅打印标记供人工参考。
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_SKIP_ASR_MARK = "检测到已有字幕，跳过 API 请求"


def _ts_of(line: str) -> datetime.datetime | None:
    m = _TS_RE.match(line)
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")


def _elapsed_seconds(log_path: Path) -> float | None:
    """日志首行→末行时间戳差（秒）；无法解析返回 None。"""
    lines = [ln for ln in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        return None
    t0, t1 = _ts_of(lines[0]), _ts_of(lines[-1])
    if t0 is None or t1 is None or t1 <= t0:
        return None
    return (t1 - t0).total_seconds()


def _linfit(pts: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """最小二乘直线 elapsed = a*duration + b，返回 (a, b, r2)；样本不足 2 个返回 None。"""
    n = len(pts)
    if n < 2:
        return None
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    ymean = sy / n
    ss_tot = sum((y - ymean) ** 2 for _, y in pts)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in pts)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return a, b, r2


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../video_summarize/summary_data")
    root = root.resolve()
    if not root.exists():
        print(f"目录不存在: {root}")
        raise SystemExit(1)

    samples: dict[str, list[tuple[float, float, bool]]] = {"full": [], "economic": []}
    for meta_path in sorted(root.rglob("metadata.json")):
        d = meta_path.parent
        try:
            duration = json.loads(meta_path.read_text(encoding="utf-8")).get("duration")
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(duration, (int, float)) or duration <= 0:
            continue
        for ver, name in (("full", "workflow.log"), ("economic", "workflow_economic.log")):
            log_path = d / name
            if not log_path.exists():
                continue
            elapsed = _elapsed_seconds(log_path)
            if elapsed is None:
                continue
            skip_asr = _SKIP_ASR_MARK in log_path.read_text(encoding="utf-8", errors="replace")
            samples[ver].append((float(duration), elapsed, skip_asr))

    for ver in ("full", "economic"):
        pts = samples[ver]
        print(f"\n=== {ver}: {len(pts)} 样本 ===")
        if not pts:
            continue
        for dur, elapsed, skip in sorted(pts):
            print(f"  时长 {dur/60:6.1f}min  耗时 {elapsed/60:6.1f}min  比例 {elapsed/dur:.3f}"
                  f"{'  (跳过ASR重跑)' if skip else ''}")
        # 剔除明显异常（耗时超过 1.5 倍时长，通常是中断续跑/跨天日志）
        clean = [(dur, el) for dur, el, _skip in pts if el < 1.5 * dur]
        if clean:
            ratios = sorted(el / dur for dur, el in clean)
            median = ratios[len(ratios) // 2]
            fit = _linfit(clean)
            print(f"  剔除异常后 {len(clean)} 例：比例中位数 {median:.3f}，"
                  f"均值 {sum(ratios)/len(ratios):.3f}")
            if fit is not None:
                a, b, r2 = fit
                print(f"  线性拟合 elapsed = {a:.3f}*duration + {b:.0f}s（r²={r2:.2f}，拟合度低说明"
                      f"耗时主要不取决于时长，比例仅作先验）")


if __name__ == "__main__":
    main()
