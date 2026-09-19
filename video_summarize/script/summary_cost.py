"""一次视频总结的真实 LLM token 用量与费用计算（可导入模块）。

只读 llm_output 落盘日志里的统计量，不做估算：输入命中/未命中/思考 token 取 DeepSeek
usage 的计费口径值；正文 token 用官方 v4 tokenizer 计数（无 transformers 时按官方换算降级）。

价格表来自 DeepSeek 官方「模型 & 价格」页（2026-08-17 起的峰谷定价）：
  高峰时段为北京时间周一至周五 9:00-12:00、14:00-18:00，其余为空闲（价格为高峰一半）。

供 script/llm_cost_stats.py（CLI 展示）与 summarize_server.py（押金结算）共用。
"""

import json
import re
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.llm_infra.config import stage_key_of

_TOKENIZER_URL = "https://cdn.deepseek.com/api-docs/deepseek_v4_tokenizer.zip"
_TOKENIZER_DIR = Path.home() / ".cache" / "llm_cost_stats" / "deepseek_v4_tokenizer"
_BEIJING = timezone(timedelta(hours=8))
_ROUND_RE = re.compile(r"round(\d+)__attempt(\d+)\.json$")

# deepseek-v4-flash 官方价格（元 / 百万 tokens）
PRICES: dict[str, dict[str, float]] = {
    "hit": {"peak": 0.10, "offpeak": 0.05},    # 输入（缓存命中）
    "miss": {"peak": 3.0, "offpeak": 1.5},     # 输入（缓存未命中）
    "output": {"peak": 9.0, "offpeak": 4.5},   # 输出（含思考与正文）
}


def load_tokenizer():
    """加载 DeepSeek 官方 v4 tokenizer；缺失则自动下载，仍失败返回 None（降级换算）。"""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return None
    tokenizer_json = next(_TOKENIZER_DIR.glob("**/tokenizer.json"), None) if _TOKENIZER_DIR.exists() else None
    if tokenizer_json is None:
        try:
            _TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
            tmp_zip = _TOKENIZER_DIR / "tokenizer.zip"
            urllib.request.urlretrieve(_TOKENIZER_URL, tmp_zip)  # noqa: S310
            with zipfile.ZipFile(tmp_zip) as zf:
                zf.extractall(_TOKENIZER_DIR)
            tmp_zip.unlink()
            tokenizer_json = next(_TOKENIZER_DIR.glob("**/tokenizer.json"), None)
        except OSError:
            return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(tokenizer_json.parent), trust_remote_code=True)
    except Exception:  # tokenizer 加载失败只降级，不影响统计主体
        return None


def count_tokens(text: str, tokenizer) -> int:
    """统计一段文本的 token 数；无 tokenizer 时用官方换算（中文≈0.6、英文≈0.3）。"""
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    zh = sum(1 for ch in text if "一" <= ch <= "鿿")
    return round(zh * 0.6 + (len(text) - zh) * 0.3)


def attempt_output_text(attempt: dict) -> str:
    """拼接一次尝试产生的正文与工具调用参数（即该次响应的 content 部分）。"""
    parts: list[str] = []
    content = attempt.get("内容")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, (dict, list)):
        parts.append(json.dumps(content, ensure_ascii=False))
    for tc in attempt.get("工具调用") or []:
        if isinstance(tc, dict):
            args = tc.get("参数") or tc.get("arguments") or tc.get("args") or {}
            parts.append(json.dumps(args, ensure_ascii=False))
    return "".join(parts)


@dataclass(frozen=True)
class Attempt:
    """一个 `round{N}__attempt{M}.json` 落盘文件对应的一次真实 API 请求。

    计费的最小单位是 attempt（每次请求都单独计价），逻辑调用 / 轮次只是它的分组维度：
      - 逻辑调用（call）= 一个 `llm_output_N/{log_name}/` 目录，业务侧的一次 call/call_conversation
      - 轮次（round）= 多轮对话里的一轮（工具调用循环每轮都是一次请求）
      - 尝试（attempt）= 同一轮内的重试（schema 校验失败 / 过载重试）
    因此「重试次数 = attempts - rounds」，把多轮对话算成重试是错的。
    """

    step: str          # llm_output_N 目录名
    call: str          # 调用目录名（= log_name）
    stage: str         # 阶段键 N_M（提取不到则退回 call 全名）
    round_idx: int
    attempt_idx: int
    hit: int           # 输入 token（缓存命中）
    miss: int          # 输入 token（缓存未命中）
    reasoning: int     # 思考 token
    content: int       # 正文 token（tokenizer 计数）
    model: str
    effort: str        # 生效的 reasoning_effort
    is_repair: bool    # log_name 含 _repair_（5_1 的修复回路）
    is_warmup: bool    # 前缀预热 leader（串行冷启动那次请求，落盘带「预热」标记）


def iter_attempts(llm_analysis_dir: Path, tokenizer) -> Iterator[Attempt]:
    """遍历 llm_analysis 下所有 attempt 落盘文件，产出计费记录。"""
    prefix_len = len(llm_analysis_dir.parts)
    for f in sorted(llm_analysis_dir.glob("llm_output_*/**/round*__attempt*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        usage = data.get("用量") or {}
        config = data.get("生效配置") or {}
        m = _ROUND_RE.search(f.name)
        call = f.parent.name
        yield Attempt(
            step=f.parts[prefix_len],
            call=call,
            stage=stage_key_of(data.get("步骤"), call) or call,
            round_idx=int(m.group(1)) if m else 1,
            attempt_idx=int(m.group(2)) if m else 1,
            hit=usage.get("prompt_cache_hit_tokens") or 0,
            miss=usage.get("prompt_cache_miss_tokens") or 0,
            reasoning=usage.get("reasoning_tokens") or 0,
            content=count_tokens(attempt_output_text(data), tokenizer),
            model=config.get("model") or "?",
            effort=config.get("reasoning_effort") or "?",
            is_repair="_repair_" in call,
            is_warmup=bool(data.get("预热")),
        )


def aggregate(attempts: list[Attempt], key: str) -> dict[str, dict[str, int]]:
    """按 Attempt 的某个字段聚合用量，并补出调用数 / 轮数 / 重试数。

    Args:
        attempts: iter_attempts 的结果
        key: Attempt 的字段名，取 "stage"（阶段）/ "step"（步骤）/ "call"（单次调用）

    Returns:
        {分组键: {attempts, rounds, calls, retries, hit, miss, reasoning, content, output}}
    """
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    calls: dict[str, set] = defaultdict(set)
    rounds: dict[str, set] = defaultdict(set)
    for a in attempts:
        k = getattr(a, key)
        s = stats[k]
        s["attempts"] += 1
        s["hit"] += a.hit
        s["miss"] += a.miss
        s["reasoning"] += a.reasoning
        s["content"] += a.content
        calls[k].add((a.step, a.call))
        rounds[k].add((a.step, a.call, a.round_idx))
    for k, s in stats.items():
        s["calls"] = len(calls[k])
        s["rounds"] = len(rounds[k])
        s["retries"] = s["attempts"] - s["rounds"]
        s["output"] = s["reasoning"] + s["content"]
    return dict(stats)


def cost_of(s: dict[str, int], period: str) -> float:
    """按峰谷单价算一个聚合分组的费用（元）。"""
    return (
        s["hit"] / 1_000_000 * PRICES["hit"][period]
        + s["miss"] / 1_000_000 * PRICES["miss"][period]
        + (s["reasoning"] + s["content"]) / 1_000_000 * PRICES["output"][period]
    )


def collect_usage(llm_analysis_dir: Path, tokenizer) -> dict[str, dict[str, int]]:
    """按阶段（N_M）聚合用量；阶段键提取不到的退回 log_name 全名。"""
    return aggregate(list(iter_attempts(llm_analysis_dir, tokenizer)), "stage")


def is_peak(ts: float | None = None) -> bool:
    """按北京时间判断是否高峰时段（周一至周五 9:00-12:00、14:00-18:00）。"""
    dt = datetime.fromtimestamp(ts, tz=_BEIJING) if ts else datetime.now(tz=_BEIJING)
    if dt.weekday() >= 5:
        return False
    return 9 <= dt.hour < 12 or 14 <= dt.hour < 18


def compute_cost(video_dir: Path, tokenizer=None, at_ts: float | None = None) -> dict | None:
    """计算某视频目录一次总结的 token 用量与费用。

    Args:
        video_dir: summary_data 下的视频目录（含 llm_analysis/）
        tokenizer: 可复用的 tokenizer（None 则自动加载，失败降级换算）
        at_ts: 计价时刻（决定峰谷单价），默认当前时间

    Returns:
        {hit, miss, reasoning, content, output, attempts, rounds, calls, retries,
         peak, cost_yuan, cost_fen, cost_yuan_peak, cost_yuan_offpeak}；无日志返回 None。
    """
    llm_analysis = video_dir / "llm_analysis"
    if not llm_analysis.exists():
        return None
    if tokenizer is None:
        tokenizer = load_tokenizer()

    attempts = list(iter_attempts(llm_analysis, tokenizer))
    if not attempts:
        return None

    agg: dict[str, int] = {
        "hit": sum(a.hit for a in attempts),
        "miss": sum(a.miss for a in attempts),
        "reasoning": sum(a.reasoning for a in attempts),
        "content": sum(a.content for a in attempts),
        "attempts": len(attempts),
    }
    agg["calls"] = len({(a.step, a.call) for a in attempts})
    agg["rounds"] = len({(a.step, a.call, a.round_idx) for a in attempts})
    agg["retries"] = agg["attempts"] - agg["rounds"]
    agg["output"] = agg["reasoning"] + agg["content"]

    peak = is_peak(at_ts)
    cost_yuan = cost_of(agg, "peak" if peak else "offpeak")
    return {
        **agg,
        "peak": peak,
        "cost_yuan": round(cost_yuan, 4),
        "cost_fen": int(round(cost_yuan * 100)),
        "cost_yuan_peak": round(cost_of(agg, "peak"), 4),
        "cost_yuan_offpeak": round(cost_of(agg, "offpeak"), 4),
    }
