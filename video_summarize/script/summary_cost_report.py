"""一次视频总结 LLM 用量与费用的完整 profiling 报告（可导入）。

展示逻辑从 script/llm_cost_stats.py 抽出，供该 CLI 与 url_video_summarize.py 等
入口共用；计算仍复用 summary_cost.py。默认输出与 CLI（--by all --period both）一致。

render_report 返回完整报告文本（无 LLM 日志返回 None），print_full_report 直接打印。
"""

import contextlib
import io
from pathlib import Path

from summary_cost import (
    PRICES,
    Attempt,
    aggregate,
    cost_of,
    iter_attempts,
    load_tokenizer,
)

# 每个步骤的中文名（llm_output_N 的 N）。步骤 3 不落 LLM 日志：行标注与联网消歧
# 已并入 2_4_关键词与剪枝，所以正常不会出现 llm_output_3。
STEP_NAMES: dict[str, str] = {
    "llm_output_1": "① 话题分割",
    "llm_output_2": "② 清洗纠错",
    "llm_output_3": "③ 行标注",
    "llm_output_4": "④ 主题聚类",
    "llm_output_5": "⑤ 分析生成",
    "llm_output_6": "⑥ 校验合并",
}

STAGE_NAMES: dict[str, str] = {
    "1_1": "话题分割",
    "1_2": "片段合并",
    "2_1": "粗筛",
    "2_2": "词表判定",
    "2_3": "联网确认",
    "2_4": "关键词与剪枝",
    "3_1": "行标注",
    "3_2": "联网消歧",
    "4_1": "向量主题去重",
    "4_2": "关键词合并",
    "4_3": "精确去重",
    "4_4": "候选粗筛",
    "4_5": "候选精筛",
    "4_6": "关系去环",
    "4_7": "祖先父生成",
    "4_8": "候选剪枝",
    "4_9": "两层压缩",
    "4_10": "引用扩展",
    "5_1": "宏观分析选段",
    "5_2": "宏观分析",
    "5_3": "引用校验(宏观)",
    "6_1": "交易技巧",
    "6_2": "敏感信号判定",
    "6_3": "引用校验(敏感信号)",
    "7_1": "主题分析",
    "7_2": "引用校验(主题)",
}


def _bar(ratio: float, width: int = 12) -> str:
    """费用占比的横向条（纯 ASCII 块字符，终端等宽对齐）。"""
    filled = min(width, int(round(ratio * width)))
    return "█" * filled + "·" * (width - filled)


def _width(text: str) -> int:
    """终端显示宽度（CJK 与全角符号占 2 列），用于手工对齐等宽表格。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    """按显示宽度左对齐补空格；超宽则按显示宽度截断（不切半个字）。"""
    if _width(text) > width:
        out = ""
        for ch in text:
            if _width(out) + _width(ch) > width:
                break
            out += ch
        text = out
    return text + " " * (width - _width(text))


def _print_table(
    stats: dict[str, dict[str, int]],
    period: str,
    label: str,
    names: dict[str, str],
    limit: int | None = None,
) -> None:
    """按费用降序打印分组明细表。

    「思考/正文」= 思考 token / 正文 token，是冗余度的直接读数：同类阶段里这个比值
    明显偏高，说明推理花的钱没有变成产出，通常就是降 reasoning_effort 的候选。
    """
    total_cost = sum(cost_of(s, period) for s in stats.values()) or 1e-12
    rows = sorted(stats.items(), key=lambda kv: cost_of(kv[1], period), reverse=True)
    shown = rows[:limit] if limit else rows

    titles = {k: f"{k} {names.get(k, '')}".strip() for k, _ in rows}
    name_w = max([_width(t) for t in titles.values()] + [_width(label), 12])

    header = (
        _pad(label, name_w)
        + f"{'调用':>6}{'轮':>5}{'重试':>6}{'命中':>13}{'未命中':>13}"
        + f"{'思考':>13}{'正文':>11}{'思考/正文':>12}{'费用元':>10}{'占比':>8}  分布"
    )
    print(f"\n{header}")
    # 数值列宽合计 95（6+5+6+13+13+13+11+10 数量列 + 10 费用 + 8 占比）+ 2 空格分隔符
    print("-" * (name_w + 97))
    for key, s in shown:
        cost = cost_of(s, period)
        ratio = cost / total_cost
        density = s["reasoning"] / s["content"] if s["content"] else None
        density_txt = "∞" if density is None else f"{density:.1f}×"
        print(
            _pad(titles[key], name_w)
            + f"{s['calls']:>6}{s['rounds']:>5}{s['retries']:>6}"
            + f"{s['hit']:>13,}{s['miss']:>13,}{s['reasoning']:>13,}{s['content']:>11,}"
            + f"{density_txt:>10}{cost:>10.2f}{ratio * 100:>7.1f}%  {_bar(ratio)}"
        )
    if limit and len(rows) > limit:
        rest = rows[limit:]
        rest_cost = sum(cost_of(s, period) for _, s in rest)
        print(
            _pad(f"（其余 {len(rest)} 项）", name_w)
            + f"{'':>77}{rest_cost:>10.2f}{rest_cost / total_cost * 100:>7.1f}%"
        )


def _print_totals(attempts: list[Attempt], periods: list[str]) -> None:
    """打印全局用量与峰谷费用。"""
    hit = sum(a.hit for a in attempts)
    miss = sum(a.miss for a in attempts)
    reasoning = sum(a.reasoning for a in attempts)
    content = sum(a.content for a in attempts)
    output = reasoning + content
    calls = len({(a.step, a.call) for a in attempts})
    rounds = len({(a.step, a.call, a.round_idx) for a in attempts})
    agg = {"hit": hit, "miss": miss, "reasoning": reasoning, "content": content}

    print(f"\n{'=' * 60}\n总量")
    print(f"  逻辑调用 {calls}  轮次 {rounds}  尝试 {len(attempts)}  重试 {len(attempts) - rounds}")
    print(f"  输入：命中 {hit:,} + 未命中 {miss:,} = {hit + miss:,}"
          f"（命中率 {hit / (hit + miss) * 100:.1f}%）" if hit + miss else "  输入：0")
    print(f"  输出：思考 {reasoning:,} + 正文 {content:,} = {output:,}"
          f"（思考占 {reasoning / output * 100:.1f}%）" if output else "  输出：0")

    for period in periods:
        price = {k: v[period] for k, v in PRICES.items()}
        label = "高峰" if period == "peak" else "空闲"
        print(
            f"\n== {label}时段（命中 {price['hit']:.2f} / 未命中 {price['miss']:.2f}"
            f" / 输出 {price['output']:.2f} 元每百万 token） =="
        )
        c_hit = hit / 1_000_000 * price["hit"]
        c_miss = miss / 1_000_000 * price["miss"]
        c_out = output / 1_000_000 * price["output"]
        print(f"  命中 {c_hit:.2f} + 未命中 {c_miss:.2f} + 输出 {c_out:.2f}"
              f"（其中思考 {reasoning / 1_000_000 * price['output']:.2f}）")
        print(f"  ** 合计 ≈ {cost_of(agg, period):.2f} 元 **")


def _print_diagnostics(attempts: list[Attempt], period: str) -> None:
    """打印省 token 相关的诊断信号：重试、修复回路、冷缓存、生效档位。"""
    print(f"\n{'=' * 60}\n诊断")

    by_stage = aggregate(attempts, "stage")

    # 1) 重试：attempts > rounds 的阶段，通常是 schema 校验反复失败
    retried = {k: s for k, s in by_stage.items() if s["retries"]}
    if retried:
        detail = "，".join(
            f"{k}({STAGE_NAMES.get(k, '')}) {s['retries']} 次" for k, s in sorted(retried.items())
        )
        print(f"  重试：{detail}")
    else:
        print("  重试：无（所有请求一次通过）")

    # 2) 修复回路：5_1 的 repair 调用，是历史上最大的思考消耗单点
    repairs = [a for a in attempts if a.is_repair]
    if repairs:
        r_think = sum(a.reasoning for a in repairs)
        total_think = sum(a.reasoning for a in attempts) or 1
        groups: dict[str, int] = {}
        for a in repairs:
            # 历史 log_name 形如 5_1_宏观分析_repair_{section}_{hex}（新实现为
            # 5_2_宏观分析_{组}_repair_{条目标识}，无 hex）；统一剥尾部随机 hex 归组
            section = a.call.split("_repair_", 1)[1].rsplit("_", 1)[0]
            groups[section] = groups.get(section, 0) + 1
        detail = "，".join(f"{k} {v} 次" for k, v in sorted(groups.items(), key=lambda kv: -kv[1]))
        print(f"  修复回路：{len({(a.step, a.call) for a in repairs})} 次调用，"
              f"思考 {r_think:,}（占全场 {r_think / total_think * 100:.1f}%）→ {detail}")
    else:
        print("  修复回路：无")

    # 3) 冷缓存：整次请求 0 命中且输入不小，多为并行争抢同一 user_id 缓存槽位
    cold = [a for a in attempts if a.hit == 0 and a.miss > 10_000]
    if cold:
        wasted = sum(a.miss for a in cold)
        by_call: dict[str, int] = {}
        for a in cold:
            by_call[a.stage] = by_call.get(a.stage, 0) + 1
        detail = "，".join(f"{k} {v} 次" for k, v in sorted(by_call.items(), key=lambda kv: -kv[1]))
        print(f"  冷缓存：{len(cold)} 次请求 0 命中，未命中输入 {wasted:,}"
              f"（若全部命中可省 {wasted / 1_000_000 * (PRICES['miss'][period] - PRICES['hit'][period]):.2f} 元）→ {detail}")
    else:
        print("  冷缓存：无")

    # 3.5) 预热：leader 串行冷启动那次请求（落盘带「预热」标记），未命中是大头
    warmups = [a for a in attempts if a.is_warmup]
    if warmups:
        w_miss = sum(a.miss for a in warmups)
        w_hit = sum(a.hit for a in warmups)
        w_out = sum(a.reasoning + a.content for a in warmups)
        total_miss = sum(a.miss for a in attempts)
        ratio = (
            f"（占全场未命中 {w_miss / total_miss * 100:.1f}%）"
            if total_miss else "（全场未命中 0）"
        )
        print(f"  预热：{len(warmups)} 次请求，未命中 {w_miss:,} {ratio}，"
              f"命中 {w_hit:,}，输出 {w_out:,}")
    else:
        print("  预热：无")

    # 4) 生效档位：核对配置是否真的按阶段落到了请求上
    efforts: dict[str, set[str]] = {}
    for a in attempts:
        efforts.setdefault(a.stage, set()).add(a.effort)
    print("  生效 reasoning_effort：" + "，".join(
        f"{k}={'/'.join(sorted(v))}" for k, v in sorted(efforts.items())
    ))

    models = sorted({a.model for a in attempts})
    print(f"  模型：{'、'.join(models)}")


def render_report(
    video_dir: Path,
    tokenizer: object | None = None,
    by: str = "all",
    period: str = "both",
    top: int = 20,
    *,
    economic: bool = False,
) -> str | None:
    """构建一次视频总结的完整 profiling 报告文本；无 LLM 日志返回 None。

    Args:
        video_dir: summary_data 下的视频目录（含 llm_analysis/ 或 llm_analysis_economic/）
        tokenizer: 可复用的 tokenizer（None 则自动加载，失败降级换算）
        by: 下钻维度 "stage" / "step" / "call" / "all"（默认）
        period: 计费时段 "peak" / "offpeak" / "both"（默认）
        top: by == "call" 时只列最贵的前 N 次调用（0=全列）
        economic: True 时读 llm_analysis_economic（经济版落盘目录）
    """
    llm_analysis = video_dir / ("llm_analysis_economic" if economic else "llm_analysis")
    if not llm_analysis.exists():
        return None
    if tokenizer is None:
        tokenizer = load_tokenizer()
    attempts = list(iter_attempts(llm_analysis, tokenizer))
    if not attempts:
        return None

    sort_period = "peak" if period == "peak" else "offpeak"
    periods: list[str] = ["peak", "offpeak"] if period == "both" else [period]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"目录：{video_dir.resolve()}")
        print(f"tokenizer：{'DeepSeek 官方 v4' if tokenizer else '官方换算 0.6中文/0.3英文（降级）'}"
              f"（仅用于正文；命中/未命中/思考取 API usage 计费口径）")

        if by in ("step", "all"):
            _print_table(aggregate(attempts, "step"), sort_period, "步骤", STEP_NAMES)
        if by in ("stage", "all"):
            _print_table(aggregate(attempts, "stage"), sort_period, "阶段", STAGE_NAMES)
        if by == "call":
            _print_table(
                aggregate(attempts, "call"), sort_period, "调用", {}, limit=top or None
            )

        _print_totals(attempts, periods)
        _print_diagnostics(attempts, sort_period)
        print("\n高峰时段：北京时间周一至周五 9:00-12:00、14:00-18:00；其余为空闲（单价减半）")
    return buf.getvalue()


def print_full_report(
    video_dir: Path,
    tokenizer: object | None = None,
    by: str = "all",
    period: str = "both",
    top: int = 20,
    *,
    economic: bool = False,
) -> bool:
    """打印一次视频总结的完整 profiling 报告；有内容时返回 True，无日志返回 False。"""
    report = render_report(video_dir, tokenizer=tokenizer, by=by, period=period, top=top, economic=economic)
    if report is None:
        return False
    print(report)
    return True
