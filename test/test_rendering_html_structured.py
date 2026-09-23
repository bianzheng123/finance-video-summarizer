"""结构化折叠式 HTML 渲染器测试。

覆盖 src/rendering/rendering_html_structured.py 的两部分：
  1. transform_analysis —— 数据归一化（观点 tone 判定、未表态过滤、板块排序等）
  2. render_structured_html —— 自包含 HTML 输出（结构合法、标签配对、折叠交互存在）

用法：
    uv run pytest test/test_rendering_html_structured.py -v
    # 或直接运行（无 pytest 时打印各用例结果）：
    uv run python test/test_rendering_html_structured.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rendering.rendering_html_structured import (
    render_structured_html,
    transform_analysis,
    _parse_opinion,
    _norm_quotes,
    _norm_stance,
    SPECIAL_KEYS,
)


# ==================== 测试夹具：一份结构完整的 analysis ====================

def make_analysis() -> dict:
    """构造覆盖各分区的合成 analysis（含未表态、多空、热点预测、暗示、技巧、精炼总结、交易记录）。"""
    return {
        "精炼总结": {
            "大盘概览": {
                "今日市场": "震荡走高，量能温和放大",
                "宏观事件": "外围议息落地",
                "后市观点": "偏乐观",
                "仓位建议": "五成左右",
                "热点板块预测": "AI 与半导体",
            },
            "主讲人暗示重要话题": [
                {"板块": ["半导体"], "信号类型": "仓位", "原话总结": "节前轻仓", "原话": "大家轻仓过节"},
            ],
            "短期机会速览": [{"名称": "中芯国际", "描述": "产能满产"}],
            "个股观点速览": [{"板块": "半导体", "名称": "中芯国际", "观点": "短线看多"}],
            "个股事件速览": [{"板块": "半导体", "事件": "扩产\n需求外溢", "涉及资产": "中芯国际"}],
            "市场知识索引": [{"板块": "半导体", "知识简要": "晶圆代工", "详细内容": "代工制造"}],
            "交易技巧摘要": [{"名称": "止损纪律", "描述": "跌破 5% 离场"}],
        },
        "宏观分析": {
            "大盘情绪与状态判断": {
                "观点": "今日市场震荡走弱，成交量萎缩明显，观望情绪浓厚",
                "引用": [{"原文": "今天量能不足", "开始时间": "00:00:10", "结束时间": "00:00:15"}],
            },
            "宏观经济与事件影响分析": {"观点": "美联储议息临近，外围扰动加大"},
            "后市观点": {"观点": "短期以防守为主，等待方向选择"},
            "仓位建议": {"观点": "建议半仓应对"},
            "板块分析": {
                "资金流向": {"观点": "资金流出周期，回流科技"},
            },
            "热点板块预测": {
                "总览": "看好人工智能与半导体",
                "板块清单": [
                    {"板块名称": "AI", "预测倾向": "看多", "时间窗口": "下周",
                     "条件/前置事件": "政策落地", "引用": [{"原文": "AI是主线", "开始时间": "00:01:00", "结束时间": "00:01:05"}]},
                    {"板块名称": "半导体", "预测倾向": "看多", "时间窗口": "一个月"},
                ],
            },
        },
        "主讲人暗示重要话题": [
            {"信号类型": "仓位", "原话总结": "节前建议轻仓",
             "引用": [{"原文": "大家轻仓过节", "开始时间": "00:05:00", "结束时间": "00:05:04"}]},
        ],
        "半导体": {
            "引用总时长": 300,
            "引用时间": [{"开始时间": "00:02:00", "结束时间": "00:07:00"}],
            "投资机会": [
                {
                    "标题": "中芯国际",
                    "短线": {"观点": "看多", "逻辑": "订单饱满产能满载",
                             "引用": [{"原文": "满产", "开始时间": "00:02:10", "结束时间": "00:02:12"}]},
                    "中长线": {"观点": "看多", "逻辑": "国产替代长逻辑"},
                    "中线": {"观点": "看多", "逻辑": "产能爬坡期红利"},
                },
                {
                    # 全部未表态 —— 应被过滤掉，不进 opportunities
                    "标题": "某ST股",
                    "短线": {"观点": "未表态"},
                    "中长线": {"观点": ""},
                },
            ],
            "个人交易记录": [
                {"标的名称": "中芯国际", "操作描述": "加仓", "操作理由": "看产能",
                 "操作结果": "浮盈",
                 "引用": [{"原文": "我加了点", "开始时间": "00:04:00", "结束时间": "00:04:03"}]},
            ],
            "其他": [
                {"内容": "主持人提了下周日程",
                 "引用": [{"原文": "下周见", "开始时间": "00:06:00", "结束时间": "00:06:02"}]},
            ],
            "事件与资产": [
                {"标题": "扩产消息", "涉及资产": ["中芯国际", "华虹"],
                 "影响与结果": "利好产业链", "逻辑": "需求外溢", "依据类型": "新闻",
                 "引用": [{"原文": "宣布扩产", "开始时间": "00:03:00", "结束时间": "00:03:05"}]},
            ],
            "市场知识": [
                {"标题": "什么是晶圆代工", "内容": "代工厂负责制造环节", "引用": []},
            ],
        },
        "白酒": {
            "引用总时长": 60,
            "投资机会": [
                {"标题": "贵州茅台", "短线": {"观点": "看空", "逻辑": "需求疲软"},
                 "长线": {"观点": "看多", "逻辑": "品牌护城河"}},
            ],
        },
        "空板块": {  # 无任何有效内容 —— 应被 _build_sectors 跳过
            "引用总时长": 10,
            "投资机会": [{"标题": "x", "短线": {"观点": "未表态"}}],
        },
        "交易技巧": {
            "止损纪律": {"具体方法": "跌破成本 5% 无条件离场", "适用范围或前提": "波段交易",
                         "引用": [{"原文": "一定要止损", "开始时间": "00:08:00", "结束时间": "00:08:03"}]},
        },
    }


# ==================== transform_analysis 单元测试 ====================

def test_parse_opinion_tone() -> None:
    assert _parse_opinion("坚定看多") == {"text": "坚定看多", "tone": "bull"}
    assert _parse_opinion("短期看空") == {"text": "短期看空", "tone": "bear"}
    assert _parse_opinion("现在很便宜") == {"text": "现在很便宜", "tone": "cheap"}
    assert _parse_opinion("未表态") == {"text": "未表态", "tone": "neutral"}
    assert _parse_opinion("") is None
    assert _parse_opinion(None) is None


def test_norm_quotes_filters_empty() -> None:
    quotes = _norm_quotes([
        {"原文": "有内容", "开始时间": "00:01:00", "结束时间": "00:01:05"},
        {"原文": "  ", "开始时间": "x"},   # 空原文 -> 过滤
        {"开始时间": "y"},                  # 无原文 -> 过滤
        "非字典",                           # -> 过滤
    ])
    assert len(quotes) == 1
    assert quotes[0] == {"text": "有内容", "start": "00:01:00", "end": "00:01:05"}
    assert _norm_quotes(None) == []


def test_norm_stance_drops_neutral() -> None:
    assert _norm_stance("短线", {"观点": "未表态"}) is None
    assert _norm_stance("短线", {"观点": ""}) is None
    assert _norm_stance("短线", None) is None
    s = _norm_stance("短线", {"观点": "看多", "逻辑": "理由"})
    assert s["tone"] == "bull" and s["label"] == "短线" and s["reason"] == "理由"


def test_transform_overview() -> None:
    view = transform_analysis(make_analysis())
    ov = view["overview"]
    assert ov is not None
    # 六个 section 齐全（含资金流向），顺序固定
    titles = [s["title"] for s in ov["sections"]]
    assert titles == [
        "大盘情绪与状态判断", "宏观经济与事件影响分析", "后市观点",
        "仓位建议", "资金流向", "热点板块预测",
    ]
    # list section：单条（dict 形态归一为单条目），无 brief 字段
    today = ov["sections"][0]
    assert today["kind"] == "list"
    assert today["items"][0]["text"] == "今日市场震荡走弱，成交量萎缩明显，观望情绪浓厚"
    assert "brief" not in today["items"][0]
    houshi = next(s for s in ov["sections"] if s["title"] == "后市观点")
    assert houshi["items"][0]["text"] == "短期以防守为主，等待方向选择"
    # 热点板块预测
    hot = next(s for s in ov["sections"] if s["title"] == "热点板块预测")
    assert hot["kind"] == "hot"
    assert hot["summary"] == "看好人工智能与半导体"
    assert [h["name"] for h in hot["items"]] == ["AI", "半导体"]


def test_transform_sectors_sorting_and_filtering() -> None:
    view = transform_analysis(make_analysis())
    sectors = view["sectors"]
    names = [s["sector"] for s in sectors]
    # 特殊键不作板块；空板块被跳过
    for special in SPECIAL_KEYS:
        assert special not in names
    assert "空板块" not in names
    # 按引用总时长降序：半导体(300) > 白酒(60)
    assert names == ["半导体", "白酒"]
    # 半导体：全未表态的「某ST股」被过滤，仅剩「中芯国际」
    semi = sectors[0]
    assert [o["title"] for o in semi["opportunities"]] == ["中芯国际"]
    assert len(semi["opportunities"][0]["stances"]) == 3  # 短线+中长线+中线
    assert semi["duration"] == "5分"
    assert len(semi["events"]) == 1 and len(semi["knowledge"]) == 1
    assert len(semi["refTimes"]) == 1
    # 个人交易记录 / 其他：归一化为通用条目
    assert len(semi["trades"]) == 1
    assert semi["trades"][0]["title"] == "中芯国际"
    assert [f["label"] for f in semi["trades"][0]["fields"]] == ["操作描述", "操作理由", "操作结果"]
    assert len(semi["other"]) == 1
    assert semi["other"][0]["fields"][0] == {"label": "内容", "text": "主持人提了下周日程"}
    # 白酒：长线变体立场与短线并存
    baijiu = sectors[1]
    assert [st["label"] for st in baijiu["opportunities"][0]["stances"]] == ["短线", "长线"]


def test_transform_short_opps_only_bull_shortterm() -> None:
    view = transform_analysis(make_analysis())
    opps = view["shortOpps"]
    # 只有「中芯国际」短线看多入选；白酒短线看空不入选
    assert [o["asset"] for o in opps] == ["中芯国际"]
    assert opps[0]["sector"] == "半导体"


def test_transform_hints_and_tips() -> None:
    view = transform_analysis(make_analysis())
    assert len(view["hints"]) == 1
    assert view["hints"][0]["summary"] == "节前建议轻仓"
    assert len(view["tips"]) == 1
    assert view["tips"][0]["name"] == "止损纪律"
    assert view["tips"][0]["method"].startswith("跌破成本")
    # 精炼总结（TLDR）不再投影到前端视图（页面已删除该分区）
    assert "tldr" not in view


def test_transform_empty_input() -> None:
    view = transform_analysis({})
    assert view == {"overview": None, "hints": [], "shortOpps": [], "sectors": [], "tips": []}
    view2 = transform_analysis(None)
    assert view2["sectors"] == []


def test_tldr_section_not_rendered() -> None:
    # 即使 analysis 含精炼总结，页面也不再渲染该分区（避免与独立分区重复）
    html = render_structured_html(make_analysis(), author=None, title=None)
    assert '<div class="section-title">精炼总结</div>' not in html
    for token in ["短期机会速览", "个股观点速览", "个股事件速览", "市场知识索引", "交易技巧摘要"]:
        assert token not in html, f"不应出现精炼总结子块：{token}"


def test_special_keys_excludes_injected_tldr() -> None:
    # render_save 会注入「精炼总结」；它必须被当作特殊键，不渲染成板块
    assert "精炼总结" in SPECIAL_KEYS
    analysis = make_analysis()
    analysis["精炼总结"] = {"大盘概览": {"今日市场": "x"}}
    names = [s["sector"] for s in transform_analysis(analysis)["sectors"]]
    assert "精炼总结" not in names


# ==================== render_structured_html 集成测试 ====================

def test_render_html_structure() -> None:
    html = render_structured_html(make_analysis(), author="测试UP", title="盘后总结", video_url="https://example.com/v")
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    # div 标签配对
    assert html.count("<div") == html.count("</div>")
    # 自包含：内联样式 + 脚本，无外部资源引用
    assert "<style>" in html and "<script>" in html
    assert "src=" not in html and "href=\"http" in html  # 唯一的 href 是视频链接
    # 折叠交互存在
    assert "fold-head" in html and "classList.toggle('open')" in html
    # 关键内容出现
    for token in ["盘后总结", "测试UP", "大盘概览", "热点板块预测", "🔒 重要暗示",
                  "⚡ 短期机会", "中芯国际", "个人交易记录", "其他", "中线",
                  "交易技巧", "止损纪律"]:
        assert token in html, f"缺少内容：{token}"
    # 大盘概览：不截断 + 原文引用改为默认折叠（quote-wrap fold）
    assert "今日市场震荡走弱，成交量萎缩明显，观望情绪浓厚" in html
    assert 'class="brief"' not in html and 'class="full"' not in html
    assert 'quote-wrap fold' in html
    assert 'quote-flat' not in html


def test_render_html_escapes_html() -> None:
    analysis = {
        "宏观分析": {"大盘情绪与状态判断": {"观点": "风险<script>alert(1)</script> & 注意"}},
    }
    html = render_structured_html(analysis, author="a<b>", title="t&t")
    # 注入的尖括号被转义，不会产生额外的裸标签
    assert "<script>alert(1)" not in html
    assert "&lt;script&gt;" in html
    assert "t&amp;t" in html


def test_render_html_minimal_input() -> None:
    # 空 analysis 也能生成合法页面（仅头部 + 空板块标题）
    html = render_structured_html({}, author=None, title=None)
    assert html.startswith("<!DOCTYPE html>")
    assert html.count("<div") == html.count("</div>")
    assert "视频总结" in html
    assert "板块总结（0）" in html


def test_render_html_real_sample_if_available() -> None:
    """若仓库内存在真实 llm_analysis.json，则用它冒烟测试（不存在则跳过）。"""
    import glob
    import json

    root = Path(__file__).parent.parent
    files = glob.glob(str(root / "summary_data" / "**" / "llm_analysis.json"), recursive=True)
    if not files:
        print("  (跳过) 未找到真实 llm_analysis.json 样本")
        return
    sample = files[0]
    with open(sample, encoding="utf-8") as f:
        analysis = json.load(f)
    html = render_structured_html(analysis, author="real", title="real")
    assert html.startswith("<!DOCTYPE html>")
    assert html.count("<div") == html.count("</div>")


# ==================== 直接运行入口（无 pytest 时） ====================

def _run_all() -> bool:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            print(f"  ✗ {name}（异常）: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 用例通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
