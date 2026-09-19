"""步骤 4：章节分析（宏观 / 交易技巧 / 主题分析）

用法：
    cd video_summarize
    uv run python test/step4_section_analysis.py --input subtitles.json

依赖：step 1、step 3 已跑过，对应缓存文件存在于 --output-dir。

输出：step_test_output/llm_analysis.json（与正式管线格式一致）
      step_test_output/step4_result_preview.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_subtitle_l, require_cache, save_preview, setup_logging

import logging
setup_logging()
logger = logging.getLogger("step4")


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤 4：章节分析")
    parser.add_argument("--input", required=True, type=Path, help="原始字幕 JSON 文件路径")
    parser.add_argument("--output-dir", type=Path, default=Path("step_test_output"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误：输入文件不存在：{args.input}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 读取前序缓存
    cache1 = require_cache(args.output_dir, 1, "剪枝终判与话题重置")
    processed = cache1.get("processed_subtitle_l") or load_subtitle_l(args.input)

    cache3 = require_cache(args.output_dir, 3, "主题引用解析")
    keyword_data = dict(cache3)

    logger.info("使用字幕 %d 条，主题 %d 个", len(processed), len(keyword_data))

    from src.llm_summarize.step5_macro import MacroAnalysisStep
    from src.llm_summarize.step6_trading import TradingSensitivityStep
    from src.llm_summarize.step7_topic import TopicAnalysisStep

    analysis = MacroAnalysisStep().analyze(keyword_data, args.output_dir, processed)
    analysis.update(
        TradingSensitivityStep().analyze(keyword_data, args.output_dir, processed)
    )
    analysis.update(
        TopicAnalysisStep().analyze(keyword_data, args.output_dir, processed)
    )

    topics = {
        k: v for k, v in analysis.items()
        if isinstance(v, dict) and k not in {"宏观分析", "交易技巧", "主讲人暗示重要话题"}
    }
    macro = analysis.get("宏观分析", {})
    trading = analysis.get("交易技巧", {})
    print("\n步骤 4（章节分析）完成")
    print(f"  宏观分析字段数：{len(macro) if isinstance(macro, dict) else 0}")
    print(f"  交易技巧分类数：{len(trading) if isinstance(trading, dict) else 0}")
    print(f"  主题分析数：{len(topics)}")
    if topics:
        print("  主题示例（最多 3 个）：")
        for name in list(topics.keys())[:3]:
            print(f"    - {name}")

    save_preview(args.output_dir, 4, {
        "macro_field_count": len(macro),
        "trading_skill_count": len(trading) if isinstance(trading, list) else 0,
        "topic_count": len(topics),
        "analysis": analysis,
    })


if __name__ == "__main__":
    main()
