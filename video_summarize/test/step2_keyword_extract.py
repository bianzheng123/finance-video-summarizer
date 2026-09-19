"""步骤 2：注释聚合 + 事件主题注入 + 4_1 向量主题去重 + 4_2 关键词合并 + 4_3 精确去重

关键词提取与黑话消歧已在第 3 步（3_1 关键词与剪枝）完成，落在逐行注释的
`关键词` 字段里；本步只把逐行条目聚合成主题字典并做语义等价合并。

用法：
    cd video_summarize
    uv run python test/step2_keyword_extract.py --input subtitles.json

依赖：step1_clean_correct.py 已跑过，缓存 step_test_output/step1_result_preview.json
（提供带 annotation 的 processed_subtitle_l）与 3_2_topic_reset.json 存在。
若缓存不存在，脚本会报错并提示先跑步骤 1。

输出：step_test_output/4_3_keyword_extraction.json（供后续步骤复用）
      step_test_output/step2_result_preview.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_subtitle_l, require_cache, save_preview, setup_logging

import logging
setup_logging()
logger = logging.getLogger("step2")


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤 2：注释聚合 + 主题去重")
    parser.add_argument("--input", required=True, type=Path, help="原始字幕 JSON 文件路径")
    parser.add_argument("--output-dir", type=Path, default=Path("step_test_output"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误：输入文件不存在：{args.input}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 读取 step 1 清洗后的字幕与话题段
    cache1 = require_cache(args.output_dir, 1, "剪枝终判与话题重置")
    processed = cache1.get("processed_subtitle_l")
    if not processed:
        # 降级：用原始字幕。注意原始字幕**没有 annotation**，聚合会得到空主题，
        # 只用于不阻断脚本；真要看结果必须先跑通 step 1。
        logger.warning("step 1 缓存中无 processed_subtitle_l，使用原始字幕（无注释，主题会为空）")
        processed = load_subtitle_l(args.input)

    import json
    # 段列表的最终形态是 3_2 话题重置的产物；1_2 只是它的输入（回退用）
    segments = None
    for name in ("3_2_topic_reset.json", "1_2_segment_merging.json"):
        path = args.output_dir / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            segments = json.load(f).get("结果")
        if segments:
            logger.info("话题段取自 %s", name)
            break
    if not segments:
        print(
            f"错误：{args.output_dir} 下找不到 3_2_topic_reset.json / "
            "1_2_segment_merging.json，请先运行 step1_clean_correct.py",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("使用字幕 %d 条，话题段 %d 个", len(processed), len(segments))

    from src.llm_summarize.clustering import ClusteringStep
    extractor = ClusteringStep()
    keyword_data = extractor._extract_keywords(processed, segments, args.output_dir)

    topics = list(keyword_data.keys())
    print("\n步骤 2（注释聚合 + 主题去重）完成")
    print(f"  聚合主题数：{len(topics)} 个")
    if not topics:
        print("  ⚠ 主题为空：通常是 step 1 的注释里没有关键词（检查 3_1 是否成功）")
    preview_topics = topics[:10]
    print(f"  主题列表（最多 10 个）：{preview_topics}")
    if len(topics) > 10:
        print(f"  ... 共 {len(topics)} 个")

    save_preview(args.output_dir, 2, {
        "topic_count": len(topics),
        "topics": topics,
        "keyword_data": keyword_data,
    })


if __name__ == "__main__":
    main()
