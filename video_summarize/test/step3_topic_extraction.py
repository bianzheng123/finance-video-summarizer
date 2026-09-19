"""步骤 3：候选生成 + 剪枝 + 引用解析

用法：
    cd video_summarize
    uv run python test/step3_topic_extraction.py --input subtitles.json

依赖：step 1、step 2 已跑过，对应缓存文件存在于 --output-dir。

输出：step_test_output/4_topic_consolidated.json（供步骤 4 复用）
      step_test_output/step3_result_preview.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_subtitle_l, require_cache, save_preview, setup_logging

import logging
setup_logging()
logger = logging.getLogger("step3")


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤 3：候选生成 + 剪枝 + 引用解析")
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

    cache2 = require_cache(args.output_dir, 2, "关键词提取")
    keyword_data = dict(cache2)
    keyword_data.pop("rank", None)

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

    logger.info("使用字幕 %d 条，主题 %d 个，话题段 %d 个", len(processed), len(keyword_data), len(segments))

    from src.llm_summarize.clustering import ClusteringStep

    step = ClusteringStep()
    pruned_kw, candidates = step._generate_candidates(
        keyword_data, processed, args.output_dir,
    )
    resolved_kw = step._resolve_references(
        pruned_kw, candidates, segments, processed, args.output_dir,
    )

    topics_with_refs = [t for t, d in resolved_kw.items() if isinstance(d, dict) and d.get("引用")]
    print("\n步骤 3（候选生成 + 引用解析）完成")
    print(f"  话题段数：{len(segments)} 个")
    print(f"  剪枝后主题数：{len(pruned_kw)} 个")
    print(f"  有引用区间的主题：{len(topics_with_refs)} 个")
    if topics_with_refs:
        print("  引用示例（最多 3 个）：")
        for topic in topics_with_refs[:3]:
            refs = resolved_kw[topic].get("引用", [])
            print(f"    {topic}：{len(refs)} 段引用")

    save_preview(args.output_dir, 3, {
        "segment_count": len(segments),
        "topic_count_after_pruning": len(pruned_kw),
        "topics_with_refs": len(topics_with_refs),
        "keyword_data": resolved_kw,
    })


if __name__ == "__main__":
    main()
