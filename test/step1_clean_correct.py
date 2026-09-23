"""步骤 1-3：话题分割 + 词级三阶段（2_1~2_3）+ 剪枝终判与话题重置（3_1 ∥ 3_2）

用法：
    uv run python test/step1_clean_correct.py --input subtitles.json

输入：字幕 JSON 列表（rowID 从 1 开始连续）
流程：确定性预清洗 → 话题分割 + 片段合并 → batch 并行的词级三阶段
      （2_1 粗筛 → [2_2 词表判定 → 2_3 联网确认]）→ 确证错字回写 + 指代物化
      → 第 3 步并行（3_1 关键词与剪枝终判 ∥ 3_2 话题重置），
      所有行保留并附逐行注释（类型 + 纠错 + 指代 + 关键词）
输出：step_test_output/2_1_coarse_screen.json / 2_2_vocab_decision.json /
      2_3_web_confirm.json / 2_corrected_subtitle.json（第 2 步整步断点）
      3_1_keywords_pruning.json / 3_2_topic_reset.json（第 3 步两个断点）
      step_test_output/1_2_segment_merging.json（1_2 话题段；3_2 修正后的最终段
      在 3_2_topic_reset.json，供步骤 4 复用）
      step_test_output/step1_result_preview.json（可读摘要，含
      processed_subtitle_l——后续步骤由 _common.require_cache 从这里取字幕）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_subtitle_l, save_preview, setup_logging

import logging
setup_logging()
logger = logging.getLogger("step1")


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤 1：话题分割 + 按话题并行清洗纠错")
    parser.add_argument("--input", required=True, type=Path, help="字幕 JSON 文件路径")
    parser.add_argument("--output-dir", type=Path, default=Path("step_test_output"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误：输入文件不存在：{args.input}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subtitle_l = load_subtitle_l(args.input)
    logger.info("加载字幕 %d 条", len(subtitle_l))

    from src.llm_summarize.step3_refine import RefineStep
    from src.llm_summarize.subtitle_cleaning import WordLevelStep
    from src.llm_summarize.topic_segmentation import TopicSegmentationStep
    from src.llm_summarize.utils.llm_text_utils import mark_unimportant_segments

    step = WordLevelStep()
    segments, raw_segments = TopicSegmentationStep().segment(subtitle_l, args.output_dir)
    # 组剪枝：与生产管线同序（1_1 判「不重要」的段整段标无关，用**合并前**段列表）
    subtitle_l = mark_unimportant_segments(subtitle_l, raw_segments)
    corrected, step2_conclusions = step.process(
        subtitle_l, segments, args.output_dir, args.output_dir, None,
    )
    # 第 3 步：3_1 剪枝终判 ∥ 3_2 话题重置（注释在此才组装完整、段列表在此才定稿）
    processed, segments = RefineStep().process(
        corrected, segments, step2_conclusions, args.output_dir,
    )

    # 修改记录从 2_2 判定检查点读（改字幕的结论才产生修改）
    import json
    cache_file = args.output_dir / "2_2_vocab_decision.json"
    real_mods = []
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            verdicts = json.load(f).get("结果") or []
        real_mods = [
            {"rowID": v.get("rowID"), "修正前词组": v.get("原先的片段"),
             "修正后词组": v.get("修正后片段")}
            for v in verdicts
            if v.get("改字幕") and v.get("修正后片段")
        ]

    from collections import Counter
    type_counter = Counter(
        sub.get("annotation", {}).get("类型", "重点") for sub in processed
    )
    reference_count = sum(
        len(sub.get("annotation", {}).get("指代", [])) for sub in processed
    )
    print("\n步骤 1-3（话题分割 + 清洗纠错 + 剪枝终判与话题重置）完成")
    print(f"  话题段数：{len(segments)} 个（3_2 重置后；困惑段 "
          f"{sum(1 for s in segments if s.get('困惑'))} 个）")
    print(f"  输入字幕：{len(subtitle_l)} 条（全部保留，逐行注释）")
    print(
        f"  类型分布：重点 {type_counter['重点']}，无关 {type_counter['无关']}，"
        f"噪音 {type_counter['噪音']}；纠错修改：{len(real_mods)} 处；指代解析：{reference_count} 条"
    )
    if real_mods:
        print("  修改示例（最多 5 条）：")
        for m in real_mods[:5]:
            print(f'    行 {m.get("rowID", "?")}: "{m.get("修正前词组")}" → "{m.get("修正后词组")}"')

    save_preview(args.output_dir, 1, {
        "segment_count": len(segments),
        "segments": segments,
        "input_count": len(subtitle_l),
        "output_count": len(processed),
        "type_distribution": dict(type_counter),
        "reference_count": reference_count,
        "modification_count": len(real_mods),
        "modifications": real_mods,
        "processed_subtitle_l": processed,
    })


if __name__ == "__main__":
    main()
