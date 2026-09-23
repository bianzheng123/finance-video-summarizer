"""视频 URL 总结入口（命令行薄封装）。

核心逻辑在 ``src/summarize_entry.py`` 的 :func:`summarize_url`；本文件只做参数解析、
批量调度与费用报告打印。

用法：
    uv run python url_video_summarize.py --url <视频URL>            # 单链接
    uv run python url_video_summarize.py --url <URL> --economic     # 单链接（经济版）
    uv run python url_video_summarize.py                            # 批量：config/video_url.json
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "script"))

from summary_cost import load_tokenizer
from summary_cost_report import print_full_report
from src.llm_infra.config import load_video_urls
from src.log_config import setup_logging
from src.summarize_entry import (  # noqa: F401  # read_subtitle_file_to_dict 供 test/replay_pipeline.py 复用
    read_subtitle_file_to_dict,
    summarize_url,
)
from src.summarize_entry import SummarizeResult

logger = logging.getLogger(__name__)

_tokenizer = None


def _reconfigure_console_encoding() -> None:
    """把 stdout/stderr 重配为 UTF-8 并容错，避免 Windows GBK 控制台崩溃。

    `script/summary_cost_report.py` 的费用报告含 `█/∞/→/≈/×` 等非 GBK 字符，
    在 Windows 默认 cp936 控制台下 `print()` 会抛 UnicodeEncodeError。这里把
    流原地切成 UTF-8 + `errors="replace"`（无法显示的字符降级为替换符而非崩溃）；
    非 tty / 已重定向等场景 reconfigure 失败则静默忽略。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _get_tokenizer() -> object | None:
    """懒加载 DeepSeek tokenizer（首次调用才加载，避免拖慢模块导入）。"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = load_tokenizer()
    return _tokenizer


def _print_result(result: SummarizeResult) -> None:
    """打印本次总结的产物路径。"""
    print("\n生成产物：")
    print(f"  HTML（完整版） : {result.summary_html}")
    print(f"  HTML（折叠式） : {result.summary_structured_html}")
    print(f"  PDF           : {result.summary_pdf or '（未生成，需 weasyprint）'}")
    print(f"  Markdown      : {result.summary_md}")
    print(f"  输出目录       : {result.save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="视频 URL 总结（完整版 / 经济版）")
    parser.add_argument("--url", help="单个视频 URL（缺省时读 config/video_url.json 批量处理）")
    parser.add_argument(
        "--economic", action="store_true",
        help="经济版：只做宏观分析，产物落 rendering_economic / llm_analysis_economic",
    )
    parser.add_argument(
        "--output-root", "-o", default="summary_data",
        help="产物输出根目录（默认 summary_data）",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)

    # 单链接模式
    if args.url:
        started = time.perf_counter()
        result = summarize_url(args.url, economic=args.economic, output_root=output_root)
        elapsed = time.perf_counter() - started
        if result is not None:
            logger.info("视频总结完成：耗时 %.1f 秒", elapsed)
            print_full_report(result.save_path, tokenizer=_get_tokenizer(), economic=args.economic)
            _print_result(result)
        else:
            logger.error("视频总结失败（无法获取视频信息或字幕识别失败）")
        return

    # 批量模式：读 config/video_url.json
    urls = load_video_urls()
    if not urls:
        logger.error("config/video_url.json 为空，请填入视频链接数组，或用 --url 指定单个链接")
        return

    logger.info("共 %d 个视频待处理（%s）", len(urls), "经济版" if args.economic else "完整版")
    with tqdm(urls, desc="处理视频", total=len(urls)) as pbar:
        for i, url in enumerate(urls, 1):
            pbar.set_description(f"处理视频 {i}/{len(urls)}")
            started = time.perf_counter()
            result = summarize_url(url, economic=args.economic, output_root=output_root)
            elapsed = time.perf_counter() - started
            pbar.update(1)
            if result is not None:
                logger.info("视频总结完成：耗时 %.1f 秒", elapsed)
                print_full_report(result.save_path, tokenizer=_get_tokenizer(), economic=args.economic)


if __name__ == "__main__":
    _reconfigure_console_encoding()
    setup_logging("url_video_summarize.log")
    main()
