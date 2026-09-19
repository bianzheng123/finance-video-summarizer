import argparse
import json
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "script"))

from summary_cost import load_tokenizer
from summary_cost_report import print_full_report
from src.client.video_info_adapter import (
    extract_bvid,
    video_info_adapter,
)
from src.utils.utils import sanitize_filename
from src.utils.video_metadata import save_metadata
from src.recognize_subtitle import SubtitleRecognizer
from src.llm_analysis import LLMAnalyzer
from src.llm_summarize.schemas import SubtitleRow
from src.rendering_controller import SummaryRenderer as RenderingController
from src.log_config import session_logger, setup_logging, step_logger
from src.push_to_website import push_summary

logger = logging.getLogger(__name__)

analyzer = LLMAnalyzer()
renderer = RenderingController()

_tokenizer = None


def _get_tokenizer() -> object | None:
    """懒加载 DeepSeek tokenizer（首次调用才加载，避免拖慢模块导入）。"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = load_tokenizer()
    return _tokenizer


def summarize_and_render(
    output_dir: Path,
    subtitle_l: list[SubtitleRow] | None = None,
    analysis: dict | None = None,
    bvid: str | None = None,
    author: str | None = None,
    title: str | None = None,
    *,
    economic: bool = False,
) -> dict:
    """统一入口：LLM 分析 → 渲染全部产物。

    两种调用方式：
    - 传 subtitle_l：先跑 LLM 分析再渲染（url_video / monitor 场景）
    - 传 analysis：跳过分析，直接渲染（record_live 场景，分析已并行完成）

    修改渲染逻辑时只需改这一个函数，三个工作流同时生效。
    经济版（economic=True）会把分析与渲染切换到 llm_analysis_economic / rendering_economic。
    返回 analysis dict，调用方可用于后续操作（如发邮件）。
    """
    if analysis is None:
        analysis = analyzer.summarize(output_dir, subtitle_l, economic=economic)
    with step_logger("渲染"):
        renderer.render_save(output_dir, analysis, bvid=bvid, author=author, title=title, economic=economic)
    # 总结产物已落盘：主动推送到云端网站（失败不阻断，try-catch 兜底）。
    # 订阅通知不在此处触发——仅在 record_live_workflow.py 直播总结后发。
    _push(output_dir, economic=economic)
    return analysis


def _push(output_dir: Path, *, economic: bool = False) -> None:
    """总结完成后的主动推送。失败只告警不阻断管线。"""
    mode = "economic" if economic else "full"
    try:
        push_summary(output_dir, mode=mode)
    except Exception as e:
        logger.warning("主动推送异常: %s", e)


def resolve_save_path(url: str) -> tuple[str, str, str, Path, dict] | None:
    """获取视频信息并计算保存路径，供 process_url 与 summarize_server 复用。

    返回 (platform, author, title, save_path, video_info)；获取不到视频信息（含不支持
    的平台）返回 None，不创建 unknown/ 垃圾目录。
    """
    # 使用 video_info_adapter 获取准确视频信息
    with step_logger("视频信息"):
        video_info = video_info_adapter.get_video_info(url)

    if not video_info:
        logger.error("无法获取视频信息: %s", url)
        return None

    platform = video_info.get("platform", "unknown")
    author = video_info.get("author", f"url_{platform}")
    title = sanitize_filename(video_info.get("title", url[:100])[:100])

    safe_author = sanitize_filename(author)
    safe_title = sanitize_filename(title)
    save_path = Path("summary_data") / platform / safe_author / safe_title
    return platform, author, title, save_path, video_info


def process_url(url: str, *, economic: bool = False) -> tuple[Path, dict] | None:
    """处理单个视频 URL：视频信息 → 字幕识别 → LLM 总结 → 渲染。

    返回 (save_path, analysis)；视频信息获取失败或字幕识别失败返回 None。
    economic=True 时走经济版（只做宏观分析，产物落 rendering_economic）。
    """
    logger.info("处理URL: %s", url)

    bvid = extract_bvid(url)

    info = resolve_save_path(url)
    if info is None:
        return None
    platform, author, title, save_path, video_info = info

    log_file = save_path / ("workflow_economic.log" if economic else "workflow.log")
    with session_logger(log_file, session_name=f"{platform}/{save_path.parent.name}-{save_path.name}"):
        # 落盘/刷新 metadata.json（含发布时间等详细发布元信息）。
        # 无论后续是否跳过下载都写：旧数据重跑也会补上/刷新。失败不阻断主流程。
        with step_logger("视频信息"):
            save_metadata(save_path, url=url, platform=platform, author=author, title=title, video_info=video_info)

        # 检查是否已经处理过（通过输出目录判断）
        if save_path.exists():
            subtitle_file = SubtitleRecognizer().find_existing_subtitle(save_path)
            if subtitle_file:
                logger.info("检测到已有字幕，跳过 API 请求，直接读取: %s", subtitle_file)
                subtitle_l = read_subtitle_file_to_dict(subtitle_file)
                author = None
                title = None
                subtitle_response_path = save_path / "recognize_subtitle" / "subtitle_response.json"
                if subtitle_response_path.exists():
                    with open(subtitle_response_path, encoding="utf-8") as f:
                        response_data = json.load(f)
                        detail = response_data.get("detail", {})
                        author = detail.get("author")
                        title = detail.get("title")
                analysis = summarize_and_render(
                    save_path, subtitle_l=subtitle_l, bvid=bvid, author=author,
                    title=title, economic=economic,
                )
                return save_path, analysis

        # 使用热词表增强识别（已上传的视频，热词可以生效）
        recognizer = SubtitleRecognizer(use_hotword=True)

        with step_logger("字幕识别"):
            result = recognizer.transcribe(url, save_path, platform, author, title)
        if result is None:
            return None

        segments, output_dir, subtitle_l = result

        analysis = summarize_and_render(
            output_dir, subtitle_l=subtitle_l, bvid=bvid, author=author, title=title,
            economic=economic,
        )
        return output_dir, analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="视频 URL 总结（完整版 / 经济版）")
    parser.add_argument(
        "--economic", action="store_true",
        help="经济版：只做宏观分析，产物落 rendering_economic / llm_analysis_economic",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config_local/video_url.json"
    if not config_path.exists():
        logger.error("未找到 %s，请创建该文件并填入 URL 列表", config_path)
        return

    with open(config_path, encoding="utf-8") as f:
        urls = json.load(f)

    if not isinstance(urls, list) or not urls:
        logger.error("video_url.json 应为非空 URL 数组")
        return

    logger.info("共 %d 个视频待处理（%s）", len(urls), "经济版" if args.economic else "完整版")
    with tqdm(urls, desc="处理视频", total=len(urls)) as pbar:
        for i, url in enumerate(urls, 1):
            pbar.set_description(f"处理视频 {i}/{len(urls)}")
            started = time.perf_counter()
            result = process_url(url, economic=args.economic)
            elapsed = time.perf_counter() - started
            pbar.update(1)
            if result is not None:
                logger.info("视频总结完成：耗时 %.1f 秒", elapsed)
                print_full_report(result[0], tokenizer=_get_tokenizer(), economic=args.economic)


def read_subtitle_file_to_dict(subtitle_path: Path) -> list[SubtitleRow]:
    """读取字幕文件并转换为 ``SubtitleRow`` 列表格式
    
    字幕文件格式: start_time\tend_time\tSpeakerX\ttext
    
    Args:
        subtitle_path: 字幕文件路径
        
    Returns:
        字幕行列表（``SubtitleRow`` 实例，含 rowID, start_time, end_time, speaker, text）
    """
    if not subtitle_path.exists():
        raise FileNotFoundError(f"字幕文件不存在: {subtitle_path}")
    
    lines = subtitle_path.read_text(encoding="utf-8").splitlines()
    subtitle_l: list[SubtitleRow] = []
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # 跳过空行
            continue
        
        parts = line.split("\t")
        if len(parts) >= 4:
            start_time = parts[0].strip()
            end_time = parts[1].strip()
            speaker_str = parts[2].strip()
            text = parts[3].strip()
            # speaker 保留完整字符串（如 "Speaker0"）
            speaker = speaker_str if speaker_str else None
        else:
            # 降级处理：仅包含文本
            start_time = ""
            end_time = ""
            speaker = None
            text = line
        
        subtitle_l.append(SubtitleRow(
            rowID=i,
            start_time=start_time,
            end_time=end_time,
            speaker=speaker,
            text=text,
        ))
    
    return subtitle_l


if __name__ == "__main__":
    setup_logging("url_video_summarize.log")
    main()

