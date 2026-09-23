"""可复用的总结入口：把「视频 URL → ASR 字幕 → 七步 LLM 分析 → 渲染产物」的完整链路
从入口脚本抽出，供命令行（``url_video_summarize.py``）与桌面 GUI 共用。

对外只暴露一个主入口 :func:`summarize_url`，返回 :class:`SummarizeResult`
（保存目录 + analysis + 四个产物路径）。其余函数（:func:`process_url` /
:func:`resolve_save_path` / :func:`summarize_and_render` /
:func:`read_subtitle_file_to_dict`）保留为模块内复用与兼容旧引用。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .client.video_info_adapter import extract_bvid, video_info_adapter
from .llm_analysis import LLMAnalyzer
from .llm_summarize.schemas import SubtitleRow
from .log_config import session_logger, step_logger
from .recognize_subtitle import SubtitleRecognizer
from .rendering_controller import SummaryRenderer as RenderingController
from .utils.utils import sanitize_filename
from .utils.video_metadata import save_metadata

logger = logging.getLogger(__name__)

analyzer = LLMAnalyzer()
renderer = RenderingController()


@dataclass
class SummarizeResult:
    """一次总结的完整产物路径。

    ``summary_pdf`` 依赖可选包 weasyprint，缺失时渲染会降级为告警、不生成 PDF，
    故此处按文件是否存在返回 ``None``。
    """

    save_path: Path
    analysis: dict
    summary_md: Path
    summary_html: Path
    summary_structured_html: Path
    summary_pdf: Path | None


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

    经济版（economic=True）会把分析与渲染切换到 llm_analysis_economic / rendering_economic。
    返回 analysis dict，调用方可用于后续操作（如发邮件）。
    """
    if analysis is None:
        analysis = analyzer.summarize(output_dir, subtitle_l, economic=economic)
    with step_logger("渲染"):
        renderer.render_save(
            output_dir, analysis, bvid=bvid, author=author, title=title, economic=economic,
        )
    return analysis


def resolve_save_path(
    url: str,
    *,
    output_root: Path = Path("summary_data"),
) -> tuple[str, str, str, Path, dict] | None:
    """获取视频信息并计算保存路径，供 :func:`summarize_url` 复用。

    返回 (platform, author, title, save_path, video_info)；获取不到视频信息（含不支持的
    平台）返回 None，不创建 unknown/ 垃圾目录。save_path 相对 ``output_root`` 绝对路径。
    """
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
    save_path = Path(output_root).resolve() / platform / safe_author / safe_title
    return platform, author, title, save_path, video_info


def _build_result(save_path: Path, analysis: dict, economic: bool) -> SummarizeResult:
    """根据 save_path 与 economic 组装产物路径。"""
    rendering_dir = save_path / ("rendering_economic" if economic else "rendering")
    pdf_path = rendering_dir / "summary.pdf"
    return SummarizeResult(
        save_path=save_path,
        analysis=analysis,
        summary_md=rendering_dir / "summary.md",
        summary_html=rendering_dir / "summary.html",
        summary_structured_html=rendering_dir / "summary_structured.html",
        summary_pdf=pdf_path if pdf_path.exists() else None,
    )


def summarize_url(
    url: str,
    *,
    economic: bool = False,
    output_root: Path = Path("summary_data"),
) -> SummarizeResult | None:
    """处理单个视频 URL：视频信息 → 字幕识别 → LLM 总结 → 渲染，返回全部产物路径。

    Args:
        url: 视频页面 URL（B 站 / YouTube / 抖音）。
        economic: True 走经济版（只做宏观分析，产物落 rendering_economic）。
        output_root: 产物根目录，默认 ``summary_data``（会解析为绝对路径）。

    Returns:
        视频信息获取失败、字幕识别失败或渲染前链路中断时返回 None。
    """
    logger.info("处理URL: %s", url)

    bvid = extract_bvid(url)

    info = resolve_save_path(url, output_root=output_root)
    if info is None:
        return None
    platform, author, title, save_path, video_info = info

    log_file = save_path / ("workflow_economic.log" if economic else "workflow.log")
    with session_logger(log_file, session_name=f"{platform}/{save_path.parent.name}-{save_path.name}"):
        # 落盘/刷新 metadata.json（含发布时间等详细发布元信息）。
        # 无论后续是否跳过下载都写：旧数据重跑也会补上/刷新。失败不阻断主流程。
        with step_logger("视频信息"):
            save_metadata(
                save_path, url=url, platform=platform, author=author, title=title,
                video_info=video_info,
            )

        # 检查是否已经处理过（通过输出目录判断）
        if save_path.exists():
            subtitle_file = SubtitleRecognizer.find_existing_subtitle(save_path)
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
                return _build_result(save_path, analysis, economic)

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
        return _build_result(output_dir, analysis, economic)


def process_url(
    url: str,
    *,
    economic: bool = False,
    output_root: Path = Path("summary_data"),
) -> SummarizeResult | None:
    """:func:`summarize_url` 的别名，保留旧调用名。"""
    return summarize_url(url, economic=economic, output_root=output_root)


def read_subtitle_file_to_dict(subtitle_path: Path) -> list[SubtitleRow]:
    """读取字幕文件并转换为 ``SubtitleRow`` 列表格式。

    字幕文件格式: start_time\\tend_time\\tSpeakerX\\ttext

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
