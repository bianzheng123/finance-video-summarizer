"""字幕 I/O 工具 - 提供字幕文件的读写和格式转换功能"""

import logging
from pathlib import Path

from ..utils.time_utils import fmt_time
from ..log_config import step_logger

logger = logging.getLogger(__name__)


def time_to_seconds(t: str) -> float:
    """将 HH:MM:SS 格式的时间字符串转换为秒数（支持毫秒）。"""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_srt_time(s: float) -> str:
    """将秒数格式化为 SRT 字幕时间戳格式（HH:MM:SS,mmm）。"""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{int(sec):02d},{ms:03d}"


def parse_subtitles(subtitle_file: Path) -> list[tuple[str, str, str]]:
    """解析字幕文件，仅支持四列格式：开始时间\t结束时间\tSpeakerX\t文本。

    Args:
        subtitle_file: 字幕文件路径

    Returns:
        元组列表，每个元组为 (开始时间, 结束时间, 文本)
    """
    with step_logger("字幕识别"):
        entries = []
        with open(subtitle_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue

                parts = line.split("\t")

                if len(parts) == 4 and parts[2].startswith("Speaker"):
                    start_time = parts[0].strip()
                    end_time = parts[1].strip()
                    text = parts[3].strip()
                    entries.append((start_time, end_time, text))
                else:
                    logger.warning("忽略无效的字幕行: %s", line)

        return entries


def save_subtitle(segments: list[dict], output_path: Path) -> None:
    """保存字幕文件，格式：开始时间\t结束时间\tSpeakerX\t文本。

    Args:
        segments: 字幕列表，每个元素包含 startTime, end, text, speaker
        output_path: 输出文件路径
    """
    with step_logger("字幕识别"):
        with open(output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                start_time = fmt_time(seg["startTime"])
                end_time = fmt_time(seg["end"])
                text = seg["text"]

                if "speaker" in seg and seg["speaker"] is not None:
                    speaker_label = f"Speaker{seg['speaker']}"
                else:
                    speaker_label = "Speaker0"

                f.write(f"{start_time}\t{end_time}\t{speaker_label}\t{text}\n")

        logger.info("字幕已保存: %s（共 %d 段，带说话人标记）", output_path, len(segments))


def txt_to_srt(subtitle_path: Path) -> Path:
    """将 TXT 字幕文件转换为 SRT 格式。

    Args:
        subtitle_path: TXT 字幕文件路径

    Returns:
        SRT 文件路径
    """
    with step_logger("字幕识别"):
        entries = parse_subtitles(subtitle_path)
        srt_path = subtitle_path.with_suffix(".srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, (start, end, text) in enumerate(entries, 1):
                f.write(f"{i}\n")
                f.write(
                    f"{seconds_to_srt_time(time_to_seconds(start))} --> {seconds_to_srt_time(time_to_seconds(end))}\n"
                )
                f.write(f"{text}\n\n")
        logger.info("SRT 已生成: %s（共 %d 条）", srt_path, len(entries))
        return srt_path
