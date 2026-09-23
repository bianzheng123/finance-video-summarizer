"""字幕识别工厂类 - 使用腾讯云ASR进行字幕识别"""

import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from .tencent_asr import TencentASRTranscriber
from .subtitle_io import save_subtitle
from ..utils.utils import sanitize_filename
from ..utils.time_utils import fmt_time
from ..knowledge_base import UnifiedVocabManager
from ..llm_summarize.schemas import SubtitleRow
from ..log_config import step_logger

logger = logging.getLogger(__name__)


class SubtitleRecognizer:
    """字幕识别工厂类

    使用腾讯云ASR进行字幕识别
    提供统一的 transcribe 接口，自动识别输入类型
    """

    def __init__(self, use_hotword: bool = True) -> None:
        """初始化转录器

        Args:
            use_hotword: 是否使用热词表增强识别，默认为 True。
                        对于直播场景建议设为 False（热词表需要约10分钟生效，直播内容实时无法受益）。
        """
        with step_logger("字幕识别"):
            # 加载环境变量
            _env_path = Path(__file__).parent.parent.parent / ".env"
            load_dotenv(dotenv_path=_env_path, override=True)

            secret_id = os.getenv("TENCENT_SECRET_ID")
            secret_key = os.getenv("TENCENT_SECRET_KEY")
            cos_bucket = os.getenv("TENCENT_COS_BUCKET")
            cos_region = os.getenv("TENCENT_COS_REGION", "ap-shanghai")

            if not secret_id or not secret_key:
                raise RuntimeError("腾讯云配置缺失（TENCENT_SECRET_ID / TENCENT_SECRET_KEY）")

            if not cos_bucket:
                raise RuntimeError("COS 配置缺失（TENCENT_COS_BUCKET）")

            # 根据 use_hotword 参数决定是否加载热词表
            hotword_id = None
            if use_hotword:
                # 使用统一的预创建热词表注入逻辑
                vocab_mgr = UnifiedVocabManager(secret_id=secret_id, secret_key=secret_key)
                hotword_id, success = vocab_mgr.inject_vocab_to_asr()
                if not success:
                    logger.warning("无法获取预创建热词表ID，将不使用热词表增强")
                    hotword_id = None
                else:
                    logger.info("热词表加载成功，ID: %s", hotword_id)
            else:
                logger.info("已禁用热词表（use_hotword=False）")

            logger.info("使用腾讯云 ASR 转录器，热词表ID: %s", hotword_id)
            self.transcriber = TencentASRTranscriber(
                secret_id, secret_key, cos_bucket, cos_region, hotword_id=hotword_id
            )
    
    def convert_to_wav(self, input_path: Path) -> Path:
        """使用 ffmpeg 转换为 16kHz mono WAV
        
        Args:
            input_path: 输入音频/视频文件路径
            
        Returns:
            转换后的 WAV 文件路径
        """
        return self.transcriber._convert_to_wav(input_path)
    
    def transcribe(self, source: Path | str, save_path: Path | str = None, platform: str = None, author: str = None, title: str = None) -> list[dict] | tuple[list[dict], Path, list[SubtitleRow]] | None:
        """转录音频/视频，可选保存结果

        自动识别输入类型：
        - Path: 调用 transcribe_from_file
        - str: 调用 transcribe_from_url

        当提供 save_path 时，会将 segments 转换为 subtitle_l 并保存文件，
        返回 (segments, output_dir, subtitle_l)；否则仅返回 segments。

        Args:
            source: 本地文件路径(Path) 或 URL(str)
            save_path: 完整的保存路径（可选）
            platform: 平台名称（可选，保存时需要）
            author: 作者名称（可选，保存时需要）
            title: 视频标题（可选，保存时需要）

        Returns:
            字幕列表，或 (字幕列表, 输出目录, 格式化字幕列表)
        """
        with step_logger("字幕识别"):
            if isinstance(source, Path):
                logger.info("从本地文件转录: %s", source.name)
                segments = self.transcriber.transcribe_from_file(source)
            else:
                logger.info("从 URL 转录: %s", source[:80] if len(source) > 80 else source)
                segments = self.transcriber.transcribe_from_url(source)

            if segments is None:
                return None

            subtitle_l = self._convert_segments_to_subtitle_l(segments)

            if save_path is not None:
                logger.info("字幕获取成功！平台: %s，UP主: %s，字幕数: %d", platform, author, len(segments))
                output_dir = self._save_subtitles(save_path, platform, author, title, segments)
                return segments, output_dir, subtitle_l

            return segments
    
    @staticmethod
    def find_existing_subtitle(folder: Path) -> Path | None:
        """查找已存在的字幕文件
        
        优先顺序:
        1. subtitle_tencent.txt（腾讯云ASR，带说话人标记）
        2. subtitle_bibigpt.txt（旧版BibiGPT，无说话人标记，向后兼容）
        
        Args:
            folder: 视频文件夹路径
            
        Returns:
            找到的字幕文件路径，或 None
        """
        with step_logger("字幕识别"):
            recognize_subtitle_dir = folder / "recognize_subtitle"
        
            # 优先查找腾讯云字幕
            tencent_file = recognize_subtitle_dir / "subtitle_tencent.txt"
            if tencent_file.exists():
                logger.info("找到已有字幕: %s", tencent_file)
                return tencent_file
        
            logger.info("未找到已有字幕文件")
            return None

    @staticmethod
    def _convert_segments_to_subtitle_l(subtitles: list) -> list[SubtitleRow]:
        """将原始 segments 转换为 subtitle_l（``SubtitleRow`` 模型）格式

        Args:
            subtitles: 字幕列表，每个元素包含 startTime, end, text, speaker

        Returns:
            格式化后的字幕列表（``SubtitleRow`` 实例）
        """
        subtitle_l = []
        for i, sub in enumerate(subtitles, 1):
            start = fmt_time(sub.get("startTime", 0))
            end = fmt_time(sub.get("end", 0))
            text = sub.get("text", "")
            speaker = sub.get("speaker")
            subtitle_l.append(SubtitleRow(
                rowID=i, start_time=start, end_time=end, speaker=speaker, text=text,
            ))
        return subtitle_l

    @staticmethod
    def _save_subtitles(save_path: Path | str, platform: str, author: str, title: str, segments: list) -> Path:
        """保存字幕文件和 subtitle_response.json

        Args:
            save_path: 完整的保存路径
            platform: 平台名称
            author: 作者名称
            title: 视频标题
            segments: 原始字幕列表

        Returns:
            输出目录路径
        """
        folder = Path(save_path)
        folder.mkdir(parents=True, exist_ok=True)

        recognize_subtitle_dir = folder / "recognize_subtitle"
        recognize_subtitle_dir.mkdir(parents=True, exist_ok=True)

        subtitle_file = recognize_subtitle_dir / "subtitle_tencent.txt"
        save_subtitle(segments, subtitle_file)

        result_data = {
            "service": platform,
            "detail": {
                "author": author,
                "title": title,
                "subtitlesArray": segments
            }
        }
        with open(recognize_subtitle_dir / "subtitle_response.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(result_data, ensure_ascii=False, indent=2))

        return folder
