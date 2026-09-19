"""转录器抽象基类 - 定义通用接口和公共方法"""

import logging
import subprocess
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from qcloud_cos import CosConfig, CosS3Client

from ..log_config import step_logger

logger = logging.getLogger(__name__)


class BaseTranscriber(ABC):
    """转录器抽象基类
    
    提供公共方法：音频转换、COS上传
    子类必须实现：transcribe_from_file 和 transcribe_from_url
    """
    
    def __init__(self, secret_id: str, secret_key: str,
                 cos_bucket: str, cos_region: str = "ap-shanghai") -> None:
        """初始化基类
        
        Args:
            secret_id: 腾讯云 Secret ID
            secret_key: 腾讯云 Secret Key
            cos_bucket: COS 存储桶名称
            cos_region: COS 区域，默认 ap-shanghai
        """
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._cos_bucket = cos_bucket
        self._cos_region = cos_region
    
    @staticmethod
    def _probe_duration(path: Path) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe 失败 ({path.name}):\n{result.stderr.decode()}")
        out = result.stdout.decode().strip()
        if not out:
            raise RuntimeError(f"ffprobe 未返回时长: {path.name}")
        return float(out)

    def _convert_to_wav(self, input_path: Path) -> Path:
        """使用 ffmpeg 转换为 16kHz mono WAV（公共方法）

        采用容错解码 + 原子写入 + 时长校验，避免直播录制 mp4 中段损坏帧
        导致 wav 被截断（典型场景：直播过程中网络抖动，mp4 完整但默认
        ffmpeg 解码遇到坏帧后提前结束音频流）。

        Args:
            input_path: 输入音频/视频文件路径

        Returns:
            转换后的 WAV 文件路径

        Raises:
            RuntimeError: ffmpeg 转换失败，或转换后 wav 时长显著短于源文件
        """
        with step_logger("字幕识别"):
            wav_path = input_path.parent / (input_path.stem + ".wav")
            if wav_path.exists():
                logger.info("跳过转换（已存在）: %s", wav_path.name)
                return wav_path

            tmp_path = wav_path.with_suffix(".wav.tmp")
            tmp_path.unlink(missing_ok=True)

            logger.info("转换音频: %s → %s", input_path.name, wav_path.name)
            result = subprocess.run(
                ["ffmpeg", "-y",
                 "-err_detect", "ignore_err",
                 "-fflags", "+genpts+discardcorrupt",
                 "-i", str(input_path),
                 "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                 "-f", "wav",
                 str(tmp_path)],
                capture_output=True,
            )
            if result.returncode != 0:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"ffmpeg 失败:\n{result.stderr.decode()}")

            src_dur = self._probe_duration(input_path)
            wav_dur = self._probe_duration(tmp_path)
            logger.info("时长校验: 源=%.1fs wav=%.1fs", src_dur, wav_dur)
            # 时长差异只作提示，不阻断转录：容器时长元数据本身存在误差，且即便音频
            # 真有缺损，能转出来的部分仍可正常 ASR。是否可转录由 ffmpeg 退出码保证，
            # 这里仅在差异较大时给出 warning 便于排查。
            if src_dur - wav_dur > 60:
                logger.warning(
                    "wav 较源文件短 %.1fs (源=%.1fs wav=%.1fs)，"
                    "源文件 %s 可能在中段有损坏，将继续转录可用部分",
                    src_dur - wav_dur, src_dur, wav_dur, input_path,
                )

            tmp_path.replace(wav_path)
            logger.info("音频已保存: %s", wav_path)
            return wav_path
    
    def _upload_to_cos(self, wav_path: Path, expire_seconds: int = 7200) -> tuple[str, str]:
        """上传文件到腾讯云 COS，返回 (预签名 URL, COS object key)（公共方法）

        Args:
            wav_path: WAV 文件路径
            expire_seconds: URL 过期时间（秒），默认 2 小时

        Returns:
            (预签名 URL, COS object key)。key 供转录结束后 _delete_from_cos 清理。

        Raises:
            RuntimeError: 当上传失败时
        """
        with step_logger("字幕识别"):
            config = CosConfig(
                Region=self._cos_region,
                SecretId=self._secret_id,
                SecretKey=self._secret_key
            )
            client = CosS3Client(config)
            # 所有直播录制文件都叫 video.wav，多个录制任务并发上传会用同一个 key
            # asr/video.wav 互相覆盖：A 上传后、ASR 异步拉取前若 B 覆盖了同名对象，
            # A 会拿到 B 的音频 → 字幕与视频对不上。故每次上传用唯一 key 隔离。
            cos_key = f"asr/{uuid.uuid4().hex}_{wav_path.name}"

            logger.info("上传到 COS: %s/%s", self._cos_bucket, cos_key)
            with open(wav_path, "rb") as f:
                client.put_object(Bucket=self._cos_bucket, Body=f, Key=cos_key)

            url = client.get_presigned_url(
                Method="GET",
                Bucket=self._cos_bucket,
                Key=cos_key,
                Expired=expire_seconds
            )
            logger.info("COS 上传完成")
            return url, cos_key

    def _delete_from_cos(self, cos_key: str) -> None:
        """删除 COS 上的临时音频对象（公共方法）

        ASR 任务为腾讯云异步拉取，必须在转录结果拿到后再删，否则会删掉尚未被拉取的音频。
        删除失败只 warn 不抛出：对象本身有唯一 key、不会污染其他任务，残留至多是存储成本问题。

        Args:
            cos_key: 上传时返回的 COS object key
        """
        with step_logger("字幕识别"):
            config = CosConfig(
                Region=self._cos_region,
                SecretId=self._secret_id,
                SecretKey=self._secret_key
            )
            client = CosS3Client(config)
            try:
                client.delete_object(Bucket=self._cos_bucket, Key=cos_key)
                logger.info("已清理 COS 临时音频: %s/%s", self._cos_bucket, cos_key)
            except Exception as e:
                logger.warning("清理 COS 临时音频失败（可忽略）: %s/%s - %s",
                               self._cos_bucket, cos_key, e)
    
    @abstractmethod
    def transcribe_from_file(self, wav_path: Path) -> list[dict]:
        """从本地 WAV 文件转录
        
        Args:
            wav_path: WAV 文件路径
            
        Returns:
            字幕列表，每个元素包含 startTime, end, text 等字段
        """
        pass
    
    @abstractmethod
    def transcribe_from_url(self, url: str) -> list[dict] | None:
        """从 URL 转录
        
        Args:
            url: 音频/视频 URL 或视频页面 URL
            
        Returns:
            字幕列表，每个元素包含 startTime, end, text, speaker（如可用）
        """
        pass
