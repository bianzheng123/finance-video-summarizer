"""腾讯云 ASR 转录器 - 使用腾讯云语音识别服务"""

import asyncio
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

from tencentcloud.asr.v20190614 import asr_client, models
from tencentcloud.common import credential as tencent_credential

from .base import BaseTranscriber
from ..client.bilibili import BilibiliClient
from ..client.youtube import YouTubeClient
from ..client.douyin import DouyinClient
from ..log_config import step_logger

logger = logging.getLogger(__name__)


class TencentASRTranscriber(BaseTranscriber):
    """腾讯云 ASR 转录器
    
    支持从本地文件和 URL 转录音频/视频
    URL 转录会自动下载视频并转换为 WAV
    """
    
    def __init__(self, secret_id: str, secret_key: str,
                 cos_bucket: str, cos_region: str = "ap-shanghai",
                 hotword_id: str | None = None) -> None:
        """初始化腾讯云转录器
        
        Args:
            secret_id: 腾讯云 Secret ID
            secret_key: 腾讯云 Secret Key
            cos_bucket: COS 存储桶名称
            cos_region: COS 区域，默认 ap-shanghai
            hotword_id: 预创建热词表ID，用于 ASR 热词增强识别
        """
        super().__init__(secret_id, secret_key, cos_bucket, cos_region)
        self._bilibili_client = BilibiliClient()
        self._youtube_client = YouTubeClient()
        self._douyin_client = DouyinClient()
        self._hotword_id = hotword_id
    
    def transcribe_from_file(self, wav_path: Path) -> list[dict]:
        """从本地 WAV 文件转录
        
        流程:
        1. 上传 WAV 到 COS
        2. 提交 ASR 任务
        3. 轮询任务状态
        4. 解析并返回结果
        
        Args:
            wav_path: WAV 文件路径
            
        Returns:
            字幕列表，每个元素包含 startTime, end, text, speaker
        """
        with step_logger("字幕识别"):
            # 1. 上传到 COS
            audio_url, cos_key = self._upload_to_cos(wav_path)

            try:
                # 2. 提交 ASR 任务
                task_id = self._submit_asr_task(audio_url)
                logger.info("ASR 任务已提交（开启说话人分离），TaskId: %s", task_id)

                # 3. 轮询任务状态
                result = self._poll_task_status(task_id)

                # 4. 解析并返回结果
                return self._parse_result(result)
            finally:
                # 结果拿到（或失败）后再删 COS 临时音频：ASR 为异步拉取，不能在拉取前删
                self._delete_from_cos(cos_key)
    
    def transcribe_from_url(self, url: str) -> list[dict]:
        """从 URL 转录（参考 batch_transcribe.py 88-150）
        
        流程:
        1. 使用 yt-dlp 下载视频（支持 B站、YouTube 等）
        2. 使用 ffmpeg 转换为 16kHz mono WAV
        3. 调用 transcribe_from_file 进行转录
        4. 清理临时文件
        
        Args:
            url: 视频页面 URL
            
        Returns:
            字幕列表，每个元素包含 startTime, end, text, speaker
        """
        with step_logger("字幕识别"):
            # 1. 创建临时目录
            temp_dir = Path(tempfile.mkdtemp())
            try:
                # 2. 下载视频
                video_path = self._download_video(url, temp_dir)
            
                # 3. 转换为 WAV
                wav_path = self._convert_to_wav(video_path)
            
                # 4. 转录
                return self.transcribe_from_file(wav_path)
            finally:
                # 5. 清理临时文件
                shutil.rmtree(temp_dir)
                logger.info("临时文件已清理: %s", temp_dir)
    
    def _download_video(self, url: str, output_dir: Path) -> Path:
        """下载视频文件（参考 batch_transcribe.py 的实现）
        
        Args:
            url: 视频 URL
            output_dir: 输出目录
            
        Returns:
            下载的视频文件路径
            
        Raises:
            RuntimeError: 当下载失败或文件不存在时
        """
        try:
            if "bilibili.com" in url:
                return self._bilibili_client.download_video(url, output_dir)
            elif "douyin.com" in url:
                return self._douyin_client.download_video(url, output_dir)
            else:
                return asyncio.run(self._youtube_client.download_video(url, output_dir))
        except Exception as e:
            logger.error("下载失败: %s - %s", url, e)
            raise RuntimeError(f"下载失败: {e}")
    
    def _submit_asr_task(self, audio_url: str) -> str:
        """提交 ASR 任务
        
        Args:
            audio_url: 音频文件 URL（COS 预签名 URL）
            
        Returns:
            任务 ID
        """
        cred = tencent_credential.Credential(self._secret_id, self._secret_key)
        client = asr_client.AsrClient(cred, "ap-shanghai")
        
        req = models.CreateRecTaskRequest()
        req.EngineModelType = "16k_zh_en_2.0"
        req.ChannelNum = 1
        req.ResTextFormat = 3  # 仅含说话人分离结果
        req.SourceType = 0
        req.Url = audio_url
        req.SpeakerDiarization = 1
        req.SpeakerNumber = 0
        
        # 注入预创建热词表增强识别
        if self._hotword_id:
            req.HotwordId = self._hotword_id
            logger.info("注入预创建热词表: %s", self._hotword_id)
        
        resp = client.CreateRecTask(req)
        return str(resp.Data.TaskId)
    
    def _poll_task_status(self, task_id: str, timeout: int = 7200) -> dict:
        """轮询任务状态
        
        Args:
            task_id: 任务 ID
            timeout: 超时时间（秒），默认 2 小时
            
        Returns:
            ASR 结果字典
            
        Raises:
            RuntimeError: 当任务失败时
            TimeoutError: 当超时时
        """
        poll_interval = 10
        cred = tencent_credential.Credential(self._secret_id, self._secret_key)
        client = asr_client.AsrClient(cred, "ap-shanghai")
        
        req = models.DescribeTaskStatusRequest()
        req.TaskId = int(task_id)
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            resp = client.DescribeTaskStatus(req)
            status = resp.Data.Status
            
            if status == 2:  # 成功
                logger.info("ASR 任务完成")
                return {
                    "ResultDetail": [
                        s._serialize() for s in (resp.Data.ResultDetail or [])
                    ]
                }
            elif status == 3:  # 失败
                raise RuntimeError(f"ASR 任务失败: {resp.Data.ErrorMsg}")
            
            logger.info("ASR 状态: %s，等待 %ds...", status, poll_interval)
            time.sleep(poll_interval)
        
        raise TimeoutError(f"ASR 超时（{timeout}s）")
    
    def _parse_result(self, result: dict) -> list[dict]:
        """解析 ASR 结果
        
        Args:
            result: ASR 结果字典
            
        Returns:
            字幕列表，每个元素包含 startTime, end, text, speaker
        """
        subtitle_segments = []
        
        for sentence in result.get("ResultDetail", []):
            speaker_id = sentence.get("SpeakerId", 0)
            start_ms = sentence.get("StartMs", 0)
            end_ms = sentence.get("EndMs", 0)
            subtitle_text = sentence.get("FinalSentence", "").strip()
            
            if not subtitle_text:
                continue
            
            subtitle_segments.append({
                "startTime": start_ms / 1000.0,
                "end": end_ms / 1000.0,
                "text": subtitle_text,
                "speaker": speaker_id,
            })
        
        subtitle_segments.sort(key=lambda x: x["startTime"])
        return subtitle_segments
    
    def _format_time(self, seconds: float) -> str:
        """将秒数格式化为 HH:MM:SS 格式
        
        Args:
            seconds: 时间（秒）
            
        Returns:
            格式化的时间字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def save_transcripts(self, result: list[dict], output_dir: Path) -> None:
        """保存转录结果为字幕文件
        
        Args:
            result: 字幕列表
            output_dir: 输出目录路径
        """
        with step_logger("字幕识别"):
            output_dir.mkdir(parents=True, exist_ok=True)
        
            # 保存字幕版本
            subtitle_path = output_dir / "transcribe_subtitle_tencent.txt"
            with open(subtitle_path, "w", encoding="utf-8") as f:
                for segment in result:
                    start_time = self._format_time(segment["startTime"])
                    end_time = self._format_time(segment["end"])
                    speaker_label = f"Speaker{segment['speaker']}" if "speaker" in segment else ""
                    if speaker_label:
                        f.write(f"{start_time}\t{end_time}\t{speaker_label}\t{segment['text']}\n")
                    else:
                        f.write(f"{start_time}\t{end_time}\t{segment['text']}\n")
            logger.info("字幕版本已保存: %s", subtitle_path)
