"""统一词汇表管理器 - 合并热词表管理和腾讯云API功能

将热词表本地管理与腾讯云API调用合并为一个类，支持：
1. 热词表的本地加载/保存（支持两种格式：权重格式、正确词格式）
2. 腾讯云热词表API的封装（创建、更新、查询、删除、设置默认）
3. 本地格式与腾讯云格式的双向转换
4. 热词表注入到ASR流程
"""

import json
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class HotwordFormat(Enum):
    """热词表数据格式类型"""
    WEIGHT_BASED = "weight_based"  # {"权重": [热词数组]}
    CORRECT_WORD = "correct_word"   # {"正确词": ["错词", ...]}


class UnifiedVocabManager:
    """统一词汇表管理器，合并本地热词表管理与腾讯云API功能"""

    def __init__(
        self,
        config_path: Path | None = None,
        format_type: HotwordFormat = HotwordFormat.WEIGHT_BASED,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "ap-shanghai"
    ) -> None:
        """初始化统一词汇表管理器
        
        Args:
            config_path: 本地热词表文件路径
            format_type: 数据格式类型，默认使用权重格式
            secret_id: 腾讯云 Secret ID（可选，仅在使用云API时需要）
            secret_key: 腾讯云 Secret Key（可选，仅在使用云API时需要）
            region: 腾讯云区域，默认 ap-shanghai
        """
        self.config_path = config_path
        self.format_type = format_type
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region
        self._data: dict = {}
        
        # 初始化腾讯云客户端（如果提供了凭证）
        self._cloud_client = None
        if secret_id and secret_key:
            try:
                from tencentcloud.asr.v20190614 import asr_client, models
                from tencentcloud.common import credential
                
                cred = credential.Credential(secret_id, secret_key)
                self._cloud_client = asr_client.AsrClient(cred, region)
                logger.info("[统一词汇表] 腾讯云API客户端初始化成功，区域: %s", region)
            except ImportError:
                logger.warning("[统一词汇表] 未安装tencentcloud-sdk-python，云端功能将不可用")
            except Exception as e:
                logger.warning("[统一词汇表] 腾讯云API客户端初始化失败: %s，云端功能将不可用", e)
        
        # 加载本地数据
        self._load()

    @classmethod
    def from_weight_format(
        cls,
        config_path: Path | None = None,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None
    ) -> "UnifiedVocabManager":
        """从权重格式创建管理器（向后兼容原 DatabaseHotwordManager）"""
        return cls(
            config_path=config_path,
            format_type=HotwordFormat.WEIGHT_BASED,
            secret_id=secret_id,
            secret_key=secret_key
        )

    def _load(self) -> None:
        """从文件加载热词表"""
        if not self.config_path or not self.config_path.exists():
            logger.info("[统一词汇表] 配置文件不存在，初始化为空: %s", self.config_path)
            return

        try:
            with open(self.config_path, encoding="utf-8") as f:
                loaded = json.load(f)

            if not isinstance(loaded, dict):
                logger.warning("[统一词汇表] 配置文件格式异常，重置为空")
                self._data = {}
            else:
                self._data = loaded
                if self.format_type == HotwordFormat.WEIGHT_BASED:
                    total_words = sum(len(v) for v in self._data.values() if isinstance(v, list))
                    logger.info("[统一词汇表] 加载成功（权重格式），共 %d 个权重级别，%d 个热词",
                               len(self._data), total_words)
                else:
                    logger.info("[统一词汇表] 加载成功（正确词格式），共 %d 个词", len(self._data))
        except Exception as e:
            logger.warning("[统一词汇表] 加载失败: %s，降级为空库", e)
            self._data = {}

    def _save(self) -> None:
        """保存热词表到文件（原子写入）"""
        if not self.config_path:
            logger.warning("[统一词汇表] 未指定配置文件路径，跳过保存")
            return

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.config_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            tmp_path.replace(self.config_path)
            logger.info("[统一词汇表] 已保存到: %s", self.config_path)
        except Exception as e:
            logger.warning("[统一词汇表] 保存失败: %s", e)

    # ---------- 本地热词表管理 ----------

    def to_tencent_vocab_format(self, default_weight: int = 10) -> list:
        """转换为腾讯云热词表格式

        将内部格式转换为 [{"Word": "热词", "Weight": 权重}]

        Returns:
            腾讯云热词表格式数组
        """
        result = []

        if self.format_type == HotwordFormat.WEIGHT_BASED:
            # 权重格式: {"权重": [热词数组]}
            for weight_str, words in self._data.items():
                try:
                    weight = int(weight_str)
                except ValueError:
                    logger.warning("[统一词汇表] 无效权重值: %s，已跳过", weight_str)
                    continue

                if not isinstance(words, list):
                    logger.warning("[统一词汇表] 权重 %s 对应的值不是数组，已跳过", weight_str)
                    continue

                for word in words:
                    word = word.strip()
                    if not word:
                        continue

                    # 热词长度限制：不超过10个字
                    if len(word) > 10:
                        logger.warning("[统一词汇表] 热词长度超过10个字，已跳过: %s", word)
                        continue

                    result.append({"Word": word, "Weight": weight})

        else:
            # 正确词格式: {"正确词": ["错词1", "错词2"]} 或 {"正确词": {"incorrect_words": [...], "weight": 10}}
            for word, value in self._data.items():
                # 向后兼容：旧格式是列表，新格式是字典
                if isinstance(value, list):
                    weight = default_weight
                elif isinstance(value, dict):
                    weight = value.get("weight", default_weight)
                else:
                    continue

                # 热词长度限制：不超过10个字
                if len(word) > 10:
                    logger.warning("[统一词汇表] 热词长度超过10个字，已跳过: %s", word)
                    continue

                result.append({"Word": word, "Weight": weight})

        logger.info("[统一词汇表] 转换为腾讯云格式，共 %d 个热词", len(result))
        return result

    def from_tencent_vocab_format(self, vocab_content: list) -> None:
        """从腾讯云热词表格式转换为本地格式

        Args:
            vocab_content: 腾讯云热词表格式数组
        """
        self._data = {}

        for item in vocab_content:
            # 处理腾讯云 SDK 返回的 HotWord 对象
            if hasattr(item, 'Word'):
                word = item.Word.strip() if item.Word else ""
                weight = item.Weight if hasattr(item, 'Weight') and item.Weight else 10
            # 处理普通字典格式
            elif isinstance(item, dict):
                word = item.get("Word", "").strip()
                weight = item.get("Weight", 10)
            else:
                continue

            if not word:
                continue

            if self.format_type == HotwordFormat.WEIGHT_BASED:
                # 权重格式
                weight_str = str(weight)
                if weight_str not in self._data:
                    self._data[weight_str] = []
                if word not in self._data[weight_str]:
                    self._data[weight_str].append(word)
            else:
                # 正确词格式
                self._data[word] = {
                    "incorrect_words": [],
                    "weight": weight
                }

        self._save()
        logger.info("[统一词汇表] 已从腾讯云格式导入，共 %d 个词", len(self._data))

    def merge_candidates(self, candidates: list[dict]) -> None:
        """合并人工校验后的易错词候选到热词表

        由 tools/merge_candidates.py 调用，经人工确认后执行。

        Args:
            candidates: 易错词候选列表，每项格式:
                {"correct_word": "正确词", "incorrect_words": ["错词1", "错词2"]}
        """
        if not candidates:
            return

        if self.format_type == HotwordFormat.CORRECT_WORD:
            updated = 0

            for item in candidates:
                correct_word = item.get("correct_word", "").strip()
                incorrect_words = item.get("incorrect_words", [])

                if not correct_word or not incorrect_words:
                    continue

                # 确保 incorrect_words 是列表
                if isinstance(incorrect_words, str):
                    incorrect_words = [incorrect_words]

                if correct_word in self._data:
                    # 合并错词列表（去重）
                    existing_incorrect = self._data[correct_word]
                    if isinstance(existing_incorrect, list):
                        combined = list(set(existing_incorrect + incorrect_words))
                        self._data[correct_word] = combined
                    else:
                        self._data[correct_word] = incorrect_words
                else:
                    self._data[correct_word] = incorrect_words
                    updated += 1

            self._save()
            logger.info("[统一词汇表] 合并完成，新增 %d 个词，总计 %d 个", updated, len(self._data))

    def get_all_words(self) -> set[str]:
        """获取所有已知热词集合

        Returns:
            热词集合
        """
        result = set()
        if self.format_type == HotwordFormat.WEIGHT_BASED:
            for words in self._data.values():
                if isinstance(words, list):
                    result.update(words)
        else:
            result.update(self._data.keys())
        return result

    # ---------- 腾讯云API功能 ----------

    def create_or_update_vocab(
        self,
        vocab_id: str,
        vocab_name: str,
        description: str = "金融媒体热词表"
    ) -> str:
        """创建或更新腾讯云热词表
        
        先尝试更新热词表，如果vocabId不存在则创建新的热词表。
        
        Args:
            vocab_id: 热词表ID（自定义，如 "financial_media_vocab"）
            vocab_name: 热词表名称
            description: 热词表描述
            
        Returns:
            热词表ID
            
        Raises:
            RuntimeError: 腾讯云客户端未初始化
        """
        if not self._cloud_client:
            raise RuntimeError("腾讯云API客户端未初始化，请提供secret_id和secret_key参数")

        # 获取腾讯云格式的热词表内容
        vocab_content = self.to_tencent_vocab_format()
        
        # 先尝试更新
        try:
            from tencentcloud.asr.v20190614 import models
            update_req = models.UpdateAsrVocabRequest()
            update_req.VocabId = vocab_id
            update_req.Name = vocab_name
            update_req.WordWeights = vocab_content
            update_req.Description = description

            self._cloud_client.UpdateAsrVocab(update_req)
            logger.info("[统一词汇表] 更新成功: %s (%s)", vocab_name, vocab_id)
            return vocab_id
        except Exception as e:
            # 如果更新失败（可能是vocabId不存在），尝试创建
            if "InvalidVocabId" in str(e) or "vocabId not exist" in str(e):
                create_req = models.CreateAsrVocabRequest()
                create_req.Name = vocab_name
                create_req.WordWeights = vocab_content
                create_req.Description = description
                
                resp = self._cloud_client.CreateAsrVocab(create_req)
                created_vocab_id = resp.VocabId if hasattr(resp, 'VocabId') else vocab_id
                logger.info("[统一词汇表] 创建成功: %s (%s)", vocab_name, created_vocab_id)
                return created_vocab_id
            else:
                raise

    def get_vocab(self, vocab_id: str) -> dict:
        """获取腾讯云热词表内容
        
        Args:
            vocab_id: 热词表ID
            
        Returns:
            热词表信息字典，包含 name, words, create_time, update_time
            
        Raises:
            RuntimeError: 腾讯云客户端未初始化
        """
        if not self._cloud_client:
            raise RuntimeError("腾讯云API客户端未初始化，请提供secret_id和secret_key参数")

        try:
            from tencentcloud.asr.v20190614 import models
            req = models.GetAsrVocabRequest()
            req.VocabId = vocab_id
            resp = self._cloud_client.GetAsrVocab(req)

            result = {
                "name": resp.Name,
                "words": resp.WordWeights,
                "create_time": resp.CreateTime,
                "update_time": resp.UpdateTime,
            }
            word_count = len(resp.WordWeights) if isinstance(resp.WordWeights, list) else 0
            logger.info("[统一词汇表] 获取成功: %s, 词数: %d", vocab_id, word_count)
            return result
        except Exception as e:
            logger.error("[统一词汇表] 获取热词表失败: %s", e)
            raise

    def set_default_vocab(self, vocab_id: str) -> None:
        """设置默认热词表
        
        设置后，调用ASR服务时若未指定热词表ID，将自动生效此热词表。
        
        Args:
            vocab_id: 热词表ID
            
        Raises:
            RuntimeError: 腾讯云客户端未初始化
        """
        if not self._cloud_client:
            raise RuntimeError("腾讯云API客户端未初始化，请提供secret_id和secret_key参数")

        try:
            from tencentcloud.asr.v20190614 import models
            req = models.SetVocabStateRequest()
            req.VocabId = vocab_id
            req.State = 1  # 1=默认
            self._cloud_client.SetVocabState(req)
            logger.info("[统一词汇表] 已设为默认: %s", vocab_id)
        except Exception as e:
            logger.error("[统一词汇表] 设置默认热词表失败: %s", e)
            raise

    def list_vocab(self) -> list[dict]:
        """列举所有腾讯云热词表
        
        Returns:
            热词表列表，每个元素包含 vocab_id, name, word_number, create_time, update_time, state
            
        Raises:
            RuntimeError: 腾讯云客户端未初始化
        """
        if not self._cloud_client:
            raise RuntimeError("腾讯云API客户端未初始化，请提供secret_id和secret_key参数")

        try:
            from tencentcloud.asr.v20190614 import models
            req = models.GetAsrVocabListRequest()
            resp = self._cloud_client.GetAsrVocabList(req)

            result = []
            if resp.VocabList:
                for vocab in resp.VocabList:
                    word_number = len(vocab.WordWeights) if isinstance(vocab.WordWeights, list) else 0
                    result.append({
                        "vocab_id": vocab.VocabId,
                        "name": vocab.Name,
                        "word_number": word_number,
                        "create_time": vocab.CreateTime,
                        "update_time": vocab.UpdateTime,
                        "state": vocab.State,  # 0=普通，1=默认
                    })
            
            logger.info("[统一词汇表] 列举完成，共 %d 个热词表", len(result))
            return result
        except Exception as e:
            logger.error("[统一词汇表] 列举热词表失败: %s", e)
            raise

    def get_default_vocab_id(self) -> Optional[str]:
        """获取当前默认热词表ID
        
        Returns:
            默认热词表ID，如果没有则返回 None
            
        Raises:
            RuntimeError: 腾讯云客户端未初始化
        """
        if not self._cloud_client:
            raise RuntimeError("腾讯云API客户端未初始化，请提供secret_id和secret_key参数")

        try:
            vocab_list = self.list_vocab()
            for vocab in vocab_list:
                if vocab["state"] == 1:
                    return vocab["vocab_id"]
            return None
        except Exception as e:
            logger.error("[统一词汇表] 获取默认热词表ID失败: %s", e)
            return None

    # ---------- 热词表注入功能 ----------

    def get_hotword_id_from_env(self, env_path: Path | None = None) -> str | None:
        """从环境变量获取热词表ID
        
        获取当前配置的热词表ID
        
        Args:
            env_path: .env 文件路径，默认为项目根目录下的 .env
            
        Returns:
            热词表ID，如果没有则返回None
        """
        if env_path is None:
            # 计算项目根目录路径：从当前文件向上4层父目录
            env_path = Path(__file__).parent.parent.parent.parent / ".env"
        
        if not env_path.exists():
            logger.warning("[统一词汇表] .env 文件不存在: %s", env_path)
            return None
        
        try:
            # 读取 .env 文件内容
            env_content = env_path.read_text(encoding="utf-8")
            lines = env_content.splitlines()
            
            for line in lines:
                if line.strip().startswith("TENCENT_HOTWORD_ID="):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        hotword_id = parts[1].strip()
                        if hotword_id:
                            return hotword_id
            return None
        except Exception as e:
            logger.error("[统一词汇表] 读取热词表ID失败: %s", e)
            return None

    def update_env_hotword_id(self, vocab_id: str, env_path: Path | None = None) -> bool:
        """更新环境变量文件中的热词表ID
        
        修改 .env 文件中的 TENCENT_HOTWORD_ID 值，使其指向指定的热词表ID
        
        Args:
            vocab_id: 要设置的腾讯云热词表ID
            env_path: .env 文件路径，默认为项目根目录下的 .env
            
        Returns:
            是否更新成功
        """
        if env_path is None:
            # 计算项目根目录路径：从当前文件向上4层父目录
            env_path = Path(__file__).parent.parent.parent.parent / ".env"
        
        if not env_path.exists():
            logger.error("[统一词汇表] .env 文件不存在: %s", env_path)
            return False
        
        try:
            # 读取 .env 文件内容
            env_content = env_path.read_text(encoding="utf-8")
            lines = env_content.splitlines()
            
            # 查找 TENCENT_HOTWORD_ID 行
            updated = False
            new_lines = []
            for line in lines:
                if line.strip().startswith("TENCENT_HOTWORD_ID="):
                    new_lines.append(f"TENCENT_HOTWORD_ID={vocab_id}")
                    updated = True
                    logger.info("[统一词汇表] 已更新热词表ID: %s", vocab_id)
                else:
                    new_lines.append(line)
            
            # 如果没找到，则添加新行
            if not updated:
                new_lines.append(f"TENCENT_HOTWORD_ID={vocab_id}")
                logger.info("[统一词汇表] 已添加热词表ID: %s", vocab_id)
            
            # 写入文件
            env_path.write_text("\n".join(new_lines), encoding="utf-8")
            return True
            
        except Exception as e:
            logger.error("[统一词汇表] 更新热词表ID失败: %s", e)
            return False

    def inject_vocab_to_asr(self, env_path: Path | None = None) -> tuple[str | None, bool]:
        """为ASR注入热词表的统一入口
        
        从环境变量或云端获取热词表ID，供工作流程文件调用
        
        Args:
            env_path: .env 文件路径，默认为项目根目录下的 .env
            
        Returns:
            (hotword_id, success) 元组
            如果成功获取热词表ID，则返回 (hotword_id, True)
            如果失败，则返回 (None, False)
        """
        # 计算项目根目录的 .env 路径
        if env_path is None:
            # 计算项目根目录路径：从当前文件向上4层父目录
            env_path = Path(__file__).parent.parent.parent.parent / ".env"
        
        if not env_path.exists():
            logger.warning("[统一词汇表] .env 文件不存在: %s", env_path)
            return None, False
        
        try:
            load_dotenv(dotenv_path=env_path, override=True)
        except Exception as e:
            logger.warning("[统一词汇表] 加载环境变量失败: %s", e)
            return None, False
        
        # 从环境变量获取热词表ID
        hotword_id = self.get_hotword_id_from_env(env_path)
        
        if hotword_id:
            logger.info("[统一词汇表] 使用环境变量中的热词表ID: %s", hotword_id)
            return hotword_id, True
        
        # 如果环境变量中没有，尝试从云端获取默认热词表ID
        logger.info("[统一词汇表] 环境变量中没有热词表ID，尝试从云端获取默认热词表")
        
        # 如果云客户端未初始化，尝试从环境变量获取凭证
        if not self._cloud_client:
            secret_id = os.getenv("TENCENT_SECRET_ID")
            secret_key = os.getenv("TENCENT_SECRET_KEY")
            
            if not secret_id or not secret_key:
                logger.warning("[统一词汇表] 腾讯云凭证缺失，无法从云端获取热词表")
                return None, False
            
            try:
                # 临时初始化云客户端
                self.secret_id = secret_id
                self.secret_key = secret_key
                from tencentcloud.asr.v20190614 import asr_client
                from tencentcloud.common import credential
                
                cred = credential.Credential(secret_id, secret_key)
                self._cloud_client = asr_client.AsrClient(cred, self.region)
                logger.info("[统一词汇表] 临时初始化腾讯云API客户端成功")
            except Exception as e:
                logger.warning("[统一词汇表] 临时初始化云客户端失败: %s", e)
                return None, False
        
        try:
            default_vocab_id = self.get_default_vocab_id()
            
            if default_vocab_id:
                logger.info("[统一词汇表] 从云端获取到默认热词表ID: %s", default_vocab_id)
                # 更新到环境变量
                self.update_env_hotword_id(default_vocab_id, env_path)
                return default_vocab_id, True
            else:
                logger.warning("[统一词汇表] 云端没有默认热词表")
                return None, False
        except Exception as e:
            logger.warning("[统一词汇表] 从云端获取热词表失败: %s", e)
            return None, False

    # ---------- 属性 ----------

    @property
    def is_empty(self) -> bool:
        """热词表是否为空"""
        return len(self.get_all_words()) == 0

    @property
    def has_cloud_client(self) -> bool:
        """是否已初始化腾讯云API客户端"""
        return self._cloud_client is not None

    @property
    def data(self) -> dict:
        """返回内部数据（只读）"""
        return self._data