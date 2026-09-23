"""黑话板块映射管理器（单例模式）

管理"板块黑话→正式板块名"映射，支持从 database 目录加载，
提供 LLM prompt 上下文生成和黑话映射查询。

数据文件：
- database/jargon_sector.json: 板块名称库
  顶层结构：
    {
      "黑话": {黑话: [正式板块名]},
      "白话": [无别名的板块名, ...]
    }
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JargonSectorManager:
    """黑话板块映射管理器（单例模式）"""

    _instance = None
    _initialized = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "JargonSectorManager":
        """单例模式：确保只有一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        jargon_config_path: Optional[Path] = None
    ) -> None:
        """初始化板块黑话管理器

        Args:
            jargon_config_path: 板块黑话数据库文件路径，默认使用 database/jargon_sector.json
        """
        if self._initialized:
            return

        self.jargon_config_path = jargon_config_path or (
            Path(__file__).parent.parent.parent / "database" / "jargon_sector.json"
        )

        self._jargon_data: dict = {}  # {黑话: [正式板块名]}
        self._jargon_reverse_map: dict = {}  # {正式板块名: [黑话]}
        self._plain_names: list[str] = []  # 无别名的板块名（"白话"）

        self._load_jargon_data()

        self._initialized = True

    def _load_jargon_data(self) -> None:
        """从文件加载板块名称库，并生成反向映射"""
        if not self.jargon_config_path.exists():
            logger.info("[板块黑话库] 配置文件不存在，初始化为空: %s", self.jargon_config_path)
            return

        try:
            with open(self.jargon_config_path, encoding="utf-8") as f:
                raw = json.load(f)

            if not isinstance(raw, dict):
                logger.warning("[板块黑话库] 顶层结构不是 dict，降级为空库: %s", type(raw).__name__)
                return

            jargon_part = raw.get("黑话", {})
            if isinstance(jargon_part, dict):
                self._jargon_data = jargon_part
            else:
                logger.warning("[板块黑话库] 黑话 字段不是 dict，忽略: %s", type(jargon_part).__name__)
                self._jargon_data = {}

            plain_part = raw.get("白话", [])
            if isinstance(plain_part, list):
                self._plain_names = [n for n in plain_part if isinstance(n, str)]
            else:
                logger.warning("[板块黑话库] 白话 字段不是 list，忽略: %s", type(plain_part).__name__)
                self._plain_names = []

            self._jargon_reverse_map = {}
            for jargon, formal_names in self._jargon_data.items():
                if isinstance(formal_names, list):
                    for formal_name in formal_names:
                        if formal_name not in self._jargon_reverse_map:
                            self._jargon_reverse_map[formal_name] = []
                        if jargon not in self._jargon_reverse_map[formal_name]:
                            self._jargon_reverse_map[formal_name].append(jargon)

            logger.info(
                "[板块黑话库] 加载成功，黑话 %d 条，反向映射 %d 个正式板块名，白话 %d 个",
                len(self._jargon_data),
                len(self._jargon_reverse_map),
                len(self._plain_names),
            )
        except Exception as e:
            logger.warning("[板块黑话库] 加载失败: %s，降级为空库", e)

    def get_jargon_formal_map(self) -> dict:
        """获取板块黑话到正式板块名列表的映射（支持一对多）

        例如："光" -> ["光通讯"]，"VC" -> ["电解液"]

        Returns:
            {黑话: [正式板块名列表]}
        """
        result = {}
        for jargon, formal_names in self._jargon_data.items():
            if isinstance(formal_names, list) and formal_names:
                result[jargon] = formal_names
        return result

    def get_jargon_reverse_map(self) -> dict:
        """获取正式板块名到板块黑话列表的映射

        Returns:
            {正式板块名: [黑话列表]}
        """
        return self._jargon_reverse_map

    def get_all_jargons(self) -> list[str]:
        """获取所有板块名称列表（黑话 keys + 白话）"""
        result = list(self._jargon_data.keys())
        for name in self._plain_names:
            if name not in result:
                result.append(name)
        return result

    def get_all_formal_names(self) -> set[str]:
        """获取所有正式板块名集合（黑话 values 扁平 + 白话）"""
        names = set()
        for formal_names in self._jargon_data.values():
            if isinstance(formal_names, list):
                names.update(formal_names)
        names.update(self._plain_names)
        return names

    def get_knowledge_context(self) -> str:
        """生成适合追加到 LLM prompt 的板块黑话知识库上下文文本"""
        lines = []
        if self._jargon_data:
            lines.append("【已知板块黑话映射】")
            lines.append("以下正式板块名在视频中可能以黑话/简称出现，提取关键词时请识别：")
            for jargon, formal_names in list(self._jargon_data.items())[:30]:
                formal_names_str = "、".join(formal_names)
                lines.append(f"  - {formal_names_str}（黑话：{jargon}）")
            if len(self._jargon_data) > 30:
                lines.append(f"  ... 等共 {len(self._jargon_data)} 个")
            lines.append("")
        return "\n".join(lines) if lines else ""

    @property
    def is_empty(self) -> bool:
        """知识库是否为空"""
        return len(self._jargon_data) == 0 and len(self._plain_names) == 0

    @property
    def data(self) -> dict:
        """返回内部数据（只读）"""
        return {
            "板块黑话数据库": self._jargon_data,
            "板块白话名称": self._plain_names,
        }

    # ---------- 静态方法：候选提取 ----------

    @staticmethod
    def extract_jargon_candidates(
        keyword_data: dict,
        existing_names: Optional[set[str]] = None,
        existing_jargon_map: Optional[dict[str, str]] = None
    ) -> dict:
        """从关键词提取结果中提取增量板块黑话候选

        Args:
            keyword_data: KeywordExtractor 返回的关键词数据
            existing_names: 已知正式板块名集合
            existing_jargon_map: 板块黑话→正式板块名映射

        Returns:
            候选字典，格式: {"板块黑话数据库": {"正式板块名（推测）": ["黑话候选"]}}
        """
        if existing_names is None:
            existing_names = set()
        if existing_jargon_map is None:
            existing_jargon_map = {}

        candidates: dict = {"板块黑话数据库": {}}

        for topic, topic_data in keyword_data.items():
            topic_stripped = topic.strip()
            if not topic_stripped:
                continue

            if isinstance(topic_data, dict):
                for jargon in topic_data.get("板块黑话", []):
                    jargon = jargon.strip()
                    if not jargon:
                        continue
                    if jargon not in existing_jargon_map and jargon not in existing_names:
                        if topic_stripped not in candidates["板块黑话数据库"]:
                            candidates["板块黑话数据库"][topic_stripped] = []
                        if jargon not in candidates["板块黑话数据库"][topic_stripped]:
                            candidates["板块黑话数据库"][topic_stripped].append(jargon)

            if topic_stripped in existing_names:
                continue
            if topic_stripped in existing_jargon_map:
                continue

            candidates["板块黑话数据库"][topic_stripped] = []

        logger.info("[板块黑话候选] 提取到 %d 个板块黑话候选", len(candidates["板块黑话数据库"]))
        return candidates


def get_sector_manager() -> JargonSectorManager:
    """获取 JargonSectorManager 单例的便捷函数"""
    return JargonSectorManager()
