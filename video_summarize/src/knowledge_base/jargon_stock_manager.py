"""黑话个股映射管理器（单例模式）

管理"黑话→正式名称"映射，支持从 database 目录加载，
提供 LLM prompt 上下文生成和黑话映射生成。

数据文件：
- database/jargon_stock.json: 个股名称库
  顶层结构：
    {
      "黑话": {黑话: [正式名称]},
      "白话": [无别名的纯名称, ...]
    }
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JargonStockManager:
    """黑话个股映射管理器（单例模式）"""

    _instance = None
    _initialized = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "JargonStockManager":
        """单例模式：确保只有一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        jargon_config_path: Optional[Path] = None
    ) -> None:
        """初始化黑话个股管理器

        Args:
            jargon_config_path: 黑话数据库文件路径，默认使用 database/jargon_stock.json
        """
        if self._initialized:
            return

        self.jargon_config_path = jargon_config_path or (
            Path(__file__).parent.parent.parent / "database" / "jargon_stock.json"
        )

        self._jargon_data: dict = {}  # {黑话: [正式名称]}
        self._jargon_reverse_map: dict = {}  # {正式名称: [黑话]} 反向映射
        self._plain_names: list[str] = []  # 无别名的纯名称（"白话"）

        self._load_jargon_data()

        self._initialized = True

    def _load_jargon_data(self) -> None:
        """从文件加载个股名称库，并生成反向映射"""
        if not self.jargon_config_path.exists():
            logger.info("[黑话库] 配置文件不存在，初始化为空: %s", self.jargon_config_path)
            return

        try:
            with open(self.jargon_config_path, encoding="utf-8") as f:
                raw = json.load(f)

            if not isinstance(raw, dict):
                logger.warning("[黑话库] 顶层结构不是 dict，降级为空库: %s", type(raw).__name__)
                return

            jargon_part = raw.get("黑话", {})
            if isinstance(jargon_part, dict):
                self._jargon_data = jargon_part
            else:
                logger.warning("[黑话库] 黑话 字段不是 dict，忽略: %s", type(jargon_part).__name__)
                self._jargon_data = {}

            plain_part = raw.get("白话", [])
            if isinstance(plain_part, list):
                self._plain_names = [n for n in plain_part if isinstance(n, str)]
            else:
                logger.warning("[黑话库] 白话 字段不是 list，忽略: %s", type(plain_part).__name__)
                self._plain_names = []

            # 生成反向映射：正式名称 -> [黑话]
            self._jargon_reverse_map = {}
            for jargon, formal_names in self._jargon_data.items():
                if isinstance(formal_names, list):
                    for formal_name in formal_names:
                        if formal_name not in self._jargon_reverse_map:
                            self._jargon_reverse_map[formal_name] = []
                        if jargon not in self._jargon_reverse_map[formal_name]:
                            self._jargon_reverse_map[formal_name].append(jargon)

            logger.info(
                "[黑话库] 加载成功，黑话 %d 条，反向映射 %d 个正式名称，白话 %d 个",
                len(self._jargon_data),
                len(self._jargon_reverse_map),
                len(self._plain_names),
            )
        except Exception as e:
            logger.warning("[黑话库] 加载失败: %s，降级为空库", e)

    def get_jargon_formal_map(self) -> dict:
        """获取黑话到正式名称列表的映射（支持一对多）

        例如："锂矿二傻" -> ["天齐锂业", "赣锋锂业"]

        Returns:
            {黑话: [正式名称列表]}
        """
        result = {}
        for jargon, formal_names in self._jargon_data.items():
            if isinstance(formal_names, list) and formal_names:
                result[jargon] = formal_names
        return result

    def get_jargon_reverse_map(self) -> dict:
        """获取正式名称到黑话列表的映射

        Returns:
            {正式名称: [黑话列表]}
        """
        return self._jargon_reverse_map

    def get_all_jargons(self) -> list[str]:
        """获取所有名称列表（黑话 keys + 白话）

        Returns:
            黑话与白话名称列表
        """
        result = list(self._jargon_data.keys())
        for name in self._plain_names:
            if name not in result:
                result.append(name)
        return result

    def get_all_formal_names(self) -> set[str]:
        """获取所有正式名称集合（黑话 values 扁平 + 白话）

        Returns:
            正式名称集合
        """
        names = set()
        for formal_names in self._jargon_data.values():
            if isinstance(formal_names, list):
                names.update(formal_names)
        names.update(self._plain_names)
        return names

    def get_knowledge_context(self) -> str:
        """生成适合追加到 LLM prompt 的知识库上下文文本

        Returns:
            知识库上下文文本段落
        """
        lines = []

        # 黑话数据库
        if self._jargon_data:
            lines.append("【已知黑话映射】")
            lines.append("以下正式名称在视频中可能以黑话/代称出现，提取关键词时请识别：")
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
            "黑话数据库": self._jargon_data,
            "白话名称": self._plain_names,
        }

    # ---------- 静态方法：候选提取 ----------

    @staticmethod
    def extract_jargon_candidates(
        keyword_data: dict,
        existing_names: Optional[set[str]] = None,
        existing_jargon_map: Optional[dict[str, str]] = None
    ) -> dict:
        """从关键词提取结果中提取增量黑话候选

        Args:
            keyword_data: KeywordExtractor 返回的关键词数据
            existing_names: 已知正式名称集合（如不提供则为空集合）
            existing_jargon_map: 黑话→正式名称映射（如不提供则为空字典）

        Returns:
            候选字典，格式: {"黑话数据库": {"正式名（推测）": ["黑话候选"]}}
        """
        if existing_names is None:
            existing_names = set()
        if existing_jargon_map is None:
            existing_jargon_map = {}

        candidates: dict = {"黑话数据库": {}}

        for topic, topic_data in keyword_data.items():
            topic_stripped = topic.strip()
            if not topic_stripped:
                continue

            # 检查"个股黑话"字段中的黑话
            if isinstance(topic_data, dict):
                for jargon in topic_data.get("个股黑话", []):
                    jargon = jargon.strip()
                    if not jargon:
                        continue
                    # 如果黑话不在数据库中，且不是已知正式名称
                    if jargon not in existing_jargon_map and jargon not in existing_names:
                        # 以当前板块名称为候选正式名称
                        if topic_stripped not in candidates["黑话数据库"]:
                            candidates["黑话数据库"][topic_stripped] = []
                        if jargon not in candidates["黑话数据库"][topic_stripped]:
                            candidates["黑话数据库"][topic_stripped].append(jargon)

            # 跳过已知名称和黑话
            if topic_stripped in existing_names:
                continue
            if topic_stripped in existing_jargon_map:
                continue

            # 作为黑话候选：主题名本身可能是某正式名称的黑话
            candidates["黑话数据库"][topic_stripped] = []

        logger.info("[黑话候选] 提取到 %d 个黑话候选", len(candidates["黑话数据库"]))
        return candidates


def get_stock_manager() -> JargonStockManager:
    """获取 JargonStockManager 单例的便捷函数

    Returns:
        JargonStockManager 实例
    """
    return JargonStockManager()


# 兼容旧名 get_manager。新代码请使用 get_stock_manager；和 get_sector_manager 命名对称。
def get_manager() -> JargonStockManager:
    """已废弃名：与 get_stock_manager 等价，仅作兜底以避免破坏旧调用点。"""
    return JargonStockManager()
