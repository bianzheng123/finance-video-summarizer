"""知识库模块 - 黑话个股 / 黑话板块映射，以及统一热词管理"""

from .jargon_stock_manager import JargonStockManager, get_manager, get_stock_manager
from .jargon_sector_manager import JargonSectorManager, get_sector_manager
from .unified_vocab_manager import UnifiedVocabManager, HotwordFormat


__all__ = [
    "UnifiedVocabManager",
    "HotwordFormat",
    "JargonStockManager",
    "JargonSectorManager",
    "get_manager",  # deprecated alias，等同 get_stock_manager
    "get_stock_manager",
    "get_sector_manager",
]
