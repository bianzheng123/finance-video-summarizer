"""事件热点缓存管理器（单例模式）

管理 `jargon_hotspot.json`，里面有两类数据：
- `事件`：地缘 / 政策 / 行业事件全档，键为事件唯一 ID，每条事件含正式名、进展（各时间段关键时间点）、字幕线索词、黑话别名等
- `人物地名简称`：原 `{别名: [正式名]}` 形态，供 ASR 热词使用

本管理器面向关键词消歧管线，提供：
- 别名直命中 / 字幕线索词匹配 → 跳过 WSA 联网消歧
- 缓存命中后刷新「最近提及」（原子写）
"""

import json
import logging
import os
import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 「需线索」事件缓存命中的最小线索重叠数（字幕线索词集合交集阈值）
MIN_CLUE_OVERLAP = 2


class JargonHotspotManager:
    """事件热点缓存管理器（单例模式）"""

    _instance = None
    _initialized = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "JargonHotspotManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[Path] = None) -> None:
        if self._initialized:
            return

        self.config_path = config_path or (
            Path(__file__).parent.parent.parent / "database" / "jargon_hotspot.json"
        )

        self._events: dict[str, dict] = {}
        self._person_place: dict[str, list[str]] = {}

        # 按话题并行消歧后，touch_recent 会被多线程并发调用，
        # 锁保护"读-改-写"的 in-memory 字典（文件侧由 _atomic_write 保证）。
        self._write_lock = threading.Lock()

        self._load()
        self._initialized = True

    def _load(self) -> None:
        if not self.config_path.exists():
            logger.info("[事件热点库] 配置文件不存在，初始化为空: %s", self.config_path)
            return

        with open(self.config_path, encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            logger.warning("[事件热点库] 顶层结构不是 dict，降级为空库")
            return

        events = raw.get("事件", {})
        if isinstance(events, dict):
            self._events = events
        person_place = raw.get("人物地名简称", {})
        if isinstance(person_place, dict):
            self._person_place = person_place

        logger.info(
            "[事件热点库] 加载成功，事件 %d 条，人物地名简称 %d 条",
            len(self._events), len(self._person_place),
        )

    # ===== 查询接口 =====

    def _snapshot_events(self) -> dict[str, dict]:
        """并发安全地复制事件表快照（读侧与写侧跨线程共享）。"""
        with self._write_lock:
            return dict(self._events)

    def get_alias_to_event(self) -> dict[str, str]:
        """{黑话别名: event_key}，用于别名直命中。同一别名出现在多条事件时取最近提及最新者"""
        alias_map: dict[str, tuple[str, str]] = {}
        for event_key, event in self._snapshot_events().items():
            if not isinstance(event, dict):
                continue
            recent = event.get("最近提及", "") or ""
            for alias in event.get("黑话别名", []) or []:
                if not isinstance(alias, str):
                    continue
                prev = alias_map.get(alias)
                if prev is None or recent > prev[1]:
                    alias_map[alias] = (event_key, recent)
        return {alias: ek for alias, (ek, _) in alias_map.items()}

    def get_event(self, event_key: str) -> dict | None:
        with self._write_lock:
            event = self._events.get(event_key)
            return dict(event) if isinstance(event, dict) else None

    def all_events(self) -> dict[str, dict]:
        return self._snapshot_events()

    def match_by_clues(
        self, alias: str, subtitle_clues: set[str], min_overlap: int = MIN_CLUE_OVERLAP,
    ) -> str | None:
        """字幕线索词命中：alias 必须 ∈ event['黑话别名']
        且 |subtitle_clues ∩ event['字幕线索词']| >= min_overlap → 返回 event_key
        """
        best_key: str | None = None
        best_score = -1
        for event_key, event in self._snapshot_events().items():
            if not isinstance(event, dict):
                continue
            aliases = event.get("黑话别名", []) or []
            if alias not in aliases:
                continue
            event_clues = set(event.get("字幕线索词", []) or [])
            overlap = len(subtitle_clues & event_clues)
            if overlap >= min_overlap and overlap > best_score:
                best_score = overlap
                best_key = event_key
        return best_key

    def requires_clues(self, event_key: str) -> bool:
        """事件是否开启「需线索」门控（缺省 False：维持旧事件别名直命中行为）。"""
        event = self._snapshot_events().get(event_key)
        return bool(isinstance(event, dict) and event.get("需线索") is True)

    def event_clue_overlap(self, event_key: str, subtitle_clues: set[str]) -> int:
        """subtitle_clues 与指定事件「字幕线索词」的交集大小（英文大小写不敏感）。"""
        event = self._snapshot_events().get(event_key)
        if not isinstance(event, dict):
            return 0
        clue_words = set(event.get("字幕线索词", []) or [])
        if not clue_words:
            return 0
        norm = {c.upper() for c in subtitle_clues}
        return sum(1 for w in clue_words if str(w).upper() in norm)

    def match_by_substring(
        self,
        word: str,
        subtitle_clues: set[str],
        min_overlap: int = MIN_CLUE_OVERLAP,
    ) -> str | None:
        """「需线索」事件的子串别名匹配：word 含某 ≥2 字别名（如 电梯PCB ⊃ 电梯）
        时过线索门控；单字别名不参与子串匹配（防 尔/霍 误伤）；多事件候选取
        overlap 最高，平手取 JSON 顺序更靠前者。非「需线索」事件不参与。
        """
        best_key: str | None = None
        best_score = -1
        norm = {c.upper() for c in subtitle_clues}
        for event_key, event in self._snapshot_events().items():
            if not isinstance(event, dict):
                continue
            if event.get("需线索") is not True:
                continue
            clue_words = set(event.get("字幕线索词", []) or [])
            for alias in event.get("黑话别名", []) or []:
                if not isinstance(alias, str) or len(alias) < 2:
                    continue
                if alias not in word:
                    continue
                overlap = sum(1 for w in clue_words if str(w).upper() in norm)
                if overlap >= min_overlap and overlap > best_score:
                    best_score = overlap
                    best_key = event_key
                break  # 同一事件多别名命中时重叠度相同，无需重复计算
        return best_key

    def get_person_place_map(self) -> dict[str, list[str]]:
        """供 update_vocab.py 使用"""
        return dict(self._person_place)

    # ===== 写入接口 =====

    def touch_recent(self, event_keys: list[str]) -> None:
        """把指定事件的"最近提及"刷新为今天，便于退化策略"""
        if not event_keys:
            return
        with self._write_lock:
            today = date.today().isoformat()
            changed = False
            for ek in event_keys:
                event = self._events.get(ek)
                if not isinstance(event, dict):
                    continue
                if event.get("最近提及") != today:
                    event["最近提及"] = today
                    changed = True
            if changed:
                self._atomic_write()

    def _atomic_write(self) -> None:
        """tempfile + os.replace 原子写，避免并发污染"""
        payload = {
            "事件": self._events,
            "人物地名简称": self._person_place,
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".jargon_hotspot_", suffix=".json.tmp",
            dir=str(self.config_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.config_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


def get_hotspot_manager() -> JargonHotspotManager:
    """获取 JargonHotspotManager 单例的便捷函数"""
    return JargonHotspotManager()
