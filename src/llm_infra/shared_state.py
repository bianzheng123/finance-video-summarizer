"""跨 batch 共享状态存储（线程安全）：黑话互证 + 同字词纠错提示，两块合并在本模块。

两个表同源设计，都服务于「词级三阶段按 batch 并行、batch 内 2_1→2_2→2_3 串行」这一
执行模型下，让后完成的 batch 读到先完成 batch 的已确认结论：

- **SharedFindings（黑话互证）**：2_2/2_3 构造 [DATABASE] 时读**当前已完成**的全部
  batch 的发现（如「达链」在别的话题已确认为英伟达供应链），不阻塞等待——不牺牲并行度，
  读到多少取决于时序。
- **SharedCorrectionHints（同字词纠错提示）**：把「某 batch 已确认的错词」在**其他批次**
  的出现位置登记下来，作为提示注入后续处理这些行的 LLM 调用，由 LLM 自行判断是否同样
  修改——取代旧「代码替 LLM 拦截 / 自动扩展」的逻辑。

共同属性：
- 线程安全：batch 间并行走全局线程池，共享本表；
- 无栅栏：先完成的发布能被后完成的读到，读到多少取决于时序（不牺牲并行度）；
- 回放确定性：`LLM_REPLAY` 下读冻结集（`freeze()` / `load_frozen()`），保证同一份日志
  回放出同一结果。
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)


def _replay_enabled() -> bool:
    """LLM_REPLAY 是否开启（每次读 env，不做 import 快照）。"""
    return os.environ.get("LLM_REPLAY", "").strip().lower() in ("1", "true", "yes")


class SharedFindings:
    """进程内共享「黑话互证」表：`{rowID: {词: 条目}}`，线程安全。

    条目形态与注释的关键词条目一致（`词`/`正式名`/`类型`/`依据`/`困惑`/`困惑原因`）。
    由 `confirmed_map` 转成 `{别名: [正式名]}`（与词表 `jargon_map` 同构）供
    [DATABASE] 渲染。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_row: dict[int, dict[str, dict]] = {}
        self._frozen: dict[int, dict[str, dict]] | None = None

    def add_many(self, entries: list[dict]) -> int:
        """并入一批已确认发现（同 (rowID, 词) 后到覆盖先到）。返回并入条数。"""
        n = 0
        with self._lock:
            for e in entries:
                row_id = e.get("rowID")
                word = (e.get("词") or "").strip()
                if row_id is None or not word:
                    continue
                self._by_row.setdefault(int(row_id), {})[word] = dict(e)
                n += 1
        return n

    def snapshot(self) -> dict[int, dict[str, dict]]:
        """当前已确认发现的快照。

        回放模式下若已 `load_frozen`，返回冻结集（确定性）；否则返回实时集。
        """
        if _replay_enabled() and self._frozen is not None:
            return {rid: dict(words) for rid, words in self._frozen.items()}
        with self._lock:
            return {rid: dict(words) for rid, words in self._by_row.items()}

    def confirmed_map(self, row_scope: set[int] | None = None) -> dict[str, list[str]]:
        """转成 `{别名: [正式名]}`（与 `jargon_map` 同构），供 [DATABASE] 渲染。

        从 snapshot 提取去重的「词 → 正式名」，过滤掉 `正式名 == 词` 的字面条目，
        并按 `row_scope` 过滤行（None 取全场）。
        """
        out: dict[str, list[str]] = {}
        for row_id, words in self.snapshot().items():
            if row_scope is not None and row_id not in row_scope:
                continue
            for word, e in words.items():
                formal = (e.get("正式名") or "").strip()
                if not formal or formal == word:
                    continue
                out.setdefault(word, [])
                if formal not in out[word]:
                    out[word].append(formal)
        return out

    def freeze(self) -> dict[str, list[dict]]:
        """导出可落盘的发现集（检查点用；键为 rowID 字符串以便 JSON 往返）。"""
        return {
            str(rid): list(words.values())
            for rid, words in self.snapshot().items()
        }

    def load_frozen(self, data: dict | None) -> None:
        """载入落盘的发现集作为回放基准（仅 LLM_REPLAY 生效）。"""
        if not isinstance(data, dict):
            return
        frozen: dict[int, dict[str, dict]] = {}
        for rid_str, entries in data.items():
            try:
                rid = int(rid_str)
            except (TypeError, ValueError):
                continue
            if not isinstance(entries, list):
                continue
            frozen[rid] = {
                (e.get("词") or "").strip(): dict(e)
                for e in entries
                if isinstance(e, dict) and (e.get("词") or "").strip()
            }
        self._frozen = frozen
        logger.info("[共享发现] 载入回放基准：%d 行", len(frozen))

    def clear(self) -> None:
        """清空（每个视频独立一份，跨视频不复用）。"""
        with self._lock:
            self._by_row.clear()
        self._frozen = None


class SharedCorrectionHints:
    """进程内共享「同字词纠错」提示表：`{原词: {修改的词, 来源阶段, 出现行号}}`。

    `原词` 是错词列表的 `原先的词`（尚未修改的原文子串），`修改的词` 是确证的
    正确写法，`出现行号` 是该原词在原始字幕中**来源批次之外**出现的行号（来源
    批次自身的出现已由该批次自行应用，不登记）；`_consumed` 记录已作为提示分发
    出去的行号（消费去重）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hints: dict[str, dict] = {}
        self._consumed: dict[str, set[int]] = {}
        self._frozen: dict[str, dict] | None = None

    def publish(
        self,
        wrong_words: list[dict],
        text_by_id: dict[int, str],
        stage: str,
        source_rows: set[int] | None = None,
    ) -> int:
        """LLM 输出错词后：对每个原词全文搜索出现行号，登记为待提示。

        只登记**来源批次之外**的行号（跨片段语义）：错词在来源批次自身的出现
        已由该批次串行链（2_1→2_2→2_3）自行应用，无需再提示；排除后无剩余
        出现行号时整条不登记——避免「只在本批出现的错词」被当作跨片段提示注入。

        Args:
            wrong_words: 错词列表条目 `{rowID, 原先的词, 修改的词}`。
            text_by_id: 原始字幕 `{rowID: text}`（未修改；错词锚点必须在原文存在）。
            stage: 来源阶段标识（"2_1" / "2_2" / "2_3"），仅用于提示文案标注。
            source_rows: 来源批次负责的行号集合；这些行的出现不登记（同批已自行修正）。

        Returns:
            实际登记的提示条数（按原词去重后，同一原词后到覆盖先到）。
        """
        n = 0
        source_rows = source_rows or set()
        with self._lock:
            for w in wrong_words or []:
                word = (w.get("原先的词") or "").strip()
                fixed = (w.get("修改的词") or "").strip()
                if not word or not fixed:
                    continue
                row_ids = {
                    rid for rid, text in text_by_id.items()
                    if word in text and rid not in source_rows
                }
                if not row_ids:
                    continue
                entry = self._hints.get(word)
                if entry is None:
                    self._hints[word] = {
                        "修改的词": fixed,
                        "来源阶段": stage,
                        "出现行号": row_ids,
                    }
                    n += 1
                else:
                    entry["修改的词"] = fixed
                    entry["来源阶段"] = stage
                    entry["出现行号"] |= row_ids
        if n:
            logger.info(
                "[同字词纠错] 发布 %d 条错词提示（来源 %s）", n, stage,
            )
        return n

    def hints_for(
        self,
        row_scope: set[int],
        current_text_by_id: dict[int, str] | None = None,
    ) -> str:
        """取出覆盖 `row_scope` 的未消费提示，渲染成文本，并把这些行标记为已消费。

        只提示**在本批当前文本中仍出现、且未被改过**的行：错词在本批当前
        [DATA] 里已经是改后的写法（`word not in current_text_by_id[row]`）时
        不再提示——那说明它已被本批（或他批）修正过了，再提示就是无关信息。

        Args:
            row_scope: 本批工作行号集合（只提示落在本批范围内的出现行）。
            current_text_by_id: 本批当前 [DATA] 文本 `{rowID: text}`（已应用本批
                前序阶段的确定性修正）。None 时不做「仍出现」过滤。

        Returns:
            带小标题的提示文本（无提示时返回空串，模板据此整块省略）。
        """
        if _replay_enabled() and self._frozen is not None:
            return self._render(self._frozen, row_scope, current_text_by_id)
        lines: list[str] = []
        with self._lock:
            for word, entry in sorted(self._hints.items()):
                pending = entry["出现行号"] - self._consumed.get(word, set())
                rows = pending & row_scope
                if not rows:
                    continue
                if current_text_by_id is not None:
                    rows = {
                        r for r in rows if word in current_text_by_id.get(r, "")
                    }
                    if not rows:
                        continue
                self._consumed.setdefault(word, set()).update(rows)
                lines.append(self._line(word, entry, rows))
        if not lines:
            return ""
        if not _replay_enabled():
            logger.info("[同字词纠错] 本批消费 %d 条提示", len(lines))
        return self._wrap(lines)

    def freeze(self) -> dict:
        """导出可落盘的提示集（检查点用；键为原词字符串，行号转 list 以便 JSON 往返）。"""
        with self._lock:
            return {
                word: {
                    "修改的词": entry["修改的词"],
                    "来源阶段": entry["来源阶段"],
                    "出现行号": sorted(entry["出现行号"]),
                }
                for word, entry in self._hints.items()
            }

    def load_frozen(self, data: dict | None) -> None:
        """载入落盘的提示集作为回放基准（仅 LLM_REPLAY 生效）。"""
        if not isinstance(data, dict):
            return
        frozen: dict[str, dict] = {}
        for word, entry in data.items():
            if not isinstance(word, str) or not isinstance(entry, dict):
                continue
            rows = entry.get("出现行号")
            if not isinstance(rows, list):
                continue
            frozen[word] = {
                "修改的词": entry.get("修改的词") or "",
                "来源阶段": entry.get("来源阶段") or "",
                "出现行号": {int(r) for r in rows if isinstance(r, (int, float))},
            }
        self._frozen = frozen
        logger.info("[同字词纠错] 载入回放基准：%d 条", len(frozen))

    def clear(self) -> None:
        """清空（每个视频独立一份，跨视频不复用）。"""
        with self._lock:
            self._hints.clear()
            self._consumed.clear()
        self._frozen = None

    @staticmethod
    def _line(word: str, entry: dict, rows: set[int]) -> str:
        rid_list = ", ".join(str(r) for r in sorted(rows))
        return (
            f"- 「{word}」→「{entry['修改的词']}」（来源 {entry['来源阶段']}），"
            f"本批出现行：{rid_list}"
        )

    @staticmethod
    def _wrap(lines: list[str]) -> str:
        return (
            "【跨片段同字词纠错提示】\n"
            "以下原词已在其他批次被确认为错字并给出正确写法；这些原词在本批当前"
            "字幕中仍出现、且尚未修改，请参考已确认写法，自行判断本批这些位置是否"
            "同样修改（由你决定，代码不强制、不拦截）：\n" + "\n".join(lines)
        )

    @classmethod
    def _render(
        cls,
        hints: dict[str, dict],
        row_scope: set[int],
        current_text_by_id: dict[int, str] | None = None,
    ) -> str:
        """按冻结集渲染（回放模式，不消费）。"""
        lines: list[str] = []
        for word, entry in sorted(hints.items()):
            rows = entry["出现行号"] & row_scope
            if current_text_by_id is not None:
                rows = {r for r in rows if word in current_text_by_id.get(r, "")}
            if rows:
                lines.append(cls._line(word, entry, rows))
        return cls._wrap(lines) if lines else ""
