"""候选剪枝 + 两层压缩（流水线第 4_8 / 4_9 小步，纯 Python 收尾）。

输入：keyword_data（4_7 祖先父生成后已含缺席桶的完整图）+ 候选父板块字典 + 话题分割数组。
逻辑：
1. 反向索引：parent → 候选 child 列表。
2. **候选剪枝（4_8）**：child 从候选父里选唯一父——统一按「共现率」打分，共现率最高者
   当选。共现率 = child 锚点落在候选父讨论区间内的数量 / child 总锚点数；事件父讨论区间
   为事件区间，板块父讨论区间为其锚点所在话题段区间，缺席桶（引用为空）共现率记 0。
   倒挂场景下大板块与事件的共现率天然很低、不会当选，故无需事件父优先或容器保护。
3. **两层压缩（4_9）**：非销毁式挂载——child 不删除、只在唯一父「子项」登记名字；辅助桶
   （4_7 建的缺席桶）折叠删除、子项上移；文本关键词全程保留。环上节点跳过挂载。
   压缩后树深 ≤ 2：根层（事件/板块/标准桶）→ 叶子层（标的）。

引用不在本步 union——根主题的完整引用由下游「引用扩展」递归段并集计算，
避免多父候选图上的 rowID 传染。

**事件不变量**：事件主题永不作为 child（候选父已在 4_2 清空，这里再兜一层），事件恒为顶层。
"""

import logging

from ...log_config import step_logger
from ..schemas import TopicData, TopicSegment
from .event_topics import (
    anchor_ratio_in_ranges,
    coerce_anchors,
    is_segment_event,
    resolve_event_ranges,
)
from .bucket_utils import is_auxiliary

logger = logging.getLogger(__name__)


def _has_self_cycle(child: str, child2parent: dict[str, str]) -> bool:
    """沿 parent 链追溯，检测 child 的祖先链是否回到 child 自身（环）。

    只判「child 是否出现在自己的祖先链上」：回到自身即环、不挂载；链上其它
    与 child 无关的环（如 B↔C 而 A→B）不影响 A 挂到 B，交给 B/C 各自自检。
    """
    visited: set[str] = {child}
    cur = child2parent.get(child)
    while cur is not None:
        if cur == child:
            return True
        if cur in visited:
            return False
        visited.add(cur)
        cur = child2parent.get(cur)
    return False


def _ancestor_chain(node: str, child2parent: dict[str, str]) -> list[str]:
    """node 的祖先链（node -> ... -> 根）。环返回 [node] 兜底。"""
    chain = [node]
    cur = node
    visited: set[str] = {node}
    while cur in child2parent:
        cur = child2parent[cur]
        if cur in visited:
            return [node]
        visited.add(cur)
        chain.append(cur)
    return chain


def _compress_to_two_layers(
    keyword_data: dict[str, TopicData],
    child2parent: dict[str, str],
) -> dict[str, str]:
    """树深压缩：深度（到根距离）> 1 的节点，父改为根（一级行业，深度 0）。

    修 select_root_nodes 后根为一级行业；所有主题（板块/标的/关键词事件）统一
    压到根（chain[-1]），实现「一级行业 -> 主题」两层，只有一级行业生成 section。
    关键词事件压到根后由下游 topic_analyzer 的「关键词事件子项」二分降级为
    一级行业的「事件与资产」。
    """
    compressed = 0
    for child in list(child2parent.keys()):
        chain = _ancestor_chain(child, child2parent)
        if len(chain) - 1 <= 1:
            continue
        new_parent = chain[-1]
        if new_parent != child2parent[child]:
            child2parent[child] = new_parent
            compressed += 1
    if compressed:
        logger.info("树深压缩：%d 个节点上提到根", compressed)
    return child2parent


class ParentRanker:
    """候选剪枝 + 两层压缩（共现率选唯一父，非销毁式挂载压成两层）。"""

    def assign_and_merge(
        self,
        keyword_data: dict[str, TopicData],
        candidates: dict[str, list[str]],
        segments: list[TopicSegment],
    ) -> dict[str, TopicData]:
        with step_logger("第4步-候选剪枝"):
            event_ranges = resolve_event_ranges(keyword_data, segments)
            if not candidates:
                logger.info("候选为空，跳过")
                return keyword_data

            # 第 0 步：反向索引 parent → 候选 child 列表（事件永不作为 child）
            parent2children: dict[str, list[str]] = {}
            event_child_blocked: list[str] = []
            for child, parent_list in candidates.items():
                if not isinstance(parent_list, list):
                    continue
                if is_segment_event(keyword_data.get(child)):
                    # 段落事件恒为顶层：4_2 已清空其候选父，这里兜住模型/回放旧数据
                    if parent_list:
                        event_child_blocked.append(child)
                    continue
                for parent in parent_list:
                    if not isinstance(parent, str) or parent == child:
                        continue
                    lst = parent2children.setdefault(parent, [])
                    if child not in lst:
                        lst.append(child)
            if event_child_blocked:
                logger.warning(
                    "拦下 %d 个事件主题的候选父（事件恒为顶层，不得作为 child）：%s",
                    len(event_child_blocked), event_child_blocked,
                )

            # 候选父的时长代理（共现率并列时的稳定排序）
            parent_durations: dict[str, int] = {}
            for parent in parent2children:
                parent_durations[parent] = self._candidate_duration(
                    keyword_data, parent2children, parent,
                )

            # 每主题锚点 → 所在话题段区间（板块候选父的讨论区间，共现率分母）
            seg_ranges_by_topic = self._build_seg_ranges_by_topic(keyword_data, segments)

            # 第 1 步：候选剪枝——统一按共现率选唯一父（共现率最高者当选）
            child2parent: dict[str, str] = {}
            for child, parent_list in candidates.items():
                if not isinstance(parent_list, list) or not parent_list:
                    continue
                if child not in keyword_data:
                    continue
                if is_segment_event(keyword_data.get(child)):
                    continue
                # 4_7 已建缺席桶，候选父理论上都在 keyword_data；防御性过滤缺席者
                valid_parents = [
                    p for p in parent_list
                    if isinstance(p, str) and p != child and p in keyword_data
                ]
                if not valid_parents:
                    continue
                child_data = keyword_data.get(child)
                child_anchors = (
                    coerce_anchors(child_data.引用 or [])
                    if isinstance(child_data, TopicData) else []
                )
                best = max(
                    valid_parents,
                    key=lambda p: (
                        self._cooccur_ratio(
                            child_anchors, p, event_ranges, seg_ranges_by_topic,
                        ),
                        parent_durations.get(p, 0),
                        -ord(p[0]) if p else 0,
                    ),
                )
                child2parent[child] = best

            logger.info("完成，共 %d 个 child 选定父板块", len(child2parent))

        with step_logger("第4步-两层压缩"):
            keyword_data = self._apply_merges(keyword_data, child2parent)

            # 挂载后的真实结果：事件的直接子项即当选其父的资产
            for ev in sorted(event_ranges):
                evd = keyword_data.get(ev)
                if not isinstance(evd, TopicData):
                    logger.warning("事件「%s」在两层压缩后已不在顶层（不应发生）", ev)
                    continue
                subs = evd.子项 or []
                logger.info(
                    "事件「%s」最终囊括资产 %d 个：%s｜引用锚点 %d 个",
                    ev, len(subs), subs, len(evd.引用 or []),
                )

        return keyword_data

    @staticmethod
    def _build_seg_ranges_by_topic(
        keyword_data: dict[str, TopicData],
        segments: list[TopicSegment],
    ) -> dict[str, list[tuple[int, int]]]:
        """每个主题的锚点 → 所在话题段区间（板块候选父的讨论区间）。"""
        seg_by_row: dict[int, tuple[int, int]] = {}
        for seg in segments:
            s, e = int(seg.开始行号), int(seg.结束行号)
            if s > e:
                s, e = e, s
            for r in range(s, e + 1):
                seg_by_row[r] = (s, e)
        out: dict[str, list[tuple[int, int]]] = {}
        for topic, td in keyword_data.items():
            out[topic] = sorted({
                seg_by_row[a]
                for a in coerce_anchors(td.引用 or [])
                if a in seg_by_row
            })
        return out

    @staticmethod
    def _cooccur_ratio(
        child_anchors: list[int],
        parent: str,
        event_ranges: dict[str, list[tuple[int, int]]],
        seg_ranges_by_topic: dict[str, list[tuple[int, int]]],
    ) -> float:
        """child 与候选父的共现率 = child 锚点落在候选父讨论区间内的占比。

        事件父讨论区间 = 事件区间；板块父讨论区间 = 板块锚点所在话题段区间；
        缺席桶（引用为空）共现率记 0。
        """
        ranges = event_ranges.get(parent) or seg_ranges_by_topic.get(parent) or []
        inside, total = anchor_ratio_in_ranges(child_anchors, ranges)
        return inside / total if total else 0.0

    @staticmethod
    def _candidate_duration(
        keyword_data: dict[str, TopicData],
        parent2children: dict[str, list[str]],
        parent: str,
    ) -> int:
        """候选父的时长代理 = 其候选 child 引用并集大小（共现率并列时稳定排序）。"""
        merged: set[int] = set()
        for child in parent2children.get(parent, []) or []:
            cdata = keyword_data.get(child)
            if isinstance(cdata, TopicData):
                for r in (cdata.引用 or []):
                    try:
                        merged.add(int(r))
                    except (TypeError, ValueError):
                        continue
        return len(merged)

    @staticmethod
    def _apply_merges(
        keyword_data: dict[str, TopicData],
        child2parent: dict[str, str],
    ) -> dict[str, TopicData]:
        """两层压缩：树深压缩 + 非销毁式挂载。

        - 树深压缩：深度 > 1 的非关键词事件节点上提到根（深度 1，先压再挂）。
        - 环检测：child 祖先链回到自身则跳过（无环才挂载），环上节点保留自身锚点。
        - 文本关键词 child：不删除、不 union 引用，只在父「子项」登记名字。
        - 辅助桶 child（4_7 建的缺席桶）：折叠删除，其子项上移给父。
        - 段落事件恒为顶层，永不作为 child。
        """
        # 树深压缩：深度 > 1 的非关键词事件节点上提到根（先压再挂）
        child2parent = _compress_to_two_layers(keyword_data, child2parent)

        mounted = 0
        folded = 0
        for child, parent in child2parent.items():
            cdata = keyword_data.get(child)
            pdata = keyword_data.get(parent)
            if not isinstance(cdata, TopicData) or not isinstance(pdata, TopicData):
                continue
            if child == parent or is_segment_event(cdata):
                continue
            if _has_self_cycle(child, child2parent):
                logger.warning("child2parent 检测到环：'%s' 不挂载", child)
                continue
            psub = list(pdata.子项 or [])
            if child not in psub:
                psub.append(child)
            if is_auxiliary(cdata):
                # 辅助桶折叠：其子项上移给父，节点删除
                for s in cdata.子项 or []:
                    if isinstance(s, str) and s and s not in psub:
                        psub.append(s)
                del keyword_data[child]
                folded += 1
            pdata.子项 = psub
            mounted += 1
        if mounted:
            logger.info("完成，共挂载 %d 个 child（折叠辅助桶 %d 个）", mounted, folded)
        return keyword_data
