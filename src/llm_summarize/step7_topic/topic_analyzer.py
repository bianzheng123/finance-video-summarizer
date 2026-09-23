"""7_1 主题分析。

`[DATA]` 是完整字幕（全阶段共用一份，供前缀缓存命中），每个主题的字幕切片由
`[ROWID]` 表达；LLM 返回后把 `引用` 收敛回该范围（`filter_citations`）。

封装：
- segment_subtitles: 按 keyword_data 行号区间分割字幕，产出各主题的 `[ROWID]` 依据
- generate_one: 单主题 LLM 调用（7_1）+ 引用越界过滤
- drop_empty_topics: 兜底剔除空壳主题
- merge_root_citations / add_citation_duration / inject_jargon: 单主题后处理
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ...log_config import step_logger
from ...llm_infra.batch import run_batch_with_fallback, split_batch_result
from ...llm_infra.client import LLMClient
from ...llm_infra.schema import get_step_output_dir
from ...llm_infra.config import get_batch_size
from ..utils.llm_text_utils import (
    render_annotations,
    render_windowed_subtitle,
    window_row_ids,
)
from ..utils.row_id_2_time_range import (
    convert_subtitle_l2rawID2subtitle_m,
)
from ..utils.rowid_scope import filter_citations, format_scope, scope_id_set
from ...utils.time_utils import hms_to_seconds
from ..analysis_common import inline_array_field, inline_single_field
from ..clustering.event_topics import is_keyword_event
from ..schemas import SubtitleRow, TopicData

_COMMON_DIR = Path(__file__).parent.parent / "analysis_common"

logger = logging.getLogger(__name__)

_FIXED_SECTION_NAMES = {"宏观分析", "交易技巧", "精炼总结", "主讲人暗示重要话题"}

# 观点原子化拆分：分号（；/;）合并两件独立的事 → 拆条
_OPINION_SPLIT_RE = re.compile(r"[；;]")
# 观点/逻辑 归一化："主讲人/主播 + 引出动词" 前缀
_OPINION_INTRO_VERBS: tuple[str, ...] = (
    "认为", "称", "评价", "指出", "表示", "觉得", "回答", "回复",
)
_OPINION_SPEAKER_WORDS: tuple[str, ...] = ("主讲人", "主播", "主讲")
# 语气虚词（比对前两侧同删；排除 了/的/得/地 防"了解"被抠成"解"）
_OPINION_HEDGE_WORDS: tuple[str, ...] = (
    "一直", "比较", "非常", "挺", "蛮", "很", "确实", "肯定", "就是",
    "啊", "嘛", "呀", "呢", "吧",
)
# 问句框架头（如"被问到氧化镝有没有机会时"）：只剥观点侧、且头内含问询语义才剥
_OPINION_MIN_CORE_LEN = 4    # 子串/子序列级判定的核心文本最小长度
_OPINION_MIN_ANCHOR_LEN = 6  # 子序列级兜底要求的最长公共连续片段长度


def _sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "_").replace("\\", "_").replace(":", "_")
        .replace("*", "_").replace("?", "_").replace('"', "_")
        .replace("<", "_").replace(">", "_").replace("|", "_")
    )


def _collect_stock_sub_items(
    topic: str,
    keyword_data: dict[str, TopicData],
    visited: set[str] | None = None,
) -> list[str]:
    """递归收集 topic 子树中所有 类型∈{股票, 个股黑话} 的节点名（含自身）。

    非销毁式树下，个股保留为子节点，根主题（如「科技」）要覆盖其下「半导体」→
    「长鑫存储」等所有收编个股，故按子树递归收集而非只扫直接子项。
    `visited` 防环（树理论无环，残留环的节点各自成根，这里兜底）。
    """
    if visited is None:
        visited = set()
    if topic in visited:
        return []
    visited = visited | {topic}
    td = keyword_data.get(topic)
    if not isinstance(td, TopicData):
        return []
    result: list[str] = []
    if td.类型 in ("股票", "个股黑话"):
        result.append(topic)
    for child in td.子项 or []:
        if isinstance(child, str) and child != topic:
            result.extend(_collect_stock_sub_items(child, keyword_data, visited))
    return result


def _normalize_opinion_core(text: str, strip_question_framing: bool = False) -> str:
    """观点/逻辑文本归一化（立场覆盖去重比对用，两侧同口径）：
    去空白与标点 →（可选，仅观点侧）剥问句框架头 → 剥"主讲人/主播+引出动词"前缀 → 删语气虚词。
    """
    core = re.sub(r"[\s\W_]+", "", text)
    if strip_question_framing:
        last_pos = -1
        for speaker in _OPINION_SPEAKER_WORDS:
            pos = core.rfind(speaker)
            if pos > last_pos:
                last_pos = pos
        if last_pos > 0:
            head = core[:last_pos]
            # 仅当头部是问询语义（被问到/有观众问…）才剥，防误剥"没什么机会"这类评价
            if "问" in head and ("机会" in head or "吗" in head):
                core = core[last_pos:]
    for speaker in _OPINION_SPEAKER_WORDS:
        for verb in _OPINION_INTRO_VERBS:
            prefix = speaker + verb
            if core.startswith(prefix):
                core = core[len(prefix):]
                break
    for hedge in _OPINION_HEDGE_WORDS:
        core = core.replace(hedge, "")
    # 同义反复折叠（"没有改变，一直没有改变"→"没有改变"），ASR 口语重复常见
    while True:
        collapsed = re.sub(r"(.{2,8})\1", r"\1", core)
        if collapsed == core:
            break
        core = collapsed
    return core


def _is_subsequence(core: str, text: str) -> bool:
    """core 是否为 text 的子序列（可跳过中间字符，用于重排措辞兜底）。"""
    it = iter(text)
    return all(ch in it for ch in core)


def _longest_common_run(a: str, b: str) -> int:
    """最长公共连续片段长度（子序列级判定的锚定条件）。"""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    best = 0
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


class TopicAnalyzer:
    """7_1 主题分析。"""

    def __init__(
        self,
        llm_client: LLMClient,
        topic_prompt: str,
        topic_schema: dict,
    ) -> None:
        self._llm_client = llm_client
        self._topic_prompt = topic_prompt
        self._topic_schema = topic_schema
        _jinja_env = Environment(
            loader=FileSystemLoader([str(Path(__file__).parent), str(_COMMON_DIR)])
        )
        _jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._user_template = _jinja_env.get_template("7_1_主题分析_user.jinja2")
        self._batch_user_template = _jinja_env.get_template("7_1_主题分析_batch_user.jinja2")
        # 批量 schema 由单主题 schema 组装：六个内容字段同构 + 主题名键，
        # 杜绝两份 schema 漂移（单主题 schema 改了批量自动跟随）。
        self._batch_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "主题": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "主题名": {"type": "string"},
                            **topic_schema["properties"],
                        },
                        "required": ["主题名", *topic_schema.get("required", [])],
                    },
                },
            },
            "required": ["主题"],
        }

    # ==================== 字幕分割 ====================

    @staticmethod
    def segment_subtitles(
        keyword_data: dict[str, TopicData], subtitle_l: list[SubtitleRow],
    ) -> dict[str, list[SubtitleRow]]:
        """按 keyword_data 各主题的引用行号区间从字幕里抽出对应字幕字典列表。"""
        with step_logger("第7步-主题分析"):
            segments: dict[str, list[SubtitleRow]] = {}

            if not subtitle_l:
                logger.warning("字幕列表为空，无法分割")
                return segments

            rowID2subtitle_m = convert_subtitle_l2rawID2subtitle_m(subtitle_l)
            logger.debug("字幕数据构建了 %d 个有效行号映射", len(rowID2subtitle_m))

            for topic in keyword_data:
                topic_data = keyword_data.get(topic)

                if not isinstance(topic_data, TopicData):
                    logger.warning(
                        "主题「%s」的数据格式异常，跳过: type=%s, value=%s",
                        topic, type(topic_data).__name__, repr(topic_data)[:100],
                    )
                    segments[topic] = []
                    continue

                time_ranges = topic_data.引用
                sub_items = topic_data.子项
                logger.debug(
                    "主题「%s」引用条数=%d, 子项=%s",
                    topic, len(time_ranges), sub_items,
                )

                if not time_ranges:
                    logger.debug("主题「%s」没有时间范围，返回空列表", topic)
                    segments[topic] = []
                    continue

                if not isinstance(time_ranges, list):
                    logger.warning(
                        "主题「%s」的时间范围不是列表，跳过: type=%s, value=%s",
                        topic, type(time_ranges).__name__, repr(time_ranges)[:100],
                    )
                    segments[topic] = []
                    continue

                all_dicts: list[SubtitleRow] = []
                seen_texts: set[str] = set()
                for tr in time_ranges:
                    if not isinstance(tr, dict):
                        logger.warning(
                            "主题「%s」的时间范围元素不是字典，跳过: %s = %s",
                            topic, type(tr).__name__, repr(tr)[:100],
                        )
                        continue
                    start_line = tr.get("开始行号")
                    end_line = tr.get("结束行号")

                    if start_line is None or end_line is None:
                        logger.debug("主题「%s」的时间范围缺少开始行号或结束行号: %s", topic, tr)
                        continue

                    try:
                        start_line_int = int(start_line)
                        end_line_int = int(end_line)
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "主题「%s」的时间范围无法转换为整数: start_line=%s, end_line=%s, 错误: %s",
                            topic, start_line, end_line, e,
                        )
                        continue

                    dicts: list[SubtitleRow] = []
                    for row_id in range(start_line_int, end_line_int + 1):
                        if row_id in rowID2subtitle_m:
                            dicts.append(rowID2subtitle_m[row_id])

                    if not dicts:
                        logger.warning(
                            "主题「%s」的行号范围 [%d-%d] 在字幕中找不到对应的行。字幕行号范围: %s...",
                            topic, start_line_int, end_line_int,
                            list(rowID2subtitle_m.keys())[:5],
                        )

                    for d in dicts:
                        text = d.text
                        if text and text not in seen_texts:
                            seen_texts.add(text)
                            all_dicts.append(d)
                segments[topic] = all_dicts
                logger.debug("主题「%s」分割完成，找到 %d 条字幕", topic, len(all_dicts))

            return segments

    # ==================== 单主题 LLM 调用（7_1）====================

    @staticmethod
    def _build_topic_block(
        topic_name: str,
        topic_row_ids: list[int],
        all_keyword_data: dict[str, TopicData] | None,
    ) -> dict:
        """组装单主题的模板数据块：兄弟主题区间 / 子项白名单 / 关键词事件子项 / 个股清单。

        批量模板的【本批主题清单】与单主题模板共用同一组装逻辑，保证两侧约束一致。
        """
        # 其他主题的引用区间：本主题 [ROWID] 内若含其他主题的完整事件叙事，
        # 不得重复展开（安全网，结构上各主题引用区间已互不相交）。
        # 只收录引用区间与本主题行范围相交的兄弟主题——远处主题与本主题无关，
        # 注入纯属噪声与 token 浪费。
        sibling_topics: dict[str, list[dict]] = {}
        for name, data in (all_keyword_data or {}).items():
            if name == topic_name or not isinstance(data, TopicData):
                continue
            ranges = [
                {"开始行号": int(r["开始行号"]), "结束行号": int(r["结束行号"])}
                for r in (data.引用 or [])
                if isinstance(r, dict)
                and r.get("开始行号") is not None
                and r.get("结束行号") is not None
            ]
            if ranges and not any(
                r["开始行号"] <= rid <= r["结束行号"]
                for r in ranges
                for rid in topic_row_ids
            ):
                continue
            if ranges:
                sibling_topics[name] = ranges

        # 黑话映射已随 [ANNOTATION] 逐行传递（第 3 步的关键词条目），不再注入 [DATABASE]
        # 子项白名单（纯数据，文案在模板里）
        topic_data_for_white = (all_keyword_data or {}).get(topic_name)
        sub_items_for_white = (
            list(topic_data_for_white.子项 or [])
            if isinstance(topic_data_for_white, TopicData) else []
        )

        # 子项二分：关键词事件（已降级为普通子项的事件）从「子项白名单」里拆出，
        # 单独作为「关键词事件子项」交给模板——它们应写入父主题的「事件与资产」，
        # 不参与铁律 A（投资机会标题）、也不单列「投资机会」条目。
        keyword_event_sub_items: list[str] = []
        regular_sub_items: list[str] = []
        if isinstance(all_keyword_data, dict):
            for name in sub_items_for_white:
                if is_keyword_event(all_keyword_data.get(name)):
                    keyword_event_sub_items.append(name)
                else:
                    regular_sub_items.append(name)

        # 个股清单（决定 当前价格 字段输出）——递归收集子树所有 股票/个股黑话 节点，
        # 使根主题总结能涵盖所有收编个股（如「科技」→「半导体」→「长鑫存储」）
        stock_sub_items: list[str] = (
            _collect_stock_sub_items(topic_name, all_keyword_data)
            if isinstance(all_keyword_data, dict) else []
        )

        return {
            "主题名": topic_name,
            "行号范围": format_scope(topic_row_ids),
            "子项白名单": regular_sub_items,
            "关键词事件子项": keyword_event_sub_items,
            "个股清单": stock_sub_items,
            "兄弟主题区间": sibling_topics,
        }

    def generate_one(
        self,
        topic_name: str,
        subtitle_l: list[SubtitleRow],
        topic_row_ids: list[int],
        base_dir: Path,
        all_keyword_data: dict[str, TopicData] | None = None,
        model_override: str | None = None,
        annotation_text: str = "",
        result_sink: dict[str, dict] | None = None,
    ) -> tuple:
        """生成单个主题的 7_1 分析。返回 (topic_name, result)。

        **`[DATA]` 只给本主题的引用扩展范围**（`topic_row_ids` 切片），与 `[ROWID]`
        表达同一范围——本主题的分析依据本就只有这几段，给整场字幕只会让模型在无关
        内容里找依据。返回后仍执行 `filter_citations` 兜底收敛引用。
        """
        safe_topic_name = _sanitize_filename(topic_name)

        # 阶段标签在本方法内自打：本方法会被丢进线程池执行，
        # 提交点的 with 块管不到 worker 线程
        with step_logger("第7_1步-主题分析"):
            block = self._build_topic_block(topic_name, topic_row_ids, all_keyword_data)

            # 阶段规则里的 {topic_name} 占位符替换（{keyword_extraction_result} 已改由模板注入）
            task_rules = self._topic_prompt.replace("{topic_name}", topic_name)

            # [DATA] 给本主题的引用扩展范围 ± N_REDUNDANT_ROWS 冗余上下文；
            # [ROWID] 仍表达工作范围（冗余行使 [DATA] ⊃ 工作范围，须显式声明）
            sliced_subtitle = render_windowed_subtitle(
                subtitle_l, topic_row_ids, with_type=True,
            )
            # [ANNOTATION] 优先用调用方渲染的版本（带 include_keywords=True，黑话映射
            # 靠它逐行传递）；缺省时按 [DATA] 实际行集补渲染一份
            sliced_annotation = annotation_text or render_annotations(
                subtitle_l, include_keywords=True, keywords_only_jargon=True,
                row_scope=window_row_ids(subtitle_l, topic_row_ids),
            )

            # 全部传纯数据，所有【】小节标题与说明文字都在模板里
            user_prompt = self._user_template.render(
                task_rules=task_rules,
                topic_name=topic_name,
                subtitle_text=sliced_subtitle,
                annotation_text=sliced_annotation,
                rowid_scope=format_scope(topic_row_ids),
                sibling_topics=block["兄弟主题区间"],
                sub_items_whitelist=block["子项白名单"],
                keyword_event_sub_items=block["关键词事件子项"],
                stock_sub_items=block["个股清单"],
            )

            log_dir = get_step_output_dir(base_dir, 7) if base_dir is not None else None
            result = self._llm_client.call(
                user_prompt=user_prompt,
                model=model_override or "deepseek-v4-flash",
                temperature=0.0,
                schema=self._topic_schema,
                log_dir=log_dir,
                log_name=f"7_1_主题分析_{safe_topic_name}",
                step_name=f"7_1_主题分析_{topic_name}",
            )

            # [ROWID] 校验：把引用收敛回本主题字幕切片
            filter_citations(
                result, scope_id_set(topic_row_ids), f"7_1 主题「{topic_name}」",
            )
            if result_sink is not None:
                # 存「校验后」结果（filter_citations 原地改 result）
                result_sink[f"7_1_主题分析_{safe_topic_name}"] = result
            return topic_name, result

    # ==================== 批量 LLM 调用（7_1 批量化）====================

    def generate_batch(
        self,
        topic_specs: list[dict],
        subtitle_l: list[SubtitleRow],
        base_dir: Path,
        all_keyword_data: dict[str, TopicData] | None,
        annotation_text: str,
        batch_idx: int,
        model_override: str | None = None,
        result_sink: dict[str, dict] | None = None,
    ) -> tuple[list[tuple[str, dict]], list[tuple[str, Exception]]]:
        """一次 LLM 调用批量生成多个主题的 7_1 分析，返回 (完成列表, 失败列表)。

        topic_specs 每项：{"主题名", "行号"（排序后的 rowID 列表）, "行号集合",
        "注释"（本主题行集渲染的 [ANNOTATION]）}——「注释」只用于整批失败/个别主题
        缺失时拆回 generate_one 单跑，保证回退路径与旧单主题调用一致。

        **[DATA] 按本批主题的引用扩展范围切片**（不再是整场字幕）：7_1 的工作范围
        本就只有这些主题各自的行区间，给整场会让模型在无关内容里找依据、prompt 也
        白白变长。[ANNOTATION] / [ROWID] 同样收敛到这个并集，三者表达同一范围。
        每个主题的行号范围/子项白名单/个股清单/兄弟主题区间随【本批主题清单】逐主题
        给出。返回后逐主题执行 filter_citations（引用收敛回该主题范围）并写 result_sink。
        """
        safe_names = {s["主题名"]: _sanitize_filename(s["主题名"]) for s in topic_specs}
        batch_label = f"7_1 批次 {batch_idx}"
        # 批并集（topic_step 按同批大小预切批，topic_specs 即单子批，
        # 函数级并集 == 模型真实 [ROWID] 作用域）：引用收敛时兄弟主题的行静默跳过
        union_scope: set[int] = set()
        for s in topic_specs:
            union_scope.update(s["行号集合"])

        def run_batch(batch_specs: list[dict]) -> dict[str, dict]:
            names = [s["主题名"] for s in batch_specs]
            blocks = [
                self._build_topic_block(s["主题名"], s["行号"], all_keyword_data)
                for s in batch_specs
            ]
            # [DATA] 给本批主题的引用扩展范围 ± N_REDUNDANT_ROWS 冗余上下文。
            # 拆批回退时 batch_specs ⊂ topic_specs，故按**实际入参**重算范围，
            # 不能用外层 union_scope（那是整批的，回退单跑会给多余字幕）。
            batch_scope: set[int] = set()
            for s in batch_specs:
                batch_scope.update(s["行号集合"])
            sliced_subtitle = render_windowed_subtitle(
                subtitle_l, batch_scope, with_type=True,
            )
            # [ANNOTATION] 用调用方渲染的版本（它带 include_keywords=True——黑话映射
            # 靠注释逐行传递，替代已删除的 [DATABASE] 注入，不能在此重渲染丢掉）
            # 规则文本共享单主题版（{topic_name} → 该主题），批量总则在模板 [TASK] 头部
            task_rules = self._topic_prompt.replace("{topic_name}", "该主题")
            user_prompt = self._batch_user_template.render(
                task_rules=task_rules,
                subtitle_text=sliced_subtitle,
                annotation_text=annotation_text,
                rowid_scope=format_scope(sorted(batch_scope)),
                topics=blocks,
            )
            log_dir = get_step_output_dir(base_dir, 7) if base_dir is not None else None
            result = self._llm_client.call(
                user_prompt=user_prompt,
                model=model_override or "deepseek-v4-flash",
                temperature=0.0,
                schema=self._batch_schema,
                log_dir=log_dir,
                log_name=f"7_1_主题分析_b{batch_idx}",
                step_name=f"7_1_主题分析_batch{batch_idx}",
            )
            mapping, missing = split_batch_result(result, "主题", "主题名", names)
            logger.info(
                "%s：%d 个主题完成%s",
                batch_label, len(mapping),
                f"（缺失 {missing}，转单主题重跑）" if missing else "",
            )
            return mapping

        def run_unit(spec: dict) -> dict:
            return self.generate_one(
                spec["主题名"], subtitle_l, spec["行号"], base_dir,
                all_keyword_data, model_override, spec["注释"], result_sink,
            )[1]

        # 阶段标签在本方法内自打：本方法会被丢进线程池执行，
        # 提交点的 with 块管不到 worker 线程
        with step_logger("第7_1步-主题分析"):
            results, failures = run_batch_with_fallback(
                topic_specs, get_batch_size("7_1_主题分析"),
                unit_key=lambda s: s["主题名"],
                run_batch=run_batch,
                run_unit=run_unit,
                batch_label=batch_label,
            )

            # 逐主题收敛引用并写 sink（拆回后每个主题仍是独立结果单元）
            final_results: list[tuple[str, dict]] = []
            final_failures: list[tuple[str, Exception]] = []
            for spec in topic_specs:
                name = spec["主题名"]
                if name in failures:
                    final_failures.append((name, failures[name]))
                    continue
                result = results[name]
                filter_citations(result, spec["行号集合"], f"7_1 主题「{name}」", union=union_scope)
                if result_sink is not None:
                    result_sink[f"7_1_主题分析_{safe_names[name]}"] = result
                final_results.append((name, result))
            return final_results, final_failures

    # ==================== 兜底：剔除空壳主题 ====================

    @staticmethod
    def drop_empty_topics(analysis: dict) -> dict:
        """剔除 LLM 六个内容字段全部缺失的主题（被 PostProcessor.filter_empty_fields 过滤光后只剩元数据）。"""
        with step_logger("第7步-主题分析"):
            content_keys = ("投资机会", "个人交易记录", "事件与资产", "市场知识", "观点", "其他")
            for topic in list(analysis.keys()):
                if topic in _FIXED_SECTION_NAMES:
                    continue
                topic_data = analysis[topic]
                if not isinstance(topic_data, dict):
                    continue
                if not any(k in topic_data for k in content_keys):
                    logger.info("剔除空壳主题: %s（LLM 未输出任何内容字段）", topic)
                    del analysis[topic]
            return analysis

    # ==================== 单主题：根级引用合并 ====================

    @staticmethod
    def merge_root_citations(
        topic_data: dict, keyword_topic_data: TopicData,
    ) -> dict:
        """把 keyword_data 中该主题的引用时间挂到 topic_data 的 引用时间 字段。"""
        if not isinstance(topic_data, dict) or not isinstance(keyword_topic_data, TopicData):
            return topic_data
        time_ranges = keyword_topic_data.引用
        if isinstance(time_ranges, list):
            topic_data["引用时间"] = [
                {"开始时间": tr.get("开始时间", ""), "结束时间": tr.get("结束时间", "")}
                for tr in time_ranges if isinstance(tr, dict)
            ]
        return topic_data

    # ==================== 单主题：引用总时长 ====================

    @staticmethod
    def add_citation_duration(topic_data: dict) -> dict:
        """给单个主题计算引用总时长（合并重叠区间），写入 引用总时长 字段。"""
        if not isinstance(topic_data, dict):
            return topic_data

        intervals: list[tuple[float, float]] = []

        def _collect(data: Any) -> None:
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in ("引用", "引用时间") and isinstance(value, list):
                        for item in value:
                            if not isinstance(item, dict):
                                continue
                            start_time = item.get("开始时间", "")
                            end_time = item.get("结束时间", "")
                            if not (start_time and end_time):
                                continue
                            try:
                                s = hms_to_seconds(start_time)
                                e = hms_to_seconds(end_time)
                            except Exception:
                                continue
                            if e > s:
                                intervals.append((s, e))
                    elif isinstance(value, (dict, list)):
                        _collect(value)
            elif isinstance(data, list):
                for item in data:
                    _collect(item)

        _collect(topic_data)
        if not intervals:
            topic_data["引用总时长"] = 0.0
            return topic_data

        intervals.sort(key=lambda x: x[0])
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        topic_data["引用总时长"] = float(sum(e - s for s, e in merged))
        return topic_data

    # ==================== 单主题：黑话注入 ====================

    @staticmethod
    def inject_jargon(topic_data: dict, used_jargons: dict[str, list[str]]) -> dict:
        """把字幕里实际出现过的黑话挂到主题各资产名字段：
        - 投资机会[].标题
        - 市场知识[].标题
        - 其他[].标题
        - 个人交易记录[].标的名称
        - 事件与资产[].标题 + 涉及资产 数组
        """
        if not isinstance(topic_data, dict) or not used_jargons:
            return topic_data
        for entry in topic_data.get("投资机会", []) or []:
            inline_single_field(entry, "标题", used_jargons)
        for entry in topic_data.get("市场知识", []) or []:
            inline_single_field(entry, "标题", used_jargons)
        for entry in topic_data.get("其他", []) or []:
            inline_single_field(entry, "标题", used_jargons)
        for entry in topic_data.get("个人交易记录", []) or []:
            inline_single_field(entry, "标的名称", used_jargons)
        for entry in topic_data.get("观点", []) or []:
            inline_single_field(entry, "标的名称", used_jargons)
        for entry in topic_data.get("事件与资产", []) or []:
            inline_single_field(entry, "标题", used_jargons)
            inline_array_field(entry, "涉及资产", used_jargons)
        return topic_data

    # ==================== 单主题：观点原子化拆分与立场覆盖去重 ====================

    @staticmethod
    def split_opinion_entries(topic_data: dict) -> dict:
        """观点原子化兜底（确定性）：按分号（；/;）把观点文本拆为独立条目，并折叠重复观点。

        每条拆分结果复制原 `引用`（超集交由 7_2 逐条校验）；`因果链` 保留给第一条、
        其余置 []（论证链无法按半句可靠切分）。空片段丢弃；归一化相同的片段折叠
        （去重）。无分号的整条观点同样按归一化核心文本判重：
        逐字相同的独立条目只保留第一条（LLM 把同一句评价重复输出两次时兜底折叠）。
        **去重粒度 =（标的名称 + 观点核心）**：同一标的的重复观点仍折叠；不同标的的
        相同观点保留——不同标的的观点各自独立，不跨标的去重。
        原地修改 topic_data["观点"] 并返回 topic_data。
        """
        if not isinstance(topic_data, dict):
            return topic_data
        opinions = topic_data.get("观点")
        if not isinstance(opinions, list):
            return topic_data

        new_opinions: list[dict] = []
        emitted: set[tuple[str, str]] = set()
        for entry in opinions:
            if not isinstance(entry, dict):
                new_opinions.append(entry)
                continue
            text = entry.get("观点")
            if not isinstance(text, str) or not text:
                new_opinions.append(entry)
                continue
            op_name = str(entry.get("标的名称") or "").strip()
            parts = [p.strip() for p in _OPINION_SPLIT_RE.split(text)]
            parts = [p for p in parts if p]
            if len(parts) <= 1:
                # 无分号整条：同样按（标的名称 + 归一化核心文本）判重
                core = _normalize_opinion_core(text)
                if (op_name, core) in emitted:
                    logger.warning("观点原子化去重：剔除重复观点：%s", text)
                    continue
                emitted.add((op_name, core))
                new_opinions.append(entry)
                continue
            chain = entry.get("因果链")
            citations = entry.get("引用")
            for i, part in enumerate(parts):
                core = _normalize_opinion_core(part)
                if (op_name, core) in emitted:
                    continue
                emitted.add((op_name, core))
                piece: dict = dict(entry)
                piece["观点"] = part
                piece["引用"] = list(citations) if isinstance(citations, list) else []
                piece["因果链"] = list(chain) if (i == 0 and isinstance(chain, list)) else []
                new_opinions.append(piece)
        topic_data["观点"] = new_opinions
        return topic_data

    @staticmethod
    def dedup_redundant_opinions(topic_data: dict) -> dict:
        """立场覆盖去重兜底（确定性）：剔除内容已被同标的 投资机会 立场逻辑完整覆盖的观点条目。

        名称匹配与渲染层速览规则同源：标的名称 vs 标题 双向子串包含。
        仅比较立场非"未表态"的 短线/中长线 `逻辑`——未表态立场行不渲染，
        被其覆盖的观点是唯一载体，必须保留。
        剔除判定（核心 ≥4 字才参与）：核心是逻辑核心的子串 → 剔除；或核心是逻辑
        核心的子序列且两者最长公共连续片段 ≥6 字（重排措辞兜底）→ 剔除。
        反向不剔（观点含额外内容时保留）。原地修改 topic_data["观点"] 并返回 topic_data。
        """
        if not isinstance(topic_data, dict):
            return topic_data
        opinions = topic_data.get("观点")
        if not isinstance(opinions, list) or not opinions:
            return topic_data
        opps = topic_data.get("投资机会")
        if not isinstance(opps, list) or not opps:
            return topic_data

        def _stance_logics(inv: dict) -> list[str]:
            """收集该投资机会条目中立场非未表态的时间维度逻辑（原始文本）。"""
            logics: list[str] = []
            for key in ("短线", "中长线"):
                sub = inv.get(key)
                if not isinstance(sub, dict):
                    continue
                stance = sub.get("观点")
                if not stance or "未表态" in str(stance):
                    continue
                logic = sub.get("逻辑")
                if isinstance(logic, str) and logic.strip():
                    logics.append(logic)
            return logics

        kept: list[dict] = []
        for entry in opinions:
            if not isinstance(entry, dict):
                kept.append(entry)
                continue
            op_name = str(entry.get("标的名称") or "").strip()
            text = entry.get("观点")
            if not op_name or not isinstance(text, str) or not text.strip():
                kept.append(entry)
                continue
            core = _normalize_opinion_core(text, strip_question_framing=True)
            dropped = False
            if len(core) >= _OPINION_MIN_CORE_LEN:
                for inv in opps:
                    if not isinstance(inv, dict):
                        continue
                    title = str(inv.get("标题") or "").strip()
                    if not title or not (op_name in title or title in op_name):
                        continue
                    for logic in _stance_logics(inv):
                        logic_core = _normalize_opinion_core(logic)
                        if core in logic_core:
                            dropped = True
                            break
                        if (
                            _is_subsequence(core, logic_core)
                            and _longest_common_run(core, logic_core) >= _OPINION_MIN_ANCHOR_LEN
                        ):
                            dropped = True
                            break
                    if dropped:
                        break
            if dropped:
                logger.warning("观点立场覆盖去重：剔除「%s」观点条目：%s", op_name, text)
                continue
            kept.append(entry)
        topic_data["观点"] = kept
        return topic_data

    @staticmethod
    def dedup_bare_conclusion_opinions(topic_data: dict) -> dict:
        """论证完整性去重兜底（确定性）：剔除「只有结论、无因果链」的裸结论观点。

        生成时 LLM 可能把一句连贯评价（大前提 → 结论）拆成两条：一条带完整
        `因果链`（含「结论」环节），另一条只有结论文本、`因果链=[]`。本函数把
        后者（裸结论）删除，只保留带完整因果链的那条——避免同一结论在报告中
        出现两次（一次带前提、一次裸结论）。

        判定（同一标的下、单向）：某条观点 `因果链=[]` 且其 `观点` 文本归一化后
        与同标的另一条观点 `因果链` 中「结论」环节内容归一化后 相等 / 子串 /
        （子序列 + 最长公共连续片段 ≥ 锚定长度）→ 剔除。只删裸结论，不反向删
        带因果链的那条。原地修改 topic_data["观点"] 并返回 topic_data。
        """
        if not isinstance(topic_data, dict):
            return topic_data
        opinions = topic_data.get("观点")
        if not isinstance(opinions, list) or len(opinions) < 2:
            return topic_data

        # 1) 收集同一标的下所有「结论」环节的归一化文本
        conclusions_by_name: dict[str, list[str]] = {}
        for entry in opinions:
            if not isinstance(entry, dict):
                continue
            chain = entry.get("因果链")
            if not isinstance(chain, list):
                continue
            for link in chain:
                if not isinstance(link, dict):
                    continue
                if link.get("环节类型") != "结论":
                    continue
                content = link.get("内容")
                if not isinstance(content, str) or not content.strip():
                    continue
                core = _normalize_opinion_core(content)
                if len(core) < _OPINION_MIN_CORE_LEN:
                    continue
                name = str(entry.get("标的名称") or "").strip()
                conclusions_by_name.setdefault(name, []).append(core)

        if not conclusions_by_name:
            return topic_data

        # 2) 剔除「因果链为空且文本命中某条结论」的裸结论观点
        kept: list[dict] = []
        for entry in opinions:
            if not isinstance(entry, dict):
                kept.append(entry)
                continue
            chain = entry.get("因果链")
            if isinstance(chain, list) and chain:
                kept.append(entry)  # 带因果链的保留
                continue
            text = entry.get("观点")
            name = str(entry.get("标的名称") or "").strip()
            if not name or not isinstance(text, str) or not text.strip():
                kept.append(entry)
                continue
            core = _normalize_opinion_core(text, strip_question_framing=True)
            if len(core) < _OPINION_MIN_CORE_LEN:
                kept.append(entry)
                continue
            dropped = False
            for conclusion_core in conclusions_by_name.get(name, []):
                if (
                    core == conclusion_core
                    or core in conclusion_core
                    or (
                        _is_subsequence(core, conclusion_core)
                        and _longest_common_run(core, conclusion_core) >= _OPINION_MIN_ANCHOR_LEN
                    )
                ):
                    dropped = True
                    break
            if dropped:
                logger.warning(
                    "观点论证完整性去重：剔除「%s」裸结论观点：%s", name, text,
                )
                continue
            kept.append(entry)
        topic_data["观点"] = kept
        return topic_data
