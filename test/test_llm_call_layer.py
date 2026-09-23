"""LLM 调用层单元测试（全程 mock，不触网）。

覆盖三块容易静默出错、且出错后症状很隐蔽的逻辑：

1. `ReasoningChatOpenAI` —— DeepSeek 的 `reasoning_content` 不在 OpenAI 标准字段里，
   langchain-openai 0.3.11 的 Chat Completions 路径会把它丢掉。曾因此导致 122 个
   历史日志的「思考」字段全为空。这里锁定透传行为，防回归。
2. `conversation_log_writer` —— 尝试文件与概览的落盘格式，以及中文不得被转成 \\uXXXX。
3. `retry_policy` —— 过载类错误无限重试、schema 校验失败有限次数。分流错了会导致
   高峰期整轮分析被瞬时 503 打断，或模型稳定输出坏结构时无限打转。
"""

import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, AIMessageChunk

from src.llm_infra.client import (
    _as_strict_tools,
    _messages_to_log_list,
    load_shared_system_prompt,
)
from src.llm_infra.config import effective_config, resolve_profile
from src.llm_infra.io import (
    write_attempt_log,
    write_overview_log,
)
from src.llm_infra.model import ReasoningChatOpenAI
from src.llm_infra.retry import (
    OPENAI_MAX_RETRIES,
    build_retrying,
)
from src.llm_infra.tools import WEB_SEARCH_TOOL


# ═══════════════════ reasoning_content 透传 ═══════════════════

def _fake_response(content: str, reasoning: str | None = None,
                   usage: dict | None = None) -> dict:
    """构造一份 DeepSeek 风格的非流式响应体。"""
    msg: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    resp: dict = {
        "id": "x", "model": "deepseek-v4-flash", "object": "chat.completion",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
    }
    if usage:
        resp["usage"] = usage
    return resp


class TestReasoningPassthrough:
    """曾经的真实 bug：reasoning 被 LangChain 丢弃，导致日志「思考」永远为空。"""

    @pytest.fixture
    def llm(self) -> ReasoningChatOpenAI:
        return ReasoningChatOpenAI(
            model="deepseek-v4-flash", api_key="dummy",
            base_url="https://api.deepseek.com",
        )

    def test_reasoning_lands_in_additional_kwargs(self, llm: ReasoningChatOpenAI) -> None:
        result = llm._create_chat_result(
            _fake_response("答案", reasoning="我先想了想…"),
        )
        msg = result.generations[0].message
        assert msg.additional_kwargs["reasoning_content"] == "我先想了想…"

    def test_absent_reasoning_does_not_inject_key(self, llm: ReasoningChatOpenAI) -> None:
        result = llm._create_chat_result(_fake_response("答案"))
        assert "reasoning_content" not in result.generations[0].message.additional_kwargs

    def test_empty_reasoning_not_injected(self, llm: ReasoningChatOpenAI) -> None:
        """空字符串按"没有"处理，避免日志里出现无意义的空思考。"""
        result = llm._create_chat_result(_fake_response("答案", reasoning=""))
        assert "reasoning_content" not in result.generations[0].message.additional_kwargs

    def test_content_still_intact(self, llm: ReasoningChatOpenAI) -> None:
        result = llm._create_chat_result(_fake_response("正文内容", reasoning="思考"))
        assert result.generations[0].message.content == "正文内容"

    def test_usage_extras_exposed(self, llm: ReasoningChatOpenAI) -> None:
        """reasoning_tokens 与前缀缓存命中数要透出，供成本与缓存率监控。"""
        result = llm._create_chat_result(_fake_response(
            "答案", reasoning="思考",
            usage={
                "prompt_tokens": 100, "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 42},
                "prompt_cache_hit_tokens": 64,
                "prompt_cache_miss_tokens": 36,
            },
        ))
        u = result.generations[0].message.additional_kwargs["deepseek_usage"]
        assert u["reasoning_tokens"] == 42
        assert u["prompt_cache_hit_tokens"] == 64
        assert u["prompt_cache_miss_tokens"] == 36

    def test_usage_absent_no_key(self, llm: ReasoningChatOpenAI) -> None:
        result = llm._create_chat_result(_fake_response("答案"))
        assert "deepseek_usage" not in result.generations[0].message.additional_kwargs

    def test_streaming_chunk_reasoning_passthrough(self, llm: ReasoningChatOpenAI) -> None:
        """流式路径当前未启用，但保留实现——若将来回退到流式，这条防止悄悄失效。

        default_chunk_class 必须传 **Chunk 类**（AIMessageChunk）而非 AIMessage：
        ChatGenerationChunk.message 的类型约束是 BaseMessageChunk，传 AIMessage 会被
        pydantic 拒掉。生产路径（langchain_openai base.py）传的也是 chunk 类。
        """
        chunk = {
            "choices": [{
                "index": 0, "delta": {"reasoning_content": "增量思考"},
                "finish_reason": None,
            }],
        }
        gen = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)
        assert gen is not None
        assert gen.message.additional_kwargs.get("reasoning_content") == "增量思考"


# ═══════════════════ profile / 阶段参数解析 ═══════════════════

class TestProfileStageResolution:
    def test_default_profile_keeps_max_thinking(self) -> None:
        """profile 级兜底保持历史行为：deepseek-v4-flash + 思考开启 + effort=max。

        注意 effort 现在按阶段分级（见 config/llm_config.json 的 stages 表），
        所以这里断言的是 **profile 级**取值；带 stage 的生效值见
        test_stage_effort_tiers。
        """
        profile = resolve_profile()
        assert profile["model"] == "deepseek-v4-flash"
        assert profile["thinking"] is True
        assert profile["reasoning_effort"] == "max"

    def test_stage_effort_tiers(self) -> None:
        """阶段分级 effort：机械型降 low、多轮/开放生成保 max、其余走 * 的 medium。

        依据日志实测的思考冗余度（逐行枚举占比 / 思考-输出比 / 思考量per决策点）。
        取值必须落在 provider 支持的枚举内——拼错不会本地报错，而是 API 400。

        2_1 曾是 low（纯机械逐行筛查），自「拼音候选 + 世界知识存在性核查」入驻后
        升到 medium：它现在要判断"这个说法在世界上存在吗"，属开放推理而非枚举。
        1_1 同理由从 low 升 medium：它现在要自评「我读懂这段了吗」（`困惑` 字段），
        判据是理解程度而非枚举边界。3_2 话题重置保 max——它要读懂 1_1 没读懂的东西。
        """
        tiers = {
            "5_1": "low",
            # 引用校验组件：在哪个步骤内联调用就用哪个步骤的校验子号（全 low）。
            # 步骤 5/6/7 各自调它（5_3 宏观、6_3 敏感信号、7_2 主题）
            "5_3": "low", "6_3": "low", "7_2": "low",
            "2_3": "max", "3_2": "max", "4_3": "max", "4_5": "max", "4_6": "max", "5_2": "max", "6_1": "max", "7_1": "max",
            "1_1": "medium", "1_2": "medium",
            "2_1": "medium", "2_2": "medium", "3_1": "medium",
            "4_1": "medium", "4_2": "medium", "4_4": "medium", "4_7": "medium",
            "6_2": "medium",
            # 第 8 步聚合概览（独立入口 summary_overview.py，不属七步管线）
            "8_1": "medium", "8_2": "medium", "8_3": "medium", None: "medium",
        }
        for stage, expected in tiers.items():
            eff = effective_config(stage=stage)
            assert eff["reasoning_effort"] == expected, f"stage={stage}"
            assert eff["thinking"] is True, f"stage={stage} 思考必须保持开启"
            assert eff["reasoning_effort"] in ("low", "medium", "high", "max")

    def test_every_llm_call_site_uses_stage_numbered_log_name(self) -> None:
        """全仓 LLM 调用点的 `log_name` / `step_name` 必须是 `N_M_名称` 形式。

        动机：`stage_key_of` 用 `^(\\d+_\\d+)` 取阶段键去查 `config.stages`。
        无前缀**不报错**，而是静默落到全局 `*` 档——该阶段的 `reasoning_effort`
        等配置全部失效，且没有任何日志提示。本测试把这个静默失效变成显式失败。

        实现上扫源码里的字面量与 f-string 前缀，避免依赖运行时（那需要真调 LLM）。
        新增阶段时若忘了带编号，这里会红。
        """
        src_root = Path(__file__).resolve().parent.parent / "src"
        # 形如 log_name="..." / log_name=f"..." / step_name=f"{VAR}_..." 的赋值
        pat = re.compile(r"(?:log_name|step_name)\s*=\s*f?\"([^\"]*)\"")
        # 允许的形态：字面 N_M 开头，或以 f-string 变量开头（阶段号由变量注入，
        # 变量本身的取值由 test_stage_prefix_constants_are_numbered 覆盖）
        ok_literal = re.compile(r"^\d+_\d+")
        starts_with_var = re.compile(r"^\{")
        offenders: list[str] = []
        for py in src_root.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                for value in pat.findall(line):
                    if not value or value.startswith("{") and starts_with_var.match(value):
                        continue
                    if ok_literal.match(value):
                        continue
                    rel = py.relative_to(src_root)
                    offenders.append(f"{rel}:{lineno} -> {value!r}")
        assert not offenders, (
            "以下 LLM 调用点的 log_name/step_name 缺少 `第几步_第几阶段` 前缀，"
            "会导致 config.stages 配置静默失效：\n  " + "\n  ".join(offenders)
        )

    def test_stage_prefix_constants_are_numbered(self) -> None:
        """注入 log_name 的阶段号常量本身必须是 `N_M`（f-string 变量那一路的保障）。"""
        src_root = Path(__file__).resolve().parent.parent / "src"
        pat = re.compile(
            r"^\s*(?:STEP_NUMBER_STR|VALIDATION_STAGE)\s*(?::\s*str\s*)?=\s*\"([^\"]+)\"",
        )
        found: dict[str, str] = {}
        for py in src_root.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                m = pat.match(line)
                if m:
                    found[f"{py.relative_to(src_root)}:{lineno}"] = m.group(1)
        assert found, "未找到任何阶段号常量——常量若被改名，本测试会失去保护作用"
        bad = {k: v for k, v in found.items() if not re.fullmatch(r"\d+_\d+", v)}
        assert not bad, f"阶段号常量必须形如 `N_M`：{bad}"

    def test_sdk_native_retry_disabled(self) -> None:
        """重试统一收拢到 retry_policy，SDK 原生重试必须关掉，否则两层各自计数。"""
        assert OPENAI_MAX_RETRIES == 0

    def test_shared_system_prompt_contains_json_word(self) -> None:
        """json_object 模式要求 system 或 user 含 json 字样，否则服务端 400。
        system 是全系统共用的，所以只要它含就对所有阶段成立——这是单点依赖。"""
        text = load_shared_system_prompt().lower()
        assert "json" in text

    def test_shared_system_declares_all_blocks(self) -> None:
        text = load_shared_system_prompt()
        for tag in ("[TASK]", "[DATA]", "[ANNOTATION]", "[DATABASE]"):
            assert tag in text

    def test_shared_system_declares_rowid_scope(self) -> None:
        """[ROWID] 在多阶段出现，声明本次调用的工作范围（闭区间数组）。
        协议必须定义它，否则用到 [ROWID] 的阶段（1_1/1_2/2_1~2_3/3_1/6_2/7_1/
        引用校验）的模型会收到一个协议里没有的块。
        """
        text = load_shared_system_prompt()
        assert "[ROWID]" in text
        assert "工作范围" in text

    def test_shared_system_task_block_first(self) -> None:
        """静态 [TASK] 必须是第一个块：窗口化后 [DATA] 每次调用都不同，
        共享前缀全靠 [TASK] 打头。"""
        text = load_shared_system_prompt()
        assert text.index("[TASK]") < text.index("[DATA]")

    def test_shared_system_cached(self) -> None:
        """lru_cache 生效——避免每次调用都读盘。"""
        assert load_shared_system_prompt() is load_shared_system_prompt()


# ═══════════════════ 工具 strict 归一化 ═══════════════════

class TestStrictToolNormalization:
    """langchain-openai 见到 response_format 就走 beta.parse()，那条路径要求
    每个 function tool 都是 strict，否则抛 "not strict. Only strict function
    tools can be auto-parsed"。DeepSeek 本身两种都收，这纯粹是过客户端校验。"""

    def test_strict_flag_added(self) -> None:
        out = _as_strict_tools([WEB_SEARCH_TOOL])
        assert out[0]["function"]["strict"] is True

    def test_additional_properties_false(self) -> None:
        out = _as_strict_tools([WEB_SEARCH_TOOL])
        assert out[0]["function"]["parameters"]["additionalProperties"] is False

    def test_all_properties_become_required(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "t", "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                    "required": ["a"],  # 原本只有 a
                },
            },
        }
        out = _as_strict_tools([tool])
        assert set(out[0]["function"]["parameters"]["required"]) == {"a", "b"}

    def test_original_tool_not_mutated(self) -> None:
        before = json.dumps(WEB_SEARCH_TOOL, sort_keys=True)
        _as_strict_tools([WEB_SEARCH_TOOL])
        assert json.dumps(WEB_SEARCH_TOOL, sort_keys=True) == before

    def test_web_search_tool_shape(self) -> None:
        fn = WEB_SEARCH_TOOL["function"]
        assert fn["name"] == "web_search"
        assert "query" in fn["parameters"]["properties"]
        # description 要写清"何时该调用"——tool_choice 只能是 auto，
        # 模型是否调用完全取决于对描述的理解
        assert "必须" in fn["description"]


# ═══════════════════ 对话日志落盘 ═══════════════════

class TestConversationLogWriter:
    """落盘格式由 conversation_log_writer 保证：尝试文件写完定稿、
    概览原子重写、中文不得被转成 \\uXXXX（人可读是调试底线）。
    payload 字段由 client.py 构造，这里只锁定写盘行为。"""

    def _payload(self) -> dict:
        return {
            "步骤": "7_1_主题分析", "轮次": 1, "尝试": 1, "状态": "成功",
            "输入": {"system_prompt": "系统提示", "user_prompt": "中文输入"},
            "思考": "我推理了一下", "内容": "中文输出", "用量": {},
        }

    def test_attempt_log_roundtrip(self, tmp_path: Path) -> None:
        """文件名由参数决定，payload 字段原样落盘。"""
        write_attempt_log(tmp_path, "t", 2, 3, self._payload())
        d = json.loads(
            (tmp_path / "t" / "round2__attempt3.json").read_text(encoding="utf-8"),
        )
        assert d["思考"] == "我推理了一下"
        assert d["内容"] == "中文输出"

    def test_overview_log_roundtrip(self, tmp_path: Path) -> None:
        write_overview_log(tmp_path, "t", self._payload())
        d = json.loads((tmp_path / "t" / "概览.json").read_text(encoding="utf-8"))
        assert d["输入"]["user_prompt"] == "中文输入"
        assert d["步骤"] == "7_1_主题分析"

    def test_chinese_not_escaped(self, tmp_path: Path) -> None:
        """日志要人可读——不能出现 \\u4e2d\\u6587。"""
        write_attempt_log(tmp_path, "t", 1, 1, self._payload())
        raw = (tmp_path / "t" / "round1__attempt1.json").read_text(encoding="utf-8")
        assert "\\u" not in raw
        assert "中文输出" in raw

    def test_log_dir_created_on_demand(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b"
        write_attempt_log(nested, "t", 1, 1, self._payload())
        assert (nested / "t" / "round1__attempt1.json").exists()

    def test_atomic_write_leaves_no_tmp(self, tmp_path: Path) -> None:
        write_attempt_log(tmp_path, "t", 1, 1, self._payload())
        write_overview_log(tmp_path, "t", self._payload())
        assert not list((tmp_path / "t").glob("*.tmp"))


# ═══════════════════ 重试分流 ═══════════════════

class _Overload(Exception):
    """模仿 openai.InternalServerError：靠类名被识别为过载。"""


_Overload.__name__ = "InternalServerError"


class TestRetryPolicy:
    def test_succeeds_first_try(self) -> None:
        fn = MagicMock(return_value="ok")
        assert build_retrying()(fn) == "ok"
        assert fn.call_count == 1

    def test_non_overload_error_bounded(self) -> None:
        """schema 校验失败要有上限，避免模型稳定输出坏结构时无限打转。"""
        fn = MagicMock(side_effect=ValueError("坏结构"))
        with pytest.raises(ValueError):
            build_retrying()(fn)
        assert 1 < fn.call_count <= 5

    def test_recovers_after_transient_failure(self) -> None:
        fn = MagicMock(side_effect=[ValueError("第一次坏"), "ok"])
        assert build_retrying()(fn) == "ok"
        assert fn.call_count == 2

    def test_args_forwarded(self) -> None:
        fn = MagicMock(return_value="ok")
        build_retrying()(fn, 1, 2, key="v")
        fn.assert_called_once_with(1, 2, key="v")


# ═══════════════════ tool_calls 消息结构化拆解 ═══════════════════

class TestMessagesToLogList:
    """tool_calls 循环的多轮 messages 要能拆成结构化列表落盘，否则调试无从下手。"""

    def test_skips_system_message(self) -> None:
        from langchain_core.messages import HumanMessage, SystemMessage
        out = _messages_to_log_list([
            SystemMessage(content="系统提示"),
            HumanMessage(content="用户输入"),
        ])
        assert out == [{"role": "user", "content": "用户输入"}]

    def test_tool_calls_rendered(self) -> None:
        from langchain_core.messages import HumanMessage
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "盛虹 PCB"},
                         "id": "call_1"}],
        )
        out = _messages_to_log_list([HumanMessage(content="u"), ai])
        assert out[0] == {"role": "user", "content": "u"}
        assert out[1]["role"] == "ai"
        assert out[1]["tool_calls"] == [{"名称": "web_search", "参数": {"query": "盛虹 PCB"}}]

    def test_tool_result_rendered(self) -> None:
        from langchain_core.messages import HumanMessage, ToolMessage
        out = _messages_to_log_list([
            HumanMessage(content="u"),
            ToolMessage(content='{"搜索结果": []}', tool_call_id="call_1"),
        ])
        assert out[1] == {
            "role": "tool",
            "content": '{"搜索结果": []}',
            "tool_call_id": "call_1",
        }

    def test_empty_message_list(self) -> None:
        assert _messages_to_log_list([]) == []

    def test_chinese_args_not_escaped(self) -> None:
        from langchain_core.messages import HumanMessage
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "中文查询"},
                         "id": "c"}],
        )
        out = _messages_to_log_list([HumanMessage(content="u"), ai])
        assert out[1]["tool_calls"][0]["参数"]["query"] == "中文查询"
