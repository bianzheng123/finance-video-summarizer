"""LLM 客户端 - 基于 LangChain 封装，统一管理所有 LLM 调用。

本模块只承载「客户端」职责：单轮 call / 多轮 call_conversation 的调用编排、
落盘编排、schema 校验与共享 system prompt 注入。其余关注点拆分到同目录内聚模块：
  - config   profile/stages 三层参数解析 + 管线批大小/阈值配置
  - model    ReasoningChatOpenAI（reasoning_content 透传）
  - retry    重试策略（过载无限重试 / 其它有限次数）
  - replay   LLM_REPLAY 零 token 回放
  - gate     同 (stage, user_id) 首请求串行预热门
  - io       JSON 原子读写 + 对话日志落盘
  - schema   JSON Schema 加载 / 数字类型转换 / 步骤输出目录
  - resource 全局并发闸 + 共享线程池
"""

import json
import logging
import os
import random
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from jsonschema import ValidationError, validate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from .cancel import check_cancelled
from .config import (
    build_extra_body,
    effective_config,
    resolve_api_key,
    stage_key_of,
)
from .model import ReasoningChatOpenAI
from ..utils.llm_user_id import get_current_user_id

from .io import write_attempt_log, write_overview_log
from .retry import OPENAI_MAX_RETRIES, build_retrying
from .schema import _coerce_numeric_by_schema
from .gate import wait_or_lead_prefix
from .resource import RESOURCE_MANAGER
from .replay import (
    replay_call,
    replay_call_conversation,
    replay_enabled,
)

_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)

# ── 模型/阶段参数配置见 config（profile → stages 表 → call 显式参数三层解析）──
# 思考模式与 reasoning_effort 不再用模块常量，统一由 config 生效配置驱动。
# 单次 LLM 调用超时（秒）：挂死的请求不能无限占用池线程与并发闸。
# 超时抛 APITimeoutError → retry 按过载类处理（有限墙钟内重试）。
LLM_CALL_TIMEOUT = float(os.getenv("LLM_CALL_TIMEOUT", "900"))

# 全系统共用的 system prompt：输入协议 + 硬性约束 + 金融领域约定。
# 各阶段特有的规则不在这里，而是由 `*_user.jinja2` 渲染进 [TASK] 块。
_SHARED_SYSTEM_PATH = Path(__file__).parent / "prompts" / "shared_system.txt"


@lru_cache(maxsize=1)
def load_shared_system_prompt() -> str:
    """加载全系统共用的 system prompt（进程内只读一次）。"""
    return _SHARED_SYSTEM_PATH.read_text(encoding="utf-8").strip()


def _stage_user_id(stage: str | None) -> str | None:
    """每个阶段独立的 user_id：`{base_uid}_{stage}`，只缓存 system_prompt + TASK。

    `[DATA]` 按批次不同、已切小，不参与缓存；用 per-stage 的 user_id 隔离命名空间，
    避免同阶段不同批次并发时 `[DATA]` 互相冲击。`base_uid` 或 `stage` 缺失时退回
    `base_uid`（调用点无阶段号、或不在 user_id_scope 内）。
    """
    base = get_current_user_id()
    return f"{base}_{stage}" if (base and stage) else base


def _as_strict_tools(tools: list[dict]) -> list[dict]:
    """把工具定义补成 OpenAI strict 形态。

    langchain-openai 在 payload 里出现 `response_format` 时会走
    `beta.chat.completions.parse()`，该路径要求每个 function tool 都是 strict——
    否则抛 "`xxx` is not strict. Only `strict` function tools can be auto-parsed"。

    strict 形态要求：`strict: True`、`additionalProperties: false`、
    且所有 properties 都列进 `required`。DeepSeek 对 strict 与非 strict 都接受
    （已实测），这里补齐纯粹是为了过 LangChain 的客户端校验。
    """
    out: list[dict] = []
    for t in tools:
        fn = dict(t.get("function") or {})
        params = dict(fn.get("parameters") or {})
        props = params.get("properties") or {}
        params["additionalProperties"] = False
        params["required"] = list(props.keys())
        fn["parameters"] = params
        fn["strict"] = True
        out.append({**t, "function": fn})
    return out


def _tc_name(tool_call) -> str:
    """取 tool_call 的工具名（兼容 dict 与 langchain 对象两种形态）。"""
    return tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name


def _tc_args(tool_call) -> dict:
    """取 tool_call 的参数（兼容 dict 与 langchain 对象两种形态）。"""
    return tool_call.get("args") if isinstance(tool_call, dict) else tool_call.args


def _exec_one_tool(
    tool_executor: Callable[[str, dict], str],
    tool_call: dict,
) -> tuple[str, dict | None, str]:
    """执行单个工具调用，失败时返回错误 JSON（不抛出）。"""
    name = _tc_name(tool_call)
    args = _tc_args(tool_call)
    try:
        tool_result = tool_executor(name, args or {})
    except Exception as e:
        logger.warning("工具 %s 执行失败：%s", name, e)
        tool_result = json.dumps(
            {"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False,
        )
    return name, args, tool_result


def _extract_ai_message_info(ai_msg: AIMessage) -> dict[str, Any]:
    """从 AIMessage 抓取日志所需元数据：思考 / 正文 / 工具调用 / 用量。

    ReasoningChatOpenAI 把 DeepSeek 响应的 reasoning_content 与 usage
    （reasoning_tokens / prompt_cache_hit|miss_tokens）挂到
    additional_kwargs["reasoning_content"] / ["deepseek_usage"]，
    单轮 json_object 与多轮 tool_calls 两条路径都透传。
    """
    extra = getattr(ai_msg, "additional_kwargs", {}) or {}
    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    return {
        "思考": extra.get("reasoning_content", "") or "",
        "内容": getattr(ai_msg, "content", "") or "",
        "工具调用": [
            {
                "名称": (tc.get("name") if isinstance(tc, dict) else tc.name),
                "参数": (tc.get("args") if isinstance(tc, dict) else tc.args) or {},
            }
            for tc in tool_calls
        ],
        "用量": extra.get("deepseek_usage") or None,
    }


def _effective_config_audit(eff: dict[str, Any]) -> dict[str, Any]:
    """把生效配置抽成审计字段，写进每次 attempt 落盘（benchmark 核验配置真的生效）。"""
    return {
        "profile": eff.get("profile"),
        "model": eff.get("model"),
        "reasoning_effort": eff.get("reasoning_effort"),
        "thinking": eff.get("thinking"),
        "temperature": eff.get("temperature"),
        "max_tokens": eff.get("max_tokens"),
    }


def _messages_to_log_list(messages: list) -> list[dict[str, Any]]:
    """把 messages 序列拆成结构化列表，作为每轮尝试文件的「输入.messages」。

    system message 由输入.system_prompt 单独记录，这里跳过；其余消息按
    role（user/ai/tool）逐条独立成一个 JSON 对象：`content` / `tool_calls`
    / `tool_call_id` 各自只在该消息确实携带时写入，空字段不写。落盘逐条
    可核对、不与 user_prompt 形成字面重复；合并成可读文本交给抽取脚本
    （extract_llm_logs.py 的 `_format_messages`）。
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = getattr(m, "type", None) or m.__class__.__name__
        if role == "system":
            continue
        if role == "human":
            role = "user"
        item: dict[str, Any] = {"role": role}
        content = getattr(m, "content", "") or ""
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            item["tool_calls"] = [
                {
                    "名称": (c.get("name") if isinstance(c, dict) else c.name),
                    "参数": (c.get("args") if isinstance(c, dict) else c.args) or {},
                }
                for c in tool_calls
            ]
        if content:
            item["content"] = content
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            item["tool_call_id"] = tool_call_id
        out.append(item)
    return out


class _InvokeFailureRecorder(BaseCallbackHandler):
    """只记录调用失败的轻量回调（不写盘）。

    多轮对话中 tenacity 包在单次 invoke 外层，轮内 transport 重试对上层不可见；
    挂这个回调把每次失败尝试记进列表，供「调用失败」尝试文件落盘。
    """

    def __init__(self) -> None:
        super().__init__()
        self.failures: list[str] = []

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.failures.append(f"{type(error).__name__}: {error}")


class LLMClient:
    """LangChain 封装的 LLM 客户端单例。

    system prompt 由本模块统一注入（`prompts/shared_system.txt`），全系统一致，
    调用方无需传入。阶段特有的规则由调用方渲染进 user prompt 的 `[TASK]` 块。

    落盘格式统一：一次逻辑调用（单轮 `call` / 多轮 `call_conversation`）落一个
    文件夹 `{log_name}/`——每次物理尝试（含重试）一个 `round{r}__attempt{n}.json`
    （记录该次尝试的完整输入、思考、正文、工具调用/结果、用量与状态），
    另有一个 `概览.json` 做调用级聚合（步骤/状态/输入/输出/结果/调用次数/轮数）。

    使用示例：
        client = LLMClient()
        result = client.call(
            user_prompt=rendered_user_prompt,   # 含 [DATA]/[DATABASE]/[ROWID]/[TASK]
            model="deepseek-v4-flash",
            schema=my_schema,
            log_dir=Path("llm_output_1"),
            log_name="2_1_粗筛_b0",
            step_name="2_1_粗筛",
        )

    `log_name` / `step_name` **必须**以 `第几步_第几阶段_名称`（`N_M_名称`）开头——
    `config.stage_key_of` 用正则 `^(\\d+_\\d+)` 从中取阶段键去查 stages 表
    （`2_1_粗筛_b0` → `2_1`）。缺前缀不会报错，而是取不到键、静默落到全局 `*` 档，
    该阶段的 `reasoning_effort` 等配置**全部失效**。改管线阶段编号时必须同步这里的命名。
    """

    _instance = None

    def __new__(cls) -> "LLMClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        # API key 按 profile 在 _build_llm 时解析（resolve_api_key），
        # 回放模式（LLM_REPLAY=1）不触网、不构造客户端，无需 key。
        self._initialized = True
        logger.info("LLMClient 单例已初始化")

    def call(
        self,
        user_prompt: str,
        schema: dict,
        model: str | None = None,
        temperature: float | None = None,
        log_dir: Path | None = None,
        log_name: str | None = None,
        step_name: str | None = None,
        system_prompt: str | None = None,
    ) -> dict:
        """单轮调用 LLM，返回经 schema 校验的结构化结果。

        默认注入 `src/llm_infra/prompts/shared_system.txt` 作为 system prompt——
        这样所有阶段的 system message 完全一致，同一视频的所有调用配合同一份
        完整字幕 [DATA]（只有 [ROWID] 不同），可以让 DeepSeek 前缀缓存持续命中。
        传 `system_prompt` 可整体覆盖默认值（仅聚合概览等板块级 persona 场景使用），
        不传时行为与原先完全一致。

        落盘：一次调用落一个文件夹 `{log_name}/`——每次物理尝试（含 tenacity
        重试）一个 `round1__attempt{n}.json`，收尾写 `概览.json`。

        Args:
            user_prompt: 已渲染好的 user prompt，含 [DATA]/[DATABASE]/[ROWID]/[TASK] 块
            schema: JSON Schema dict，用于输出解析与校验
            model: 模型覆盖；None 时由 profile / stages 表决定
            temperature: 温度覆盖；None 时由 profile / stages 表决定
            log_dir: 日志输出目录（Path），None 则不落盘
            log_name: 调用文件夹名，如 "2_1_清洗与纠错_topic3"
            step_name: 步骤名称，写入日志的 "步骤" 字段，同时用于阶段参数匹配
            system_prompt: system prompt 整体覆盖；None 时注入共享默认模板

        Returns:
            schema 校验通过后的 dict。
        """
        # 回放模式：从历史日志读结果，不触网、不落盘（见 replay.py）
        if replay_enabled():
            return replay_call(user_prompt, log_dir=log_dir, log_name=log_name)
        stage = stage_key_of(step_name, log_name)
        eff = effective_config(
            stage=stage, model=model, temperature=temperature,
        )
        user_id = _stage_user_id(stage)
        if system_prompt is None:
            system_prompt = load_shared_system_prompt()
        # system_prompt 为空串时不发送 system message（轻量调用场景，
        # 如 4_1 主题去重——任务规则完全自含于 user prompt）
        messages: list = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))
        llm = self._build_llm(eff, user_id=user_id)
        # json_object 保证模型输出合法 JSON 形态；解析与 schema 校验在重试单元内完成，
        # 校验失败（非过载错误）同样触发 tenacity 重试（与旧 with_structured_output
        # 链路的 JsonOutputParser 失败语义等价）。
        bound = llm.bind(response_format={"type": "json_object"})
        messages_log = _messages_to_log_list(messages)
        attempt_count = 0
        attempt_content = ""  # 最后一次尝试的正文（供概览的「输出」）
        last_failure = ""  # 最后一次尝试的失败原因（供概览的「失败原因」）
        warmup_flag = False  # 前缀预热 leader 标记：leader 串行冷启动那次请求为 True

        def _write_attempt(status: str, info: dict, failure_reason: str = "") -> None:
            """把一个物理尝试写成 round1__attempt{n}.json。"""
            nonlocal attempt_count, attempt_content, last_failure
            attempt_count += 1
            attempt_content = info.get("内容", "")
            if failure_reason:
                last_failure = failure_reason
            if log_dir is None or not log_name:
                return
            payload: dict = {
                "步骤": step_name or log_name,
                "轮次": 1,
                "尝试": attempt_count,
                "状态": status,
                "输入": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "messages": messages_log,
                },
                "思考": info.get("思考", ""),
                "内容": info.get("内容", ""),
                "工具调用": info.get("工具调用", []),
                "工具结果": [],
                "用量": info.get("用量"),
                "生效配置": _effective_config_audit(eff),
            }
            if warmup_flag:
                payload["预热"] = True
            if failure_reason:
                payload["失败原因"] = failure_reason
            try:
                write_attempt_log(log_dir, log_name, 1, attempt_count, payload)
            except Exception as e:
                logger.warning("LLM 尝试日志落盘失败（%s）: %s", log_name, e)

        def _write_overview(
            status: str,
            output: str = "",
            result: Any = None,
            failure_reason: str | None = None,
        ) -> None:
            if log_dir is None or not log_name:
                return
            payload: dict = {
                "步骤": step_name or log_name,
                "状态": status,
                "输入": {"system_prompt": system_prompt, "user_prompt": user_prompt},
                "输出": output,
                "结果": result,
                "调用次数": attempt_count,
                "轮数": 1,
            }
            if user_id:
                payload["user_id"] = user_id
            if failure_reason:
                payload["失败原因"] = failure_reason
            try:
                write_overview_log(log_dir, log_name, payload)
            except Exception as e:
                logger.warning("LLM 概览日志落盘失败（%s）: %s", log_name, e)

        def _run_once() -> dict:
            """invoke + 解析 + schema 校验的一个完整尝试单元（tenacity 重试单位）。

            每次物理尝试都落一个尝试文件，重试历史与坏输出全部留痕：invoke 成功
            之后，解析 / 校验无论以何种异常失败（含 TypeError、SchemaError 等
            非 JSONDecodeError / ValidationError 的异常），坏输出都必须先落盘
            再抛出交给重试——凡调用了 LLM，其状态与结果必须落盘。
            """
            check_cancelled()  # 用户主动取消：在真正发出请求前拦截（含重试循环）
            try:
                with RESOURCE_MANAGER.llm(eff["model"]):
                    ai_msg = bound.invoke(messages)
            except Exception as e:
                _write_attempt("调用失败", {}, f"{type(e).__name__}: {e}")
                raise
            info: dict[str, Any] = {}
            try:
                info = _extract_ai_message_info(ai_msg)
            except Exception as e:
                _write_attempt("调用失败", {}, f"响应信息提取失败: {type(e).__name__}: {e}")
                raise
            try:
                parsed = json.loads(info["内容"])
                validated = self._validate_against_schema(parsed, schema)
            except json.JSONDecodeError as e:
                _write_attempt("校验失败", info, f"正文不是合法 JSON: {e}")
                raise
            except ValidationError as e:
                _write_attempt("校验失败", info, f"JSON Schema 校验失败: {e.message}")
                raise
            except Exception as e:
                _write_attempt("校验失败", info, f"输出处理异常: {type(e).__name__}: {e}")
                raise
            _write_attempt("成功", info)
            return validated

        # 前缀预热：同 (stage, user_id) 的首个请求串行完整跑真实请求（写缓存），
        # 其余等待者等它完成后并发——后续请求命中 system_prompt + TASK 共享前缀。
        # 必须在 LLM 并发闸之前等待（见 gate 模块说明）；leader 结果直接复用，
        # 绝不 invoke 两次。
        leader_result: dict | None = None
        leader_failure: Exception | None = None

        def _lead_run() -> None:
            nonlocal leader_result, leader_failure, warmup_flag
            warmup_flag = True
            try:
                leader_result = build_retrying()(_run_once)
            except Exception as e:
                leader_failure = e
            finally:
                warmup_flag = False

        is_leader = wait_or_lead_prefix(stage, user_id, _lead_run)

        # 全局限流在重试单元内按尝试持有；按异常类型分流重试：过载类无限重试，
        # 其它有限次数（见 retry）。
        if is_leader:
            if leader_failure is not None:
                _write_overview(
                    "失败",
                    failure_reason=last_failure
                    or f"{type(leader_failure).__name__}: {leader_failure}",
                )
                raise leader_failure
            # leader_failure is None ⟺ _run_once 成功返回（必非 None）
            assert leader_result is not None
            validated = leader_result
            _write_overview("成功", output=attempt_content, result=validated)
            return validated

        try:
            validated = build_retrying()(_run_once)
        except Exception as e:
            _write_overview("失败", failure_reason=last_failure or f"{type(e).__name__}: {e}")
            raise
        _write_overview("成功", output=attempt_content, result=validated)
        return validated

    def call_conversation(
        self,
        user_prompt: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        schema: dict | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_rounds: int = 8,
        dedup_directions: bool = False,
        max_tool_calls: int | None = None,
        tool_call_limit_names: Iterable[str] | None = None,
        log_dir: Path | None = None,
        log_name: str | None = None,
        step_name: str | None = None,
        system_prompt: str | None = None,
    ) -> tuple[dict | str | None, list[dict]]:
        """多轮对话调用：带原生 tool_calls 循环（对齐 DeepSeek Tool Calls 规范）。

        循环逻辑：
          1. 带 `tools` 调模型。
          2. 若 `finish_reason == "tool_calls"`：一轮多个工具并行执行，
             把 assistant 消息与各 `role="tool"` 结果追加进 messages，进入下一轮。
          3. 若模型给出正文：按 schema 解析校验后作为最终结果返回。
          4. 校验失败且未到 max_rounds：把错误文本注入 messages 占下一轮校正。
          5. 轮数用尽仍未收尾：返回 (None, trace)。

        落盘：一次调用落一个文件夹 `{log_name}/`——每轮每次尝试一个
        `round{r}__attempt{n}.json`（输入为该次实际发送的完整消息序列），
        每轮结束后原子重写 `概览.json`（崩溃残留状态为「进行中」）。

        实测约束（见 https://api-docs.deepseek.com/guides/tool_calls）：
          - 思考模式下 `tool_choice` 只能 `"auto"`；`"required"` 与指定函数字典会被拒。
          - `response_format={"type":"json_object"}` 可与 `tools` 同轮共存，不会压制 tool_calls，
            因此每轮都带 json_object，收尾轮的正文自然是合法 JSON。

        Args:
            user_prompt: 已渲染好的 user prompt（含 [DATA]/[DATABASE]/[TASK] 块）
            tools: OpenAI 格式的工具定义列表
            tool_executor: 工具执行器 `(name, args_dict) -> str`，返回值作为 tool 消息内容
            schema: 最终结果的 JSON Schema；None 表示不校验、返回原始正文
            max_rounds: 最多调用模型几轮（含收尾轮与校正轮）
            system_prompt: system prompt 整体覆盖；None 时注入共享默认模板

        Returns:
            `(最终结果, 工具调用轨迹)`。轨迹每项形如
            `{"round": 1, "name": "web_search", "args": {...}, "result_chars": 1234}`。
        """
        # 回放模式：从历史日志读结果与轨迹，不触网、不落盘（见 replay.py）
        if replay_enabled():
            return replay_call_conversation(
                user_prompt, schema, log_dir=log_dir, log_name=log_name,
            )
        stage = stage_key_of(step_name, log_name)
        eff = effective_config(
            stage=stage, model=model, temperature=temperature,
        )
        user_id = _stage_user_id(stage)
        if system_prompt is None:
            system_prompt = load_shared_system_prompt()
        # system_prompt 为空串时不发送 system message（与 call() 同口径）
        messages: list = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))
        trace: list[dict] = []
        used_directions: set[str] = set()  # 方向去重：已搜索过的方向标签
        next_correction = ""  # 上一轮校验失败时注入下一轮的校正消息
        attempt_total = 0

        llm = self._build_llm(eff, user_id=user_id)
        bound = llm.bind_tools(_as_strict_tools(tools), tool_choice="auto")
        if schema:
            # json_object 与 tools 同轮共存：收尾轮正文即合法 JSON。
            # 注意：langchain-openai 见到 response_format 就走
            # `beta.chat.completions.parse()`，那条路径要求工具必须是 strict——
            # 故上面对工具做了 strict 归一化。DeepSeek 本身两种都接受。
            bound = bound.bind(response_format={"type": "json_object"})

        # 前缀预热：同 (stage, user_id) 的首个请求串行完整跑首轮 invoke（写首轮
        # 缓存），其余等待者等它完成后并发——后续请求命中 system_prompt + TASK
        # 共享前缀（后续轮含 assistant 历史、前缀不同不再暖）。leader 首轮结果
        # 直接复用，绝不 invoke 两次。
        leader_first_msg: AIMessage | None = None
        leader_first_exc: Exception | None = None
        leader_first_failures: list[str] = []

        def _lead_first_round() -> None:
            nonlocal leader_first_msg, leader_first_exc
            rec = _InvokeFailureRecorder()
            check_cancelled()  # 用户主动取消：预热首请求发出前拦截
            try:
                with RESOURCE_MANAGER.llm(eff["model"]):
                    leader_first_msg = build_retrying()(
                        bound.invoke, messages, config={"callbacks": [rec]}
                    )
            except Exception as e:
                leader_first_exc = e
                leader_first_failures.extend(rec.failures)

        is_leader = wait_or_lead_prefix(stage, user_id, _lead_first_round)

        def _write_attempt(
            round_i: int,
            attempt_n: int,
            status: str,
            info: dict,
            messages_log: list[dict[str, Any]],
            tool_results: list[dict] | None = None,
            failure_reason: str = "",
            ) -> None:
            """把一个轮/尝试写进调用文件夹。"""
            if log_dir is None or not log_name:
                return
            payload: dict = {
                "步骤": step_name or log_name,
                "轮次": round_i,
                "尝试": attempt_n,
                "状态": status,
                "输入": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "messages": messages_log,
                },
                "思考": info.get("思考", ""),
                "内容": info.get("内容", ""),
                "工具调用": info.get("工具调用", []),
                "工具结果": tool_results or [],
                "用量": info.get("用量"),
                "生效配置": _effective_config_audit(eff),
            }
            if round_i == 1 and is_leader:
                payload["预热"] = True
            if failure_reason:
                payload["失败原因"] = failure_reason
            try:
                write_attempt_log(log_dir, log_name, round_i, attempt_n, payload)
            except Exception as e:
                logger.warning("LLM 尝试日志落盘失败（%s）: %s", log_name, e)

        def _write_overview(
            round_i: int,
            status: str,
            output: str = "",
            result: Any = None,
            failure_reason: str | None = None,
        ) -> None:
            if log_dir is None or not log_name:
                return
            payload: dict = {
                "步骤": step_name or log_name,
                "状态": status,
                "输入": {"system_prompt": system_prompt, "user_prompt": user_prompt},
                "输出": output,
                "结果": result,
                "调用次数": attempt_total,
                "轮数": round_i,
            }
            if user_id:
                payload["user_id"] = user_id
            if failure_reason:
                payload["失败原因"] = failure_reason
            try:
                write_overview_log(log_dir, log_name, payload)
            except Exception as e:
                logger.warning("LLM 概览日志落盘失败（%s）: %s", log_name, e)

        for round_i in range(1, max_rounds + 1):
            check_cancelled()  # 用户主动取消：每轮（含工具调用轮）发出请求前拦截
            round_attempt = 0
            recorder = _InvokeFailureRecorder()
            if next_correction:
                messages.append(HumanMessage(content=next_correction))
                next_correction = ""
            # 本轮实际发送给模型的消息序列（含历史轮与工具结果），在追加
            # 本轮工具结果前快照——它才是本轮的输入，工具结果是本轮产出。
            round_messages_log = _messages_to_log_list(messages)

            def _record_attempt(
                status: str,
                info: dict,
                tool_results: list[dict] | None = None,
                failure_reason: str = "",
            ) -> None:
                nonlocal round_attempt, attempt_total
                round_attempt += 1
                attempt_total += 1
                _write_attempt(
                    round_i, round_attempt, status, info,
                    round_messages_log, tool_results, failure_reason,
                )

            # 全局限流：按模型区分并发上限（见 resource），跨视频跨主播统一管理。
            # tenacity 只包 invoke：轮内 transport 重试（过载无限 / 其它 ≤4 次），
            # 每次失败尝试由 recorder 收集，各写一个「调用失败」尝试文件。
            if round_i == 1 and is_leader:
                # leader 已在预热阶段串行跑完首轮，直接复用其结果（不重复 invoke）
                if leader_first_exc is not None:
                    for msg in leader_first_failures:
                        _record_attempt("调用失败", {}, failure_reason=msg)
                    if not leader_first_failures:
                        _record_attempt(
                            "调用失败", {},
                            failure_reason=f"{type(leader_first_exc).__name__}: {leader_first_exc}",
                        )
                    failure_reason = (
                        leader_first_failures[-1]
                        if leader_first_failures
                        else f"{type(leader_first_exc).__name__}: {leader_first_exc}"
                    )
                    _write_overview(round_i, "失败", failure_reason=failure_reason)
                    raise leader_first_exc
                assert leader_first_msg is not None
                ai_msg = leader_first_msg
            else:
                try:
                    with RESOURCE_MANAGER.llm(eff["model"]):
                        ai_msg = build_retrying()(bound.invoke, messages, config={"callbacks": [recorder]})
                except Exception as e:
                    for msg in recorder.failures:
                        _record_attempt("调用失败", {}, failure_reason=msg)
                    if not recorder.failures:
                        _record_attempt("调用失败", {}, failure_reason=f"{type(e).__name__}: {e}")
                    failure_reason = (
                        recorder.failures[-1] if recorder.failures else f"{type(e).__name__}: {e}"
                    )
                    _write_overview(round_i, "失败", failure_reason=failure_reason)
                    raise

            try:
                info = _extract_ai_message_info(ai_msg)
            except Exception as e:
                _record_attempt("调用失败", {}, failure_reason=f"响应信息提取失败: {type(e).__name__}: {e}")
                _write_overview(round_i, "失败", failure_reason=f"{type(e).__name__}: {e}")
                raise
            tool_calls = getattr(ai_msg, "tool_calls", None) or []

            # ── 模型要调工具 ──
            if tool_calls:
                # 工具调用硬上限：累计已执行的受限工具数 + 本轮拟执行数，超限则整轮拒绝、强制收尾
                if max_tool_calls is not None:
                    limit_names = set(tool_call_limit_names) if tool_call_limit_names is not None else None
                    executed_cnt = (
                        sum(1 for t in trace if t["name"] in limit_names)
                        if limit_names is not None else len(trace)
                    )
                    incoming_cnt = (
                        sum(1 for tc in tool_calls if _tc_name(tc) in limit_names)
                        if limit_names is not None else len(tool_calls)
                    )
                    if executed_cnt + incoming_cnt > max_tool_calls:
                        _record_attempt(
                            "达工具上限", info,
                            failure_reason=(
                                f"工具调用将超硬上限 {max_tool_calls} 次"
                                f"（已 {executed_cnt} + 本轮 {incoming_cnt}）"
                            ),
                        )
                        next_correction = (
                            f"工具调用次数已达硬上限（{max_tool_calls} 次）。"
                            "请不要再调用任何工具，直接输出最终 JSON 结果。"
                        )
                        # 达上限被拦：把本轮拟发起的 web_search direction 同步记入去重集合，
                        # 防止下一轮模型换同方向绕过上限重新执行（b10 实测曾出现此问题）。
                        if dedup_directions:
                            for tc in tool_calls:
                                if _tc_name(tc) == "web_search":
                                    direction = ((_tc_args(tc) or {}).get("direction") or "").strip()
                                    if direction:
                                        used_directions.add(direction)
                        _write_overview(round_i, "进行中")
                        continue

                messages.append(ai_msg)
                # 方向去重：同一 direction 本轮/跨轮只执行一次；本轮同方向多个随机保留一个，
                # 其余与被跨轮拒绝的一并回灌提示。无 direction（或非 web_search，如 row_pinyin）
                # 不参与去重、直接执行。
                groups: dict[str, list] = {}   # direction -> [(tc, name, args)]
                no_dir: list = []              # 不参与去重的 tool_call
                for tc in tool_calls:
                    name = _tc_name(tc)
                    args = _tc_args(tc) or {}
                    direction = ""
                    if dedup_directions and name == "web_search":
                        direction = (args.get("direction") or "").strip()
                    if direction:
                        groups.setdefault(direction, []).append((tc, name, args))
                    else:
                        no_dir.append((tc, name, args))

                execute: list = []            # 要执行的 (tc, name, args)
                rejections: list[tuple] = []  # (tc, name, args, reason)
                for direction, items in groups.items():
                    if direction in used_directions:
                        for tc, name, args in items:
                            rejections.append(
                                (tc, name, args, "该方向已搜索过，请换一个不同方向或直接输出结果")
                            )
                    else:
                        keep_tc, keep_name, keep_args = random.choice(items)
                        used_directions.add(direction)
                        for tc, name, args in items:
                            if tc is keep_tc:
                                execute.append((tc, name, args))
                            else:
                                rejections.append((tc, name, args, "同方向已合并，未重复搜索"))
                execute.extend(no_dir)

                # 并行执行要执行的工具调用
                if execute:
                    if len(execute) > 1:
                        futures = [
                            RESOURCE_MANAGER.submit(_exec_one_tool, tool_executor, tc)
                            for tc, _n, _a in execute
                        ]
                        executed_results = [f.result()[2] for f in futures]
                    else:
                        executed_results = [_exec_one_tool(tool_executor, execute[0][0])[2]]
                else:
                    executed_results = []

                tool_results: list[dict] = []
                for (tc, name, args), tool_result in zip(execute, executed_results):
                    tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
                    trace.append({
                        "round": round_i,
                        "name": name,
                        "args": args,
                        "result_chars": len(tool_result),
                    })
                    tool_results.append({
                        "名称": name,
                        "结果": tool_result,
                        "字符数": len(tool_result),
                    })
                    messages.append(ToolMessage(content=tool_result, tool_call_id=tc_id))
                for tc, name, args, reason in rejections:
                    tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
                    reject_msg = json.dumps({"拒绝": reason}, ensure_ascii=False)
                    trace.append({
                        "round": round_i,
                        "name": name,
                        "args": args,
                        "result_chars": len(reject_msg),
                        "拒绝": reason,
                    })
                    tool_results.append({
                        "名称": name,
                        "结果": reject_msg,
                        "字符数": len(reject_msg),
                    })
                    messages.append(ToolMessage(content=reject_msg, tool_call_id=tc_id))
                _record_attempt("成功", info, tool_results=tool_results)
                _write_overview(round_i, "进行中")
                continue

            # ── 模型给出正文：收尾 ──
            content = info["内容"]
            if not schema:
                _record_attempt("成功", info)
                _write_overview(round_i, "成功", output=content)
                return content, trace
            try:
                parsed = json.loads(content)
                validated = self._validate_against_schema(parsed, schema)
            except Exception as e:
                if isinstance(e, json.JSONDecodeError):
                    fail_reason = f"正文不是合法 JSON: {e}"
                elif isinstance(e, ValidationError):
                    fail_reason = f"JSON Schema 校验失败: {e.message}"
                else:
                    fail_reason = f"输出处理异常: {type(e).__name__}: {e}"
                # 坏输出先落盘留痕，再决定注入校正消息进下一轮 / 收尾失败。
                # 任何异常（含 TypeError、SchemaError 等）都走这里——
                # 凡调用了 LLM，其状态与结果必须落盘。
                _record_attempt("校验失败", info, failure_reason=fail_reason)
                if round_i < max_rounds:
                    # 把失败的 assistant 输出回灌历史：DeepSeek 思考模式要求
                    # 历史 assistant 消息（连同其 reasoning_content）必须原样
                    # 回传，且模型要能看到自己上一轮的坏输出才能定向修正。
                    # 下一轮输入序为 [..., ai_msg(坏 JSON), Human(校正提示)]。
                    messages.append(ai_msg)
                    next_correction = (
                        f"你上一条输出的 JSON 校验失败：{fail_reason}。"
                        "请按 schema 修正后重新输出完整 JSON，不要再调用工具。"
                    )
                    _write_overview(round_i, "进行中")
                    continue
                _write_overview(round_i, "失败", output=content, failure_reason=f"schema 校验失败: {fail_reason}")
                return None, trace
            _record_attempt("成功", info)
            _write_overview(round_i, "成功", output=content, result=validated)
            return validated, trace

        logger.warning(
            "%s 轮数用尽（%d 轮）仍未收尾，已调用工具 %d 次",
            log_name or step_name or "?", max_rounds, len(trace),
        )
        _write_overview(
            max_rounds,
            "失败",
            failure_reason=f"轮数用尽（{max_rounds} 轮）仍未收尾，已调用工具 {len(trace)} 次",
        )
        return None, trace

    @staticmethod
    def _validate_against_schema(parsed: Any, schema: dict) -> Any:
        """链路内的 schema 校验步骤（数字类型转换 + jsonschema.validate）。

        json_mode 只保证合法 JSON、不校验结构，因此这里显式校验：
        - 先按 schema 把字符串形态的数字字段转成数值（避免误伤 ETF/港股代码）；
        - 再用 jsonschema.validate 校验结构，不合规时抛 ValidationError。

        单轮 call 中抛出的 ValidationError 会被 tenacity 捕获并触发重新调用 LLM
        （属非过载错误，走有限次数分支）；多轮 call_conversation 中由调用循环
        捕获并注入校正消息占下一轮。
        """
        coerced = _coerce_numeric_by_schema(parsed, schema)
        # DeepSeek 在 response_format={"type":"json_object"} 下偶发把该元字段
        # {"type":"json_object"} 原样写进 JSON 顶层，触发 schema 顶层
        # additionalProperties:false 判非法而引发无谓重试。该字段是响应格式
        # 元信息、永非业务字段，校验前剔除（全量 schema 顶层均无合法 type 业务字段）。
        if isinstance(coerced, dict) and coerced.get("type") == "json_object":
            coerced.pop("type", None)
        try:
            validate(instance=coerced, schema=schema)
        except ValidationError as e:
            logger.warning(
                "JSON Schema 校验失败：%s（路径: %s）",
                e.message, list(e.absolute_path),
            )
            raise
        return coerced

    def _build_llm(
        self,
        eff: dict[str, Any],
        timeout: float = LLM_CALL_TIMEOUT,
        user_id: str | None = None,
        streaming: bool = False,
    ) -> ReasoningChatOpenAI:
        """按生效配置构造 LLM 客户端（profile → stages 表 → call 显式参数三层解析后）。

        - `streaming`：默认 False（非流式，reasoning_content 直接在 message 上，无需
          拼增量）；预留参数（当前无流式调用点，保留以防回退）。
        - `reasoning_effort` / `thinking`：按 provider 翻译（config.build_extra_body），
          reasoning_effort 为 None 时不传（非思考类 provider 无此参数）。
        - `max_tokens`：输出预算（**含思维链**），profile 层拉到模型上限，避免
          reasoning 吃光默认预算后正文 0 token。经 `extra_body` 下发而非
          ChatOpenAI 同名字段——后者会被改名成 DeepSeek 忽略的
          `max_completion_tokens`（见 config.build_extra_body）。
        - `extra_body["user_id"]`：DeepSeek 的 user_id KV 隔离，**每阶段独立派生**
          （`{base_uid}_{stage}`，见 `_stage_user_id`），仅 kv_user_id=true 的 profile
          注入；不在 user_id_scope 内则不传。
        - `timeout`：SDK 超时（默认 `LLM_CALL_TIMEOUT`）。
        """
        kwargs: dict[str, Any] = {
            "model": eff["model"],
            "temperature": eff.get("temperature", 0.0),
            "base_url": eff["base_url"],
            "api_key": resolve_api_key(eff),
            "streaming": streaming,
            "max_retries": OPENAI_MAX_RETRIES,  # SDK 原生重试关掉，统一交给 retry
            "timeout": timeout,
        }
        if eff.get("reasoning_effort") is not None:
            kwargs["reasoning_effort"] = eff["reasoning_effort"]
        kwargs["extra_body"] = build_extra_body(eff, user_id)
        return ReasoningChatOpenAI(**kwargs)
