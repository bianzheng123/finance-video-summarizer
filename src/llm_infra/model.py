"""带 reasoning_content 透传的 ChatOpenAI 子类。

背景：DeepSeek 的 Chat Completions 接口会在 `choices[].message.reasoning_content`
（流式为 `choices[].delta.reasoning_content`）里返回思维链，但 langchain-openai 0.3.11
在 Chat Completions 路径上完全不处理该字段——它只在 Responses API 路径认 `reasoning`。
结果是 reasoning 在响应转成 `AIMessage` 时被丢掉，回调侧永远读不到。

本模块覆盖三个转换钩子：
  - `_create_chat_result`            非流式接收侧：整段 reasoning + 思考 token 数 + 缓存命中数
  - `_convert_chunk_to_generation_chunk`  流式接收侧：逐块增量（当前走非流式，保留以防回退）
  - `_get_request_payload`           发送侧：把历史 assistant 消息的 reasoning_content
                                     写回请求 payload（DeepSeek 思考模式多轮对话的
                                     硬性要求，缺失即 HTTP 400）

流式增量无需手动累积——langchain-core 在 `AIMessageChunk` 相加时会自动拼接
`additional_kwargs` 里的同名字符串字段。
"""

from typing import Any, Optional, Union

from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

_KEY = "reasoning_content"


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI + reasoning_content 透传。除该字段外行为与父类完全一致。"""

    def _create_chat_result(
        self,
        response: Union[dict, Any],
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)

        # 响应可能是 dict，也可能是 openai SDK 的 pydantic 对象
        raw = response if isinstance(response, dict) else response.model_dump()
        choices = raw.get("choices") or []

        for gen, choice in zip(result.generations, choices):
            msg = (choice or {}).get("message") or {}
            reasoning = msg.get(_KEY)
            if reasoning:
                gen.message.additional_kwargs[_KEY] = reasoning

        # usage 归一化：按当前 profile 的 usage_map（config）把各家字段
        # 提取成三个标准键（reasoning_tokens / prompt_cache_hit|miss_tokens），
        # 挂到第一条 generation 上——下游日志与 llm_cost_stats.py 口径不变。
        usage = raw.get("usage") or {}
        if usage and result.generations:
            from .config import extract_usage, get_usage_map

            result.generations[0].message.additional_kwargs["deepseek_usage"] = (
                extract_usage(usage, get_usage_map())
            )

        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ) -> Optional[ChatGenerationChunk]:
        gen_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if gen_chunk is None:
            return None

        choices = chunk.get("choices") or []
        if choices:
            delta = (choices[0] or {}).get("delta") or {}
            reasoning = delta.get(_KEY)
            if reasoning:
                gen_chunk.message.additional_kwargs[_KEY] = reasoning

        return gen_chunk

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """发送侧透传：把历史 assistant 消息的 reasoning_content 写回请求 payload。

        DeepSeek 思考模式的硬性要求：多轮对话中，之前每一条 assistant 消息的
        reasoning_content 必须原样回传，缺失即 HTTP 400。父类（langchain-openai）
        只序列化 role/content/tool_calls 等标准字段，本方法在父类构造好的
        payload["messages"] 上按原始消息序列逐条补回。

        父类实现内部用 `self._convert_input(input_).to_messages()` 生成
        payload["messages"]，这里重新做同一转换（纯函数、无副作用），两条列表
        严格等长同序，zip(strict=True) 对齐；若走 Responses API（payload 无
        "messages" 键）则自动跳过。单轮 call() 的消息序列里没有 assistant 消息，
        本方法对其是 no-op。
        """
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        message_dicts = payload.get("messages")
        if message_dicts:
            messages = self._convert_input(input_).to_messages()
            for msg, msg_dict in zip(messages, message_dicts, strict=True):
                reasoning = msg.additional_kwargs.get(_KEY)
                if reasoning:
                    msg_dict[_KEY] = reasoning
        return payload
