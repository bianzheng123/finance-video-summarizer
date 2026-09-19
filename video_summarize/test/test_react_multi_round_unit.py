"""不触网单测：reasoning_content 发送侧透传 + 致命错误不重试。

覆盖本次修复的两个纯逻辑改动：
  1. ReasoningChatOpenAI._get_request_payload 把历史 assistant 消息的
     reasoning_content 写回请求 payload（DeepSeek 思考模式多轮对话的硬性要求）；
  2. retry 对客户端致命错误（400 等）单次尝试即放弃，不再重放 3 遍。

用法：cd video_summarize && uv run python test/test_react_multi_round_unit.py
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from tenacity import wait_none

from _common import setup_logging

from src.llm_infra import retry
from src.llm_infra.model import ReasoningChatOpenAI


class BadRequestError(Exception):
    """openai 400 形态：类名命中致命集合。"""
    status_code = 400


class PlainStatus400(Exception):
    """类名不在致命集合，仅靠 status_code=400 命中。"""
    status_code = 400


class ValidationError(Exception):
    """本模块局部类，仅用于模拟 schema 校验失败（类名不在致命集合）。"""


def _build_llm() -> ReasoningChatOpenAI:
    return ReasoningChatOpenAI(model="deepseek-v4-flash", api_key="test")


def test_request_payload_passthrough() -> None:
    """历史 assistant 消息的 reasoning_content 应写回 payload，其余消息不带。"""
    llm = _build_llm()
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "web_search",
            "args": {"query": "黄金ETF"},
            "id": "call_1",
            "type": "tool_call",
        }],
        additional_kwargs={"reasoning_content": "先搜索确认。"},
    )
    tool_msg = ToolMessage(content='{"搜索词": "黄金ETF", "搜索结果": []}', tool_call_id="call_1")
    payload = llm._get_request_payload([
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        ai_msg,
        tool_msg,
    ])
    msgs = payload["messages"]
    assert len(msgs) == 4
    assert msgs[2]["reasoning_content"] == "先搜索确认。"
    for i in (0, 1, 3):
        assert "reasoning_content" not in msgs[i], f"第 {i} 条消息不应带 reasoning_content"


def test_request_payload_noop_for_plain_messages() -> None:
    """单轮 call() 形态（纯 system/human）payload 与父类一致，不新增任何键。"""
    llm = _build_llm()
    payload = llm._get_request_payload([
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
    ])
    assert len(payload["messages"]) == 2
    for m in payload["messages"]:
        assert "reasoning_content" not in m


def test_fatal_error_detection() -> None:
    assert retry._is_fatal_error(BadRequestError("x"))
    assert retry._is_fatal_error(PlainStatus400("x"))
    assert not retry._is_fatal_error(ValidationError("x"))
    assert not retry._is_fatal_error(Exception("server overloaded"))  # 过载走无限重试分支


def test_build_retrying_fatal_stops_after_one_attempt() -> None:
    """致命错误：恰好 1 次尝试即放弃，无重放。"""
    calls = 0

    def fail() -> None:
        nonlocal calls
        calls += 1
        raise BadRequestError("400")

    try:
        retry.build_retrying()(fail)
    except BadRequestError:
        pass
    assert calls == 1


def test_build_retrying_non_fatal_bounded() -> None:
    """非致命非过载错误（如 schema 校验失败）仍走有限次数重试。"""
    calls = 0

    def fail() -> None:
        nonlocal calls
        calls += 1
        raise ValidationError("bad schema")

    retrying = retry.build_retrying()
    retrying.wait = wait_none()  # 免退避等待，加速测试
    try:
        retrying(fail)
    except ValidationError:
        pass
    assert calls == retry._NON_OVERLOAD_MAX_ATTEMPTS


def main() -> None:
    setup_logging()
    test_request_payload_passthrough()
    test_request_payload_noop_for_plain_messages()
    test_fatal_error_detection()
    test_build_retrying_fatal_stops_after_one_attempt()
    test_build_retrying_non_fatal_bounded()
    print("全部单测通过")


if __name__ == "__main__":
    main()
