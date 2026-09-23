"""联网冒烟：call_conversation 多轮工具循环（假工具，不触真实 WSA 搜索）。

验证本次修复的实际效果：
  1. 第 2 轮起的请求带上历史 assistant 消息的 reasoning_content，不再 400；
  2. 每个轮次文件 状态=成功 且 思考 非空；
  3. round2 的 输入.messages 包含 round1 的工具结果全文（前 n 轮搜索结果
     作为第 n+1 轮输入）；
  4. 单轮 call()（3_1 新走此路径）不受 _get_request_payload 覆写影响。

日志落在 summary_data/_react_test/ 下，随后可跑
`uv run python script/extract_llm_logs.py --dir _react_test` 验证抽取布局。

用法：uv run python test/test_react_multi_round_live.py
（需 .env 里有 DEEPSEEK_API_KEY，会产生少量真实 token 费用）
"""

import json
from pathlib import Path

from _common import setup_logging

from src.llm_infra.client import LLMClient

_MARKER = "假搜索结果标记"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "搜索指定关键词，返回网页摘要列表。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索词"}},
            "required": ["query"],
        },
    },
}]

SCHEMA = {
    "type": "object",
    "properties": {
        "结论": {"type": "string"},
        "证据": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["结论", "证据"],
}

_LOG_DIR = Path("summary_data/_react_test/llm_output_9")


def dummy_executor(name: str, args: dict) -> str:
    """假工具：返回带唯一标记的确定性文本，供断言历史回灌。"""
    q = args.get("query", "")
    hits = [{"title": f"{q} 资料{i}", "snippet": f"{_MARKER} {q} 第{i}条"} for i in range(2)]
    return json.dumps({"query": q, "hits": hits}, ensure_ascii=False)


def _run_conversation() -> None:
    client = LLMClient()
    result, trace = client.call_conversation(
        user_prompt=(
            "[TASK] 先用 web_search 搜索「黄金ETF」至少一轮，"
            "再输出 JSON：结论与证据数组。"
        ),
        tools=TOOLS,
        tool_executor=dummy_executor,
        schema=SCHEMA,
        max_rounds=6,
        log_dir=_LOG_DIR,
        log_name="test_react_multi_round",
        step_name="test_react",
    )
    assert isinstance(result, dict), f"未收尾：{result!r}"
    assert len(trace) >= 1, "至少应有一轮工具调用"

    call_dir = _LOG_DIR / "test_react_multi_round"
    round_files = sorted(call_dir.glob("round*__attempt*.json"))
    assert len(round_files) >= 2, f"应至少落盘 2 个轮次文件，实际 {len(round_files)}"
    for f in round_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["状态"] == "成功", f"{f.name} 状态={data['状态']}，失败原因={data.get('失败原因', '')}"
        assert data["思考"], f"{f.name} 思考为空"
    round2 = json.loads((call_dir / "round2__attempt1.json").read_text(encoding="utf-8"))
    round2_tool_texts = [
        m.get("content", "")
        for m in round2["输入"]["messages"]
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    assert any(_MARKER in t for t in round2_tool_texts), "round2 输入未见 round1 工具结果"
    assert any(trace[0]["args"].get("query") in t for t in round2_tool_texts)
    print(f"对话冒烟通过：{len(trace)} 次工具调用、{len(round_files)} 个轮次文件、全部成功且思考非空")


def _run_single_call() -> None:
    """单轮 call() 回归（3_1 新走此路径）：纯 [system, human]，覆写应为 no-op。"""
    client = LLMClient()
    result = client.call(
        user_prompt="[TASK] 输出 JSON：{\"值\": 42}",
        schema={
            "type": "object",
            "properties": {"值": {"type": "integer"}},
            "required": ["值"],
        },
        log_dir=_LOG_DIR,
        log_name="test_single_call",
        step_name="test_single",
    )
    assert result.get("值") == 42, f"单轮 call() 结果异常：{result!r}"
    print("单轮 call() 冒烟通过")


def main() -> None:
    setup_logging()
    _run_conversation()
    _run_single_call()
    print("冒烟全部通过")


if __name__ == "__main__":
    main()
