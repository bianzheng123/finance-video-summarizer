"""回放客户端单元测试（全离线，tmp_path 伪造日志目录）。

覆盖回放匹配的三条路径（精确 / repair 模糊 / 报错）与返回语义与 live 对齐：

- `call` 回放返回 概览.结果（与 live 的"已过 schema 校验 dict"等价）；
  状态=失败 → 抛 ReplayMissError（live 语义是最终 raise）。
- `call_conversation` 回放状态=失败 → 返回 (None, trace) 不抛
  （live 语义是轮数用尽/schema 未过返回 None，管线继续）。
- 5_1 修复调用的 log_name 带随机 hex 后缀（macro_analyzer.py），
  精确查找失败后按归一化名建候选队列，user_prompt 相等者优先消费——
  修复各轮 repair_targets 不同且并发执行，顺序不定，prompt 是排他依据。

注意：LLM_REPLAY 必须在 import src 之前设置（虽然 replay_enabled() 每次
读取，但本文件同时验证 LLMClient 的 key 校验分支，先设好环境最稳）。
"""

import json
import os
import sys
from pathlib import Path

os.environ["LLM_REPLAY"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.llm_infra import replay
from src.llm_infra.client import LLMClient
from src.llm_infra.replay import (
    ReplayMissError,
    replay_call,
    replay_call_conversation,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_round_file(
    folder: Path,
    round_i: int,
    attempt_n: int,
    *,
    status: str = "成功",
    tool_calls: list[dict] | None = None,
    tool_results: list[dict] | None = None,
) -> None:
    _write_json(folder / f"round{round_i}__attempt{attempt_n}.json", {
        "步骤": folder.name,
        "轮次": round_i,
        "尝试": attempt_n,
        "状态": status,
        "输入": {"system_prompt": "sys", "messages": "摊平文本"},
        "思考": "思考内容",
        "内容": "",
        "工具调用": tool_calls or [],
        "工具结果": tool_results or [],
        "用量": {},
    })


def _make_call_dir(
    log_dir: Path,
    name: str,
    *,
    status: str = "成功",
    result: dict | None = None,
    output: str = "",
    prompt: str = "",
    rounds: list[dict] | None = None,
) -> Path:
    """伪造一个调用文件夹：概览.json + 可选 round 文件。"""
    folder = log_dir / name
    folder.mkdir(parents=True)
    _write_json(folder / "概览.json", {
        "步骤": name,
        "状态": status,
        "输入": {"system_prompt": "sys", "user_prompt": prompt},
        "输出": output,
        "结果": result,
        "调用次数": max(1, len(rounds or [])),
        "轮数": 1,
    })
    for r in rounds or []:
        _make_round_file(folder, **r)
    return folder


@pytest.fixture(autouse=True)
def _clear_queues() -> None:
    """候选消费队列是进程级注册表，逐用例清空保证隔离。"""
    replay._QUEUE_REGISTRY.clear()


# ═══════════════════ 精确命中 ═══════════════════

class TestExactMatch:
    def test_call_returns_recorded_result(self, tmp_path: Path) -> None:
        _make_call_dir(tmp_path, "7_1_主题分析_A股指数", result={"观点": "看多"})
        out = replay_call("任意 prompt", tmp_path, "7_1_主题分析_A股指数")
        assert out == {"观点": "看多"}

    def test_conversation_returns_result_and_trace(self, tmp_path: Path) -> None:
        _make_call_dir(
            tmp_path, "3_2_联网消歧_topic15",
            result={"消歧结果": []},
            rounds=[
                {"round_i": 1, "attempt_n": 1,
                 "tool_calls": [{"名称": "web_search", "参数": {"query": "盛虹"}}],
                 "tool_results": [{"名称": "web_search", "结果": "{}", "字符数": 2}]},
                {"round_i": 2, "attempt_n": 1},
            ],
        )
        result, trace = replay_call_conversation(
            "任意 prompt", {}, tmp_path, "3_2_联网消歧_topic15",
        )
        assert result == {"消歧结果": []}
        assert trace == [{
            "round": 1, "name": "web_search",
            "args": {"query": "盛虹"}, "result_chars": 2,
        }]

    def test_missing_log_dir_raises(self) -> None:
        with pytest.raises(ReplayMissError):
            replay_call("p", None, "任意")

    def test_missing_log_name_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ReplayMissError):
            replay_call("p", tmp_path, "")


# ═══════════════════ 未命中报错 ═══════════════════

class TestMissError:
    def test_non_repair_name_miss_lists_nearest(self, tmp_path: Path) -> None:
        _make_call_dir(tmp_path, "7_1_主题分析_A股指数", result={})
        with pytest.raises(ReplayMissError) as ei:
            replay_call("p", tmp_path, "7_1_主题分析_A股指教")
        assert "7_1_主题分析_A股指数" in str(ei.value)

    def test_failed_call_raises(self, tmp_path: Path) -> None:
        """call 回放状态=失败 → 抛错（live 语义是最终 raise，管线不会静默继续）。"""
        _make_call_dir(
            tmp_path, "6_3_引用校验_x", status="失败", output="坏 JSON",
        )
        with pytest.raises(ReplayMissError):
            replay_call("p", tmp_path, "6_3_引用校验_x")

    def test_failed_conversation_returns_none(self, tmp_path: Path) -> None:
        """call_conversation 回放状态=失败 → (None, trace) 不抛（live 语义）。"""
        _make_call_dir(tmp_path, "3_2_联网消歧_topic1", status="失败", output="")
        result, trace = replay_call_conversation(
            "p", {}, tmp_path, "3_2_联网消歧_topic1",
        )
        assert result is None
        assert trace == []


# ═══════════════════ repair 模糊匹配 ═══════════════════

class TestRepairFuzzyMatch:
    def _make_repairs(self, log_dir: Path) -> None:
        _make_call_dir(
            log_dir, "5_1_宏观分析_repair_仓位建议_08cb98",
            result={"仓位建议": "v1"}, prompt="修复轮1的 prompt",
        )
        _make_call_dir(
            log_dir, "5_1_宏观分析_repair_仓位建议_6d2939",
            result={"仓位建议": "v2"}, prompt="修复轮2的 prompt",
        )

    def test_exact_hex_name_hits_directly(self, tmp_path: Path) -> None:
        self._make_repairs(tmp_path)
        out = replay_call("修复轮2的 prompt", tmp_path, "5_1_宏观分析_repair_仓位建议_6d2939")
        assert out == {"仓位建议": "v2"}

    def test_new_random_hex_matches_by_prompt(self, tmp_path: Path) -> None:
        """新一次运行生成新 hex 后缀 → 精确未命中，靠 prompt 选中对应录制。"""
        self._make_repairs(tmp_path)
        out = replay_call("修复轮2的 prompt", tmp_path, "5_1_宏观分析_repair_仓位建议_ffff00")
        assert out == {"仓位建议": "v2"}

    def test_prompt_match_is_order_independent(self, tmp_path: Path) -> None:
        """并发执行顺序不定：先请求"修复轮2"也能命中对应候选。"""
        self._make_repairs(tmp_path)
        first = replay_call("修复轮2的 prompt", tmp_path, "5_1_宏观分析_repair_仓位建议_aa0001")
        second = replay_call("修复轮1的 prompt", tmp_path, "5_1_宏观分析_repair_仓位建议_aa0002")
        assert first == {"仓位建议": "v2"}
        assert second == {"仓位建议": "v1"}

    def test_queue_consumption_and_exhaustion(self, tmp_path: Path) -> None:
        self._make_repairs(tmp_path)
        replay_call("任意 prompt", tmp_path, "5_1_宏观分析_repair_仓位建议_aa0001")
        replay_call("任意 prompt", tmp_path, "5_1_宏观分析_repair_仓位建议_aa0002")
        with pytest.raises(ReplayMissError):
            replay_call("任意 prompt", tmp_path, "5_1_宏观分析_repair_仓位建议_aa0003")

    def test_plain_name_with_hex_suffix_not_stripped(self, tmp_path: Path) -> None:
        """非 repair 的 log_name 以 _<6hex> 结尾（如主题名本身）→ 不做模糊匹配。"""
        with pytest.raises(ReplayMissError):
            replay_call("p", tmp_path, "7_1_主题分析_某主题_08cb98")


# ═══════════════════ trace 重建与 schema=None ═══════════════════

class TestTraceAndNoSchema:
    def test_trace_rebuilt_from_round_files_in_natural_order(
        self, tmp_path: Path,
    ) -> None:
        _make_call_dir(
            tmp_path, "2_3_ReAct_topic3", result={"x": 1},
            rounds=[
                {"round_i": 1, "attempt_n": 1,
                 "tool_calls": [{"名称": "web_search", "参数": {"query": "a"}}],
                 "tool_results": [{"名称": "web_search", "结果": "r1", "字符数": 10}]},
                {"round_i": 2, "attempt_n": 1, "status": "校验失败"},
                {"round_i": 2, "attempt_n": 2,
                 "tool_calls": [{"名称": "web_search", "参数": {"query": "b"}}],
                 "tool_results": [{"名称": "web_search", "结果": "r2", "字符数": 11}]},
                {"round_i": 10, "attempt_n": 1,
                 "tool_calls": [{"名称": "web_search", "参数": {"query": "c"}}],
                 "tool_results": [{"名称": "web_search", "结果": "r3", "字符数": 12}]},
            ],
        )
        result, trace = replay_call_conversation(
            "p", {}, tmp_path, "2_3_ReAct_topic3",
        )
        assert result == {"x": 1}
        assert [t["round"] for t in trace] == [1, 2, 10]
        assert [t["args"]["query"] for t in trace] == ["a", "b", "c"]
        assert [t["result_chars"] for t in trace] == [10, 11, 12]

    def test_trace_skips_attempts_without_tool_calls(self, tmp_path: Path) -> None:
        _make_call_dir(tmp_path, "2_3_ReAct_topic3", result={"x": 1}, rounds=[])
        _, trace = replay_call_conversation("p", {}, tmp_path, "2_3_ReAct_topic3")
        assert trace == []

    def test_no_schema_returns_output_string(self, tmp_path: Path) -> None:
        """schema=None 的 live 语义是返回正文（字符串），回放取「输出」。"""
        _make_call_dir(
            tmp_path, "2_3_ReAct_topic3", output="纯文本正文", result=None,
        )
        result, _ = replay_call_conversation("p", None, tmp_path, "2_3_ReAct_topic3")
        assert result == "纯文本正文"

    def test_result_chars_falls_back_to_len(self, tmp_path: Path) -> None:
        """字符数字段缺失时用结果文本长度兜底。"""
        _make_call_dir(
            tmp_path, "2_3_ReAct_topic3", result={"x": 1},
            rounds=[{
                "round_i": 1, "attempt_n": 1,
                "tool_calls": [{"名称": "web_search", "参数": {"query": "a"}}],
                "tool_results": [{"名称": "web_search", "结果": "12345"}],
            }],
        )
        _, trace = replay_call_conversation("p", {}, tmp_path, "2_3_ReAct_topic3")
        assert trace[0]["result_chars"] == 5


# ═══════════════════ LLMClient 集成 ═══════════════════

class TestLLMClientReplay:
    def test_replay_shortcuts_call_entry(self, tmp_path: Path) -> None:
        """走 LLMClient.call 真实入口，验证分流生效且不写盘。"""
        _make_call_dir(tmp_path, "7_1_主题分析_A股指数", result={"观点": "看多"})
        out = LLMClient().call(
            "任意 prompt", schema={}, log_dir=tmp_path,
            log_name="7_1_主题分析_A股指数",
        )
        assert out == {"观点": "看多"}
        # 回放不写盘：文件夹里只有录制时的概览，没有新文件
        assert sorted(p.name for p in (tmp_path / "7_1_主题分析_A股指数").iterdir()) == ["概览.json"]

    def test_init_never_requires_api_key(self, monkeypatch) -> None:
        """key 检查已下沉到 _build_llm（按 profile 的 api_key_env 解析），
        __init__ 与回放路径都不需要 key。"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        LLMClient._instance = None  # 重置单例，强制重新走 __init__
        client = LLMClient()
        assert client._initialized

    def test_replay_miss_through_call_entry(self, tmp_path: Path) -> None:
        with pytest.raises(ReplayMissError):
            LLMClient().call(
                "p", schema={}, log_dir=tmp_path, log_name="不存在的调用",
            )
