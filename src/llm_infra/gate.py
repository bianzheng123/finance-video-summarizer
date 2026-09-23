"""同 (stage, user_id) 的首请求串行预热门。

DeepSeek 前缀缓存只在请求「完整完成」（输入结束 + 模型输出结束两个请求边界）
后才持久化；流式拿首个 token 即中止的请求没有输出结束边界、不写缓存。因此预热
不能靠「流式钉」低成本完成，只能让同 (stage, user_id) 的首个请求**串行完整跑
自己的真实请求**（写入缓存），其余等待者等它完成后并发发出——此时后续请求命中
system_prompt + TASK 的共享前缀（[DATA] 按批次不同不参与缓存）。

门必须在 LLM 并发闸（RESOURCE_MANAGER.llm）之前等待——否则 waiter 持满闸
名额、leader 拿不到闸会死锁；调用方（client）在函数体级插入本门，天然
满足该顺序约束。
"""

import logging
import os
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def pin_wait_seconds() -> float:
    """等待者的最长等待（秒）：防 leader 真实请求挂死导致全批阻塞的兜底，默认 1800s。

    leader 现在是「完整真实请求」（思考模式完整推理 + 过载重试墙钟），最长可达
    `LLM_OVERLOAD_MAX_TOTAL_SECONDS`（1800s），故等待兜底与之对齐。leader 最终
    必进 done/failed 并 notify，此超时仅防极端挂死。
    """
    return float(os.getenv("LLM_PIN_WAIT", "1800"))


def _warmup_stages() -> frozenset[str]:
    """预热阶段名单（env `WARMUP_STAGES`，逗号分隔的**精确 `N_M` 键**，如 `1_1,3_1,5_1`）。"""
    return frozenset(
        s.strip() for s in os.getenv("WARMUP_STAGES", "").split(",") if s.strip()
    )


class _Gate:
    """单个 (stage, user_id) 的预热门：leader 串行跑完真实请求后唤醒全部等待者。"""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._state = "unclaimed"  # unclaimed | pinning | done | failed
        self._waiters = 0

    def wait_or_lead(self, lead_fn: Callable[[], None]) -> bool:
        """leader 执行 lead_fn 后返回 True；waiter 等 leader 完成后返回 False。

        认领（unclaimed → pinning）在锁内原子完成，保证同 key 只有一个
        leader——其余到达者看到的 state 必是 pinning/done/failed。
        """
        with self._cond:
            if self._state == "unclaimed":
                self._state = "pinning"
                is_leader = True
            else:
                is_leader = False
        if is_leader:
            started = time.monotonic()
            try:
                lead_fn()
                new_state = "done"
                logger.info(
                    "首请求预热完成（真实请求已完成、缓存已写入，耗时 %.1f 秒）",
                    time.monotonic() - started,
                )
            except Exception as e:
                new_state = "failed"
                logger.warning("首请求预热失败，降级直接放行：%s", e)
            with self._cond:
                self._state = new_state
                self._cond.notify_all()
            return True

        with self._cond:
            self._waiters += 1
            try:
                wait_secs = pin_wait_seconds()
                deadline = time.monotonic() + wait_secs
                while self._state == "pinning":
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.warning(
                            "等待首请求预热超时（%.0f 秒），直接放行", wait_secs,
                        )
                        return False
                    self._cond.wait(timeout=remaining)
            finally:
                self._waiters -= 1
        return False

    def is_failed(self) -> bool:
        with self._cond:
            return self._state == "failed"


_PIN_GATES: dict[tuple[str, str], _Gate] = {}
_PIN_GATES_LOCK = threading.Lock()


def wait_or_lead_prefix(
    stage: str | None,
    user_id: str | None,
    lead_fn: Callable[[], None],
) -> bool:
    """同 (stage, user_id) 的首个请求执行 lead_fn（真实请求，串行写缓存），其余等待。

    返回 is_leader：True 表示本请求是 leader（lead_fn 已执行、结果就绪），
    False 表示 waiter（已等 leader 完成，需自行执行真实请求）。名单外阶段 /
    stage 为 None 直接返回 False（调用方自行执行，无预热）。

    done 态门常驻，后续同 (stage, user_id) 的请求零等待跳过——保证每阶段只预热
    一次（system_prompt + TASK 共享前缀只暖一次，[DATA] 按批次不同不参与缓存）。
    仅 failed 态弹出重试；断点续跑新进程首调用重新预热一次（缓存 TTL 已过期）。
    生命周期清理见 clear_user_id（挂在 user_id_scope 退出）。
    """
    if stage is None or stage not in _warmup_stages():
        return False
    key = (stage, user_id)
    with _PIN_GATES_LOCK:
        gate = _PIN_GATES.get(key)
        if gate is None:
            gate = _Gate()
            _PIN_GATES[key] = gate
    is_leader = gate.wait_or_lead(lead_fn)
    # done 态常驻：同 (stage, user_id) 后续请求零等待跳过，保证每阶段只预热一次。
    # 仅 failed 态弹出，让下一请求重新预热；统一清理挂在 user_id_scope 退出
    # （clear_user_id），防长跑服务字典无界增长。
    if gate.is_failed():
        with _PIN_GATES_LOCK:
            _PIN_GATES.pop(key, None)
    return is_leader


def clear_user_id(user_id: str) -> None:
    """删除该 user_id 及其派生 id（`{user_id}_*`）的全部预热门。

    挂在 `user_id_scope` 退出时调用：一个视频处理完，其 `base_uid` 与所有
    `{base_uid}_{stage}` 派生 id 的预热门一并清理，防长跑服务（monitor）按视频数
    无界累积 done 态预热门。
    """
    prefix = user_id + "_"
    with _PIN_GATES_LOCK:
        stale = [k for k in _PIN_GATES if k[1] == user_id or k[1].startswith(prefix)]
        for k in stale:
            _PIN_GATES.pop(k, None)
