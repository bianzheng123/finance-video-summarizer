"""LLM 调用的重试策略。

DeepSeek 高峰期会返回瞬时 503 Server Overloaded / 429，属于可自愈的服务端过载。
章节分析阶段最多 8 路并发，若退避太短、次数太少，几秒内就会把重试耗尽并把整轮
后处理（含 B 站投稿 / 草稿 / 邮件）连带炸掉。因此按异常类型分流重试：

  - 过载类错误（503 / 429 / 超时 / 连接错误）：**无限重试**，但受单次调用的
    总墙钟上限约束（`LLM_OVERLOAD_MAX_TOTAL_SECONDS`，默认 1800s）——超限即
    reraise 放弃，由各步断点（2_subtitle_cleaning.json 等）承接续跑，
    而不是无限挂起卡死整条管线。
  - 客户端致命错误（400 / 401 / 403 / 404 / 422，如思考模式缺 reasoning_content
    回传的 400）：**不重试**、单次尝试即放弃——重放相同请求不可能成功，重试只会
    刷出内容完全相同的失败尝试文件。
  - 其它错误（主要是 schema 校验失败 ValidationError）：仍保留有限次数，避免
    模型稳定输出坏结构时无限打转。

openai SDK 原生重试关掉（OPENAI_MAX_RETRIES=0），把重试完全收拢到这一层统一控制，
避免两层各自计数、日志刷屏、且无法按类型区分停止。
"""

import logging
import os
import time

from tenacity import RetryCallState, Retrying, retry_if_exception, wait_exponential_jitter

from .cancel import PipelineCancelled

logger = logging.getLogger(__name__)

# openai SDK 原生重试次数：关掉，重试统一收拢到本模块的 tenacity 层。
OPENAI_MAX_RETRIES = 0

# 非过载错误（如 schema 校验失败）的最大尝试次数。
_NON_OVERLOAD_MAX_ATTEMPTS = 4

# 指数退避参数：2s 起，封顶 60s，另加 0~2s 抖动。
_RETRY_INITIAL = 2.0
_RETRY_MAX = 60.0
_RETRY_EXP_BASE = 2.0
_RETRY_JITTER = 2.0

# 视为「服务端过载、可自愈」的 HTTP 状态码——命中即无限重试。
_OVERLOAD_STATUS_CODES = {429, 500, 502, 503, 504}

# 按类名匹配的过载 / 瞬时网络异常，避免在此处硬依赖 openai / httpx 的具体导入路径。
_OVERLOAD_EXCEPTION_NAMES = {
    "InternalServerError",   # openai 5xx
    "RateLimitError",        # openai 429
    "APITimeoutError",       # openai 超时
    "APIConnectionError",    # openai 连接错误
    "ConnectError",          # httpx
    "ConnectTimeout",        # httpx
    "ReadTimeout",           # httpx
    "PoolTimeout",           # httpx
    "RemoteProtocolError",   # httpx（服务端提前断流）
}

# 视为「客户端致命错误、重放必败」的 HTTP 状态码——命中即不重试、单次放弃。
# 402（余额不足）也在此列：重放相同请求不可能成功，应立即失败并让 GUI 提示充值。
_FATAL_STATUS_CODES = {400, 401, 402, 403, 404, 422}

# 按类名匹配的 openai 客户端致命异常（与过载集合同理，避免硬依赖 openai 导入
# 路径）。注意：jsonschema 的 ValidationError 不在此列，schema 校验失败仍走
# 有限次数分支。
_FATAL_EXCEPTION_NAMES = {
    "BadRequestError",           # openai 400
    "AuthenticationError",       # openai 401
    "PermissionDeniedError",     # openai 403
    "NotFoundError",             # openai 404（如模型名不存在）
    "UnprocessableEntityError",  # openai 422
}


def _is_overload_error(exc: BaseException) -> bool:
    """判断异常是否为服务端过载 / 瞬时网络类错误（应无限重试）。

    三重判定：HTTP 状态码 → 异常类名 → 错误文本兜底。
    ValidationError 等结构性错误不命中，走有限次数分支。
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _OVERLOAD_STATUS_CODES:
        return True

    if type(exc).__name__ in _OVERLOAD_EXCEPTION_NAMES:
        return True

    # 兜底：错误文本里带过载/超载字样（如上游代理把 503 包成普通 Exception）
    text = str(exc).lower()
    return "overload" in text or "server overloaded" in text


def _is_fatal_error(exc: BaseException) -> bool:
    """判断异常是否为客户端致命错误（重放相同请求不可能成功，如缺
    reasoning_content 回传的 400）。命中即不重试、单次放弃。"""
    if isinstance(exc, PipelineCancelled):
        return True  # 用户主动取消：立即停止、穿透，不做任何重试
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _FATAL_STATUS_CODES:
        return True
    return type(exc).__name__ in _FATAL_EXCEPTION_NAMES


def _overload_max_total_seconds() -> float:
    """过载类无限重试的单次调用总墙钟上限（秒），超限即放弃，由各步断点承接。"""
    return float(os.getenv("LLM_OVERLOAD_MAX_TOTAL_SECONDS", "1800"))


def _stop_unless_overload(retry_state: RetryCallState) -> bool:
    """tenacity 停止条件：致命错误立即放弃（不重试）；过载类错误在总墙钟
    上限内永不停止；其它错误到 _NON_OVERLOAD_MAX_ATTEMPTS 停。"""
    outcome = retry_state.outcome
    if outcome is not None and outcome.failed:
        exc = outcome.exception()
        # 致命错误最先判定：重放相同请求不可能成功（如缺 reasoning_content
        # 回传的 400），重试只会刷出内容完全相同的「调用失败」尝试文件，
        # 造成「同一次尝试重复多遍」的误导日志。
        if _is_fatal_error(exc):
            logger.warning("致命错误不重试，立即放弃：%s: %s", type(exc).__name__, exc)
            return True
        if _is_overload_error(exc):
            start = retry_state.start_time
            elapsed = (time.monotonic() - start) if start is not None else 0.0
            if elapsed >= _overload_max_total_seconds():
                logger.warning(
                    "过载重试已累计 %.0fs，达到上限 %.0fs，放弃本次调用",
                    elapsed, _overload_max_total_seconds(),
                )
                return True  # 放弃 → reraise，由各步断点承接续跑
            return False  # 过载且在时限内 → 继续重试
    return retry_state.attempt_number >= _NON_OVERLOAD_MAX_ATTEMPTS


def _log_before_sleep(retry_state: RetryCallState) -> None:
    """每次重试前打日志：区分过载（无限重试）与普通错误（有限次数）。"""
    outcome = retry_state.outcome
    if outcome is None or not outcome.failed:
        return
    exc = outcome.exception()
    if _is_overload_error(exc):
        logger.warning(
            "服务端过载，第 %d 次重试（总墙钟上限 %.0fs），%.1fs 后重试：%s",
            retry_state.attempt_number, _overload_max_total_seconds(),
            retry_state.next_action.sleep, exc,
        )
    else:
        logger.warning(
            "调用失败，第 %d/%d 次重试，%.1fs 后重试：%s",
            retry_state.attempt_number, _NON_OVERLOAD_MAX_ATTEMPTS,
            retry_state.next_action.sleep, exc,
        )


def build_retrying() -> Retrying:
    """构造按异常类型分流停止的 tenacity Retrying 控制器。

    过载类无限重试、其它有限次数；退避为指数抖动；reraise=True 让原始异常穿透
    （而非包成 RetryError），便于上层按类型处理。
    """
    return Retrying(
        stop=_stop_unless_overload,
        wait=wait_exponential_jitter(
            initial=_RETRY_INITIAL,
            max=_RETRY_MAX,
            exp_base=_RETRY_EXP_BASE,
            jitter=_RETRY_JITTER,
        ),
        retry=retry_if_exception(lambda _exc: True),  # 是否停由 stop 决定，这里一律允许重试
        before_sleep=_log_before_sleep,
        reraise=True,
    )
