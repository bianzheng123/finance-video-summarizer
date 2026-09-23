"""统一资源管理器 — 跨视频跨主播全局统一管理 LLM / WSA 调用的并发。

单例全局变量形式：`ResourceManager` 经 `__new__` 保证进程内唯一实例，
模块级全局变量 `RESOURCE_MANAGER` 是唯一入口，client 与 clients 共同引用：
  - LLM：按模型区分并发上限（deepseek-v4-flash 默认 2500，deepseek-v4-pro 默认 500），
    可用 `LLM_MODEL_CONCURRENCY` env（JSON dict）覆盖；未知模型走兜底 500。
  - WSA：联网搜索统一并发上限（`WSA_MAX_CONCURRENCY`，默认 6）。
  - 全局共享线程池：各阶段模块的并行 fan-out 统一由本池承载（模块不再自建线程池），
    max_workers 默认 = 各模型限额之和 + WSA 限额，可用 `SHARED_EXECUTOR_MAX_WORKERS` 覆盖。

死锁约束：嵌套 fan-out（话题任务内再扇出分块/ReAct 子任务）会占用池线程等待子任务，
因此池容量必须大于「同时阻塞在等待子任务上的父任务数」。
父任务数上限 ≈ 并发视频数（monitor 的默认执行器 ≤32）× 每视频话题数（≤32）≈ 1024，
远小于默认池容量（2500+500+6），始终有线程消化叶子任务，不会出现池内互等死锁。
"""

import json
import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)

# 默认按模型区分的 LLM 并发上限。代码中实际只使用这两个模型名
# （topic / section 阶段按视频时长选择 flash / pro）。
_DEFAULT_LLM_LIMITS: dict[str, int] = {
    "deepseek-v4-flash": 2500,
    "deepseek-v4-pro": 500,
}
# 未知模型的兜底并发上限（新增模型但未配置时按保守值放行）。
_DEFAULT_LLM_FALLBACK = 500
# WSA 联网搜索默认并发上限。
_WSA_DEFAULT_LIMIT = 6

_env_model_limits: dict = json.loads(os.getenv("LLM_MODEL_CONCURRENCY") or "{}")
LLM_MODEL_LIMITS: dict[str, int] = {**_DEFAULT_LLM_LIMITS, **_env_model_limits}
WSA_LIMIT = int(os.getenv("WSA_MAX_CONCURRENCY", str(_WSA_DEFAULT_LIMIT)))


class ResourceManager:
    """全局资源管理器（单例）：按模型 LLM 并发闸 + WSA 并发闸 + 全局共享线程池。

    单例实现同 LLMClient：`__new__` 保证任何 `ResourceManager()` 构造都返回
    进程内同一实例；模块级全局变量 `RESOURCE_MANAGER` 是唯一入口。
    """

    _instance: "ResourceManager | None" = None
    _initialized: bool = False

    def __new__(cls) -> "ResourceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        # 初始化即加载：导入时就把全局共享线程池与所有已配置资源的并发闸建好，
        # 日志顺序固定（线程池 → 各模型闸 → WSA 闸），不在运行期按首触达懒创建。
        self._llm_semaphores: dict[str, threading.BoundedSemaphore] = {}
        self._semaphores_lock = threading.Lock()
        self._executor_lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        # 1) 全局共享线程池（各阶段 fan-out 的统一执行载体）
        self._get_executor()
        # 2) 各已配置模型的 LLM 并发闸
        for model, limit in LLM_MODEL_LIMITS.items():
            self._llm_semaphores[model] = threading.BoundedSemaphore(limit)
            logger.info("为模型 %s 创建并发闸，上限 %d", model, limit)
        # 3) WSA 联网搜索并发闸
        self._wsa_semaphore: threading.BoundedSemaphore = threading.BoundedSemaphore(WSA_LIMIT)
        logger.info("创建 WSA 并发闸，上限 %d", WSA_LIMIT)

    def llm(self, model: str) -> threading.BoundedSemaphore:
        """取该模型的 LLM 并发闸；未在配置中的模型走兜底限额（首次使用时创建）。"""
        sem = self._llm_semaphores.get(model)
        if sem is None:
            with self._semaphores_lock:
                sem = self._llm_semaphores.get(model)
                if sem is None:
                    sem = threading.BoundedSemaphore(_DEFAULT_LLM_FALLBACK)
                    self._llm_semaphores[model] = sem
                    logger.info(
                        "为模型 %s 创建并发闸，上限 %d（兜底）",
                        model, _DEFAULT_LLM_FALLBACK,
                    )
        return sem

    def wsa(self) -> threading.BoundedSemaphore:
        """取 WSA 联网搜索并发闸（初始化时已创建）。"""
        return self._wsa_semaphore

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        """提交任务到全局共享线程池（各阶段 fan-out 的统一执行载体）。

        ThreadPoolExecutor.submit 已被 log_config 打了上下文传播补丁，
        因此 user_id（KVCache 隔离）与 session 日志会随任务进入池线程。
        """
        return self._get_executor().submit(fn, *args, **kwargs)

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    default_workers = sum(LLM_MODEL_LIMITS.values()) + WSA_LIMIT
                    max_workers = int(
                        os.getenv("SHARED_EXECUTOR_MAX_WORKERS", str(default_workers))
                    )
                    self._executor = ThreadPoolExecutor(max_workers=max_workers)
                    logger.info(
                        "创建全局共享线程池，max_workers=%d", max_workers,
                    )
        return self._executor


# 单例全局变量：全进程唯一入口。client / clients / 各阶段模块
# 共享同一实例，monitor 多视频并发、record 多直播间并发时都汇入同一套限制。
# 即便某处直接写 `ResourceManager()` 构造，也会经 __new__ 返回这同一实例。
RESOURCE_MANAGER = ResourceManager()


def fanout(
    stage: str,
    specs: list[tuple[Callable[..., Any], tuple, dict]],
) -> list[Future[Any]]:
    """提交一批任务到全局共享线程池，返回与 specs 对齐的 futures。

    纯 submit-all：全部任务一次性提交、并发执行。预热职责已移交
    client 层的首请求串行预热门（gate.wait_or_lead_prefix）——同
    (stage, user_id) 的首个物理请求串行完整跑真实请求（写缓存），其余请求等它
    完成后放行（后续请求命中 system_prompt + TASK 共享前缀），不再需要在这里
    串行等待首任务。
    `stage` 参数保留（调用点签名不变，供日志/审计辨识批次来源）。
    """
    return [
        RESOURCE_MANAGER.submit(fn, *args, **kwargs)
        for fn, args, kwargs in specs
    ]
