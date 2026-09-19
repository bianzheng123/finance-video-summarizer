import contextvars
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

_log_dir = Path(__file__).parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)

# 当前所在会话（一场直播 / 一个视频）的日志文件绝对路径；不在任何会话内时为 None。
# 用 ContextVar 而非全局变量：每个 asyncio.Task 自带独立 context 副本，
# 因此多场直播并发时各自的会话值互不干扰。
_session_file: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_file", default=None)
# 会话展示名（如 "bilibili/钱博士-钱博士直播回放"），写入控制台与会话文件的
# [会话] 标签，便于区分并发场次的行；不在任何会话内时为空字符串（标签整体省略）。
_session_name: contextvars.ContextVar[str] = contextvars.ContextVar("session_name", default="")
# 当前所处步骤（如 "第4_2步-关键词合并"），写入每条日志的 [步骤] 标签；
# 不在任何步骤上下文内时为空字符串（标签整体省略）。
_step_tag: contextvars.ContextVar[str] = contextvars.ContextVar("step_tag", default="")


class _SessionFilter(logging.Filter):
    """只放行属于本会话的日志（ContextVar 命中 target）。"""

    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target

    def filter(self, record: logging.LogRecord) -> bool:
        return _session_file.get() == self.target


class _MainFilter(logging.Filter):
    """主工作流文件只收非会话日志（开播检测、全部完成等）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return _session_file.get() is None


class _PipelineFormatter(logging.Formatter):
    """统一行格式：时间 [会话] [级别] [步骤] 消息。

    会话与步骤标签为空字符串时整体省略对应方括号，避免出现 "[]"。
    with_session 控制是否渲染 [会话] 标签：控制台与会话文件渲染（便于区分并发场次），
    主工作流文件不渲染（其内容本就不属于任何会话）。
    """

    def __init__(self, datefmt: str, with_session: bool) -> None:
        super().__init__(datefmt=datefmt)
        self.with_session = with_session

    def format(self, record: logging.LogRecord) -> str:
        session = getattr(record, "session", "")
        step = getattr(record, "step", "")
        parts = [self.formatTime(record, self.datefmt)]
        if self.with_session and session:
            parts.append(f"[{session}]")
        parts.append(f"[{record.levelname}]")
        if step:
            parts.append(f"[{step}]")
        parts.append(record.getMessage())
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))
        return " ".join(parts)


def _install_context_propagation() -> None:
    """让 ThreadPoolExecutor.submit 把提交时的 ContextVar 上下文带进 worker 线程。

    线程池默认不继承 contextvars，而会话路由与步骤标签分别依赖 _session_file / _step_tag。
    LLM 分析阶段层层嵌套线程池（清洗/关键词/主题/分析），逐个包裹易漏，
    故在此统一给 submit 打补丁——每次 submit 都用 copy_context() 运行 fn，
    递归传播：父 worker 已带上下文，其子线程池 submit 时再次复制即可延续。
    loop.run_in_executor 内部也走 submit，一并覆盖。
    """
    if getattr(ThreadPoolExecutor.submit, "_ctx_wrapped", False):
        return
    _orig_submit = ThreadPoolExecutor.submit

    def _ctx_submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        ctx = contextvars.copy_context()
        return _orig_submit(self, lambda: ctx.run(fn, *args, **kwargs))

    _ctx_submit._ctx_wrapped = True
    ThreadPoolExecutor.submit = _ctx_submit


def _install_record_factory() -> None:
    """给每条日志注入 record.session（当前会话展示名）与 record.step（当前步骤标签），
    供各格式器引用。"""
    if getattr(logging.getLogRecordFactory(), "_session_aware", False):
        return
    _orig_factory = logging.getLogRecordFactory()

    def _factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = _orig_factory(*args, **kwargs)
        record.session = _session_name.get()
        record.step = _step_tag.get()
        return record

    _factory._session_aware = True
    logging.setLogRecordFactory(_factory)


def setup_logging(log_filename: str) -> None:
    """配置 root logger：控制台 + logs/{log_filename}（主工作流文件）。

    主文件只保留非会话日志；会话日志经 session_logger 分流到各自文件。
    多次调用安全——已有 handler 则跳过。
    """
    _install_context_propagation()
    _install_record_factory()

    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(_log_dir / log_filename, encoding="utf-8")
    fh.setFormatter(_PipelineFormatter(datefmt="%Y-%m-%d %H:%M:%S", with_session=False))
    fh.addFilter(_MainFilter())
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(_PipelineFormatter(datefmt="%Y-%m-%d %H:%M:%S", with_session=True))
    root.addHandler(sh)

    logging.getLogger("httpx").setLevel(logging.WARNING)


@contextmanager
def to_main_log() -> Iterator[None]:
    """在 with 块内临时脱离当前会话上下文，使这几条日志落到主工作流文件。

    用于在 session_logger 块内部（如各场结束的 finally）向主文件写"某场已完成/失败"
    这类总览指针，而正文详情仍留在会话文件里。
    """
    token = _session_file.set(None)
    try:
        yield
    finally:
        _session_file.reset(token)


@contextmanager
def session_logger(log_file: Path, session_name: str = "") -> Iterator[None]:
    """把当前 with 块（及其派生线程）的日志分流到独立文件 log_file。

    通过 ContextVar 标记会话身份，root 上临时挂一个只放行本会话记录的 FileHandler；
    退出时复位 ContextVar 并移除关闭 handler。因线程池已装上下文传播补丁，
    块内 run_in_executor / 嵌套 ThreadPoolExecutor 产生的日志也会正确归入本文件。

    Args:
        log_file: 会话日志文件的完整路径（父目录会自动创建）。
        session_name: 会话展示名，写入控制台与会话文件格式；缺省用文件名（去扩展名）。
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    target = str(log_file)
    name = session_name or log_file.stem

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(_PipelineFormatter(datefmt="%Y-%m-%d %H:%M:%S", with_session=True))
    fh.addFilter(_SessionFilter(target))
    root = logging.getLogger()
    root.addHandler(fh)

    file_token = _session_file.set(target)
    name_token = _session_name.set(name)
    try:
        yield
    finally:
        _session_file.reset(file_token)
        _session_name.reset(name_token)
        root.removeHandler(fh)
        fh.close()


@contextmanager
def step_logger(tag: str) -> Iterator[None]:
    """把当前 with 块（及其派生线程）的日志打上步骤标签 tag。

    标签命名约定：
    - 有 prompt / schema 的 LLM 阶段用「第{步}_{阶段}步-{名称}」（如 "第4_2步-关键词合并"），
      编号与 `src/llm_summarize/*/` 下的 prompt 文件名（`4_2_关键词合并.txt`）一一对应；
    - 确定性纯 Python 阶段（无 LLM 调用）用「第{步}步-{名称}」（如 "第4步-引用扩展"），
      不带阶段号——因此日志里带下划线编号的行就是 LLM 阶段。

    通过 ContextVar 注入 record.step，嵌套时内层标签覆盖外层、退出后自动恢复，
    因此步骤内更细的子阶段可以再包一层更具体的标签。注意 ContextVar 在
    `ThreadPoolExecutor.submit` 时被快照，**提交点**的标签才会带进 worker 线程：
    要给线程池里执行的 LLM 阶段打标签，得在被执行的函数体内包 with，
    而不是在组装 / 提交处包。
    """
    token = _step_tag.set(tag)
    try:
        yield
    finally:
        _step_tag.reset(token)
