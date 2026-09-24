"""任务进度组件与步骤标签映射。

进度条节点由管线日志 ``record.step`` 标签驱动（``log_stream.LogLine.step`` 已携带
``step_logger`` 注入的标签）。本模块只做两件事，不改管线：

- :func:`step_key_of`：把任意 step 标签确定性地映射到标准进度节点名（纯函数，便于排查）；
- :class:`TaskProgressBar`：百分比进度条 + 「正在做：xxx」+ 「预计剩余约 xx 分钟」。

第 5/6/7 步（宏观/交易/主题）在 ``llm_analysis.py`` 里并行执行，到达顺序不定，
故进度按「到达即计入」处理：已到达节点的权重累加为完成百分比，最新到达的节点名
显示为当前正在做的事。

剩余时间（ETA）双路估算取较大值，避免乐观跳变：

- **时长系数**：``视频时长 × FULL/ECONOMIC_TIME_RATIO - 已耗时``（需 worker 预取时长）；
- **耗时外推**：进度 ≥20% 后按 ``已耗时 × (1-p) / p`` 自适应真实速度。
"""

from __future__ import annotations

import re
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

# LLM 步骤号 → 进度节点显示名（与 llm_analysis.py 的七步一一对应，用完整名便于理解）。
_LLM_STEP_NAMES: dict[int, str] = {
    1: "话题分割",
    2: "字幕纠错",
    3: "关键词剪枝",
    4: "主题聚类",
    5: "宏观分析",
    6: "交易技巧",
    7: "主题分析",
}

# 步骤标签形如「第4步-聚类」「第4_2步-关键词合并」，统一用前缀数字归入主步骤。
_LLM_STEP_RE = re.compile(r"^第(\d+)")


def step_key_of(tag: str) -> str | None:
    """把 ``record.step`` 标签映射为标准进度节点名；无法归入进度条的返回 ``None``。

    - ``"字幕识别"`` → ``"语音识别"``；``"渲染"`` → ``"渲染"``；
    - ``"第N步-…"`` / ``"第N_M步-…"`` → 第 N 步对应的节点名（N=1..7）；
    - ``"视频信息"`` 等不点亮任何节点，返回 ``None``（该阶段只是解析 URL，很快）。
    """
    if not tag:
        return None
    if tag == "字幕识别":
        return "语音识别"
    if tag == "渲染":
        return "渲染"
    m = _LLM_STEP_RE.match(tag)
    if m:
        return _LLM_STEP_NAMES.get(int(m.group(1)))
    return None


# 完整版与经济版的进度节点序列（对外共享，SummaryPage 按任务是否经济版选用）。
# 注意：节点名必须与 step_key_of 的返回值一致，mark_reached 才能命中。
FULL_STEPS: list[str] = [
    "语音识别",
    "话题分割",
    "字幕纠错",
    "关键词剪枝",
    "主题聚类",
    "宏观分析",
    "交易技巧",
    "主题分析",
    "渲染",
]

ECONOMIC_STEPS: list[str] = ["语音识别", "宏观分析", "渲染"]

# 节点耗时权重（合计 100）：按各阶段在总耗时中的占比估值，决定进度条百分比。
# ASR 与 LLM 分析是大头；1~4 步按调用量与实测日志占比分摊。
_STEP_WEIGHTS_FULL: dict[str, int] = {
    "语音识别": 30,
    "话题分割": 5,
    "字幕纠错": 15,
    "关键词剪枝": 5,
    "主题聚类": 10,
    "宏观分析": 10,
    "交易技巧": 8,
    "主题分析": 12,
    "渲染": 5,
}
_STEP_WEIGHTS_ECONOMIC: dict[str, int] = {
    "语音识别": 55,
    "宏观分析": 35,
    "渲染": 10,
}

# 预计总耗时 = 视频时长 × 系数。系数按 video_summarize/summary_data 历史运行日志
# （workflow.log / workflow_economic.log 起止时间差 ÷ metadata.json.duration）实测：
# - 完整版 18 例比例中位数 0.43，区间 0.23~0.97，方差大（受字幕行数、LLM 修复回路、
#   API 负载影响，见 script/calibrate_eta.py），取 0.45 略偏保守；
# - 经济版样本少（本仓库 1 例 0.05，历史记录 0.05~0.109），取 0.08。
# 比例只是先验，进度 ≥20% 后 ETA 会叠加耗时外推自动纠偏，收敛到真实速度。
FULL_TIME_RATIO = 0.45
ECONOMIC_TIME_RATIO = 0.08

# 进度超过该比例后启用耗时外推（此前样本太少，外推不稳）。
_EXTRAPOLATE_MIN_P = 0.2


def _format_duration(seconds: float) -> str:
    """把秒数格式化为「X 小时 Y 分钟 / X 分钟 / 不到 1 分钟」。"""
    if seconds < 60:
        return "不到 1 分钟"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"约 {minutes} 分钟"
    hours, mins = divmod(minutes, 60)
    return f"约 {hours} 小时 {mins} 分钟" if mins else f"约 {hours} 小时"


class TaskProgressBar(QWidget):
    """单任务进度条：百分比 + 当前步骤 + 预计剩余时间。

    构造传入步骤名列表，随后用 :meth:`mark_reached` 按节点名推进；未知节点名安全忽略。
    ETA 每秒自动刷新；任务成功时调 :meth:`complete` 置 100% 并停表。
    """

    def __init__(self, steps: list[str], *, economic: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._steps = list(steps)
        self._weights = _STEP_WEIGHTS_ECONOMIC if economic else _STEP_WEIGHTS_FULL
        self._ratio = ECONOMIC_TIME_RATIO if economic else FULL_TIME_RATIO
        self._reached: set[int] = set()
        self._duration_s: float | None = None
        self._start_ts = time.monotonic()
        self._finished = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        bar_row = QHBoxLayout()
        bar_row.setSpacing(10)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        # 百分比放右侧文字而非条内：细进度条内嵌文字在任何进度下都难兼顾对比度。
        self._bar.setTextVisible(False)
        bar_row.addWidget(self._bar, 1)
        self._step_label = QLabel("0% · 准备中…")
        self._step_label.setStyleSheet("color: #6B7280;")
        bar_row.addWidget(self._step_label)
        root.addLayout(bar_row)

        self._eta_label = QLabel("预计剩余：估算中…")
        self._eta_label.setStyleSheet("color: #6B7280;")
        root.addWidget(self._eta_label)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh_eta)
        self._timer.start()

    # ---- 进度推进 ----
    def _fraction(self) -> float:
        """完成比例 = 已到达节点权重和 / 总权重（0~1）。"""
        total = sum(self._weights.get(name, 0) for name in self._steps)
        if total <= 0:
            return 0.0
        done = sum(self._weights.get(self._steps[i], 0) for i in self._reached)
        return min(done / total, 1.0)

    def mark_reached(self, key: str) -> None:
        """``key`` 命中节点列表时计入进度；未知 key 忽略。"""
        if self._finished or key not in self._steps:
            return
        self._reached.add(self._steps.index(key))
        pct = round(self._fraction() * 100)
        self._bar.setValue(pct)
        self._step_label.setText(f"{pct}% · 正在做：{key}")
        self._refresh_eta()

    def complete(self) -> None:
        """全部节点置为完成（任务成功收尾时调用）。"""
        self._finished = True
        self._reached = set(range(len(self._steps)))
        self._bar.setValue(100)
        self._step_label.setText("100% · 已完成")
        self._eta_label.hide()
        self._timer.stop()

    def fail(self) -> None:
        """任务失败：停表并提示（保留已到的百分比作参考）。"""
        self._finished = True
        pct = round(self._fraction() * 100)
        self._step_label.setText(f"{pct}% · 已失败")
        self._eta_label.hide()
        self._timer.stop()

    def cancel(self) -> None:
        """任务被用户取消：停表并提示（保留已到的百分比作参考）。"""
        self._finished = True
        pct = round(self._fraction() * 100)
        self._step_label.setText(f"{pct}% · 已取消")
        self._eta_label.hide()
        self._timer.stop()

    def reset(self) -> None:
        """清空进度，回到初始状态。"""
        self._finished = False
        self._reached.clear()
        self._duration_s = None
        self._start_ts = time.monotonic()
        self._bar.setValue(0)
        self._step_label.setText("0% · 准备中…")
        self._eta_label.setText("预计剩余：估算中…")
        self._eta_label.show()
        self._timer.start()

    # ---- ETA ----
    def set_video_duration(self, seconds: float) -> None:
        """设置视频时长（秒），启用时长系数 ETA；由 worker 的 info_ready 信号喂入。"""
        if seconds > 0:
            self._duration_s = seconds
            self._refresh_eta()

    def _remaining_seconds(self) -> float | None:
        """剩余秒数：时长系数与耗时外推取较大值；都无依据返回 None。"""
        p = self._fraction()
        elapsed = time.monotonic() - self._start_ts
        estimates: list[float] = []
        if self._duration_s:
            estimates.append(max(self._duration_s * self._ratio - elapsed, 0.0))
        if _EXTRAPOLATE_MIN_P <= p < 1.0:
            estimates.append(elapsed * (1.0 - p) / p)
        if not estimates:
            return None
        return max(estimates)

    def _refresh_eta(self) -> None:
        if self._finished:
            return
        remaining = self._remaining_seconds()
        if remaining is None:
            self._eta_label.setText("预计剩余：估算中…")
        else:
            self._eta_label.setText(f"预计剩余：{_format_duration(remaining)}")
