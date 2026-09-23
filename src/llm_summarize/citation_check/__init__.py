"""引用内容支撑性校验组件（不是步骤，被步骤 5 与 7 内联调用）。

步骤 6 交易技巧**刻意不调用**本组件（权重最低的 section，不值得为它发校验调用）。

prompt / schema 与本模块共置且**不带步骤号**——它们是组件资产，阶段号由调用方
在 `validate(stage_prefix=...)` 时注入（落盘目录与阶段参数随之跟随调用方步骤）。
"""

from .citation_validator import CitationValidator

__all__ = ["CitationValidator"]
