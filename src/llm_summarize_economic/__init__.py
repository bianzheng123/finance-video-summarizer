"""经济版分析管线：跳过预处理/聚类/主题，直接以原始字幕跑宏观分析 + 板块分析。

与完整版 ``llm_summarize`` 是**完全独立的两个版本**：
- 宏观分析（step1，``step1_macro/``）+ 板块分析（step2，``step2_sector/``）都是
  经济版独立步骤，不复用完整版 ``step5_macro``；步号从 1 开始独立编号，stage key
  用 ``e`` 前缀（``e1_1/e1_2/e2_1/e2_2``）避免与完整版 ``1_1`` 等冲突；
- 两步骤直接对整场字幕按字段组分析，prompt 结构为 ``[DATA]`` 前（完整字幕成为
  KV 缓存稳定前缀），生成阶段 7 字段组共享同一 user_id、完整字幕只预热一次；
- 绑定经济版独立 system prompt（本包根目录 ``shared_system.txt``）。

仍复用全局基础组件（``llm_infra`` / ``utils`` / ``citation_check`` /
``analysis_common``），但分析编排、step 号、system prompt、缓存调度全部独立。
"""

from .economic_analysis import EconomicAnalyzer

__all__ = ["EconomicAnalyzer"]
