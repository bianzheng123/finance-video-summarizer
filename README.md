# finance-video-summarizer

> 把中文财经视频变成结构化、可溯源的金融分析报告：ASR 字幕识别 + 多阶段 LLM 分析一站式管线。

**Turn Chinese financial videos into structured, citation-backed research reports via ASR transcription and multi-stage LLM analysis.**

给定一个 B 站 / YouTube / 抖音视频链接，自动完成：

1. 视频信息获取（作者 / 标题 / 时长 / 元数据）
2. ASR 语音转写（腾讯云 ASR，支持热词增强）
3. 多阶段 LLM 分析（去噪纠错 → 话题切分 → 宏观 / 交易技巧 / 主题分析）
4. 渲染结构化中文金融报告（`summary.md` / `summary.html` / `summary.pdf` / `summary_structured.html`）

核心亮点是**可溯源**：报告中的每个观点、数据、判断都带原文引用区间，可在字幕中逐条核对。

## 特性

- 多平台视频：B 站 / YouTube / 抖音
- 七步分析管线：预处理（去噪、纠错、指代消解、黑话消歧）→ 板块分析（宏观 / 交易 / 主题）
- 引用校验：内置 `citation_check` 组件，确保结论有据可查
- 多产物渲染：Markdown（`summary.md`）、HTML（`summary.html` / `summary_structured.html`）、PDF（`summary.pdf`）

## 两种使用方式

本项目提供两种使用入口，按需选择：

| 入口 | 适合谁 | 怎么用 |
|---|---|---|
| **命令行 / 可编程** | 懂代码、要批量或自动化 | `uv run python url_video_summarize.py --url <视频URL>`（或读 `config/video_url.json` 批量） |
| **桌面 GUI** | 不懂代码的新手 | `uv run python desktop_app.py`，图形界面里粘贴链接一键总结 |

两者共享同一套核心逻辑（`src/summarize_entry.py` 的 `summarize_url`），产物完全一致。

## 目录结构

```
.
├── pyproject.toml          # 依赖声明（uv 管理）
├── .env.example            # 环境变量模板
├── install.bat             # Windows 一键安装脚本
├── INSTALL_WINDOWS.md      # Windows 零基础安装教程
├── url_video_summarize.py  # 命令行入口：处理单个/批量 URL
├── desktop_app.py          # 桌面 GUI 启动器
├── finance_video_summarizer.spec  # PyInstaller 打包配置
├── script/                 # 脚本：费用统计 + Windows/macOS 打包/安装
├── src/                    # 核心实现
│   ├── summarize_entry.py  # 可复用总结入口（CLI 与 GUI 共用）
│   ├── desktop/            # 桌面 GUI（设置页/后台任务/日志面板）
│   ├── client/             # 多平台视频信息（bilibili/youtube/douyin）
│   ├── recognize_subtitle/ # ASR 字幕识别
│   ├── llm_infra/          # LLM 调用层（配置/落盘/重试/并发）
│   ├── llm_summarize/      # 七步分析管线
│   ├── rendering/          # 报告渲染
│   └── ...
├── config/                 # 配置（llm_config.json + video_url.json）
├── database/               # 黑话词表 / 板块映射 / 热词
├── doc/                    # 详细文档（打包 / 费用 / 配置）
└── test/                   # 单元测试（全程 mock 不触网）
```

## Windows 安装（零基础新手）

Windows 用户请看专门的新手教程 → **[INSTALL_WINDOWS.md](INSTALL_WINDOWS.md)**：

- 想省事：装个 CodeBuddy（AI 代码编辑器），把教程里的「一键安装提示词」粘贴给它，AI 自动帮你装好环境。
- 或者：双击仓库里的 `install.bat`，脚本自动装 Python / uv / ffmpeg / 依赖。

## 快速开始

### 前置条件

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（依赖管理）
- ffmpeg（视频转音频，ASR 字幕识别必需；Windows 下 `install.bat` 会自动安装）

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入必填项：
#   - DEEPSEEK_API_KEY          （LLM）
#   - TENCENT_SECRET_ID / KEY   （腾讯云 ASR）
#   - TENCENT_COS_BUCKET / REGION（COS 存储）
#   - TENCENT_LLM_API_KEY       （联网搜索/主题去重）
```

### 3. 配置待处理 URL 列表（可选，仅批量模式用）

编辑 `config/video_url.json`，填入你要总结的链接数组：

```json
[
  "https://www.bilibili.com/video/BV1kCuH6dExG"
]
```

支持 B 站 / YouTube / 抖音视频链接。单链接模式（`--url`）无需配置此项。

### 4. 运行（两种方式任选其一）

**方式 A：命令行**（懂代码 / 要批量或自动化）

单链接总结（直接返回 HTML 文件路径，可直接复制运行试试结果）：

```bash
uv run python url_video_summarize.py --url "https://www.bilibili.com/video/BV15NYu6vEP8/"
# 经济版：uv run python url_video_summarize.py --url <URL> --economic
# 指定输出目录：加 --output-root <目录>
```

批量总结（读 `config/video_url.json`）：

```bash
uv run python url_video_summarize.py
```

**方式 B：桌面 GUI**（不懂代码的新手，图形界面一键总结）

```bash
uv sync
uv run python desktop_app.py
```

首次启动会弹出设置页，在界面里填 DEEPSEEK / 腾讯云密钥即可（无需手动编辑 `.env`），具体见下方「桌面 GUI」一节。

处理完成后，产物落在 `summary_data/<platform>/<author>/<title>/` 下，核心报告为 `summary.md`，同目录下还有 `summary.html`（完整版）、`summary_structured.html`（折叠式）、`summary.pdf`。

> **注意**：入口必须在项目根目录下运行（`src.*` 与 `script.*` 为相对包导入）。

### 运行结果示例

以上面这条钱博士视频（`BV1kCuH6dExG`）为例，跑完后输出目录长这样：

```
summary_data/bilibili/钱博士直播回放/钱博士_直播回放 2026.8.6/
├── metadata.json                    # 视频元信息
├── recognize_subtitle/              # ASR 字幕
├── llm_analysis/                    # 完整版中间产物（七步管线）
├── llm_analysis_economic/           # 经济版中间产物（仅宏观分析）
├── rendering/                       # 完整版渲染产物（下面四种文件）
│   ├── summary.md                   # Markdown 报告
│   ├── summary.html                 # 完整版 HTML
│   ├── summary.pdf                  # PDF（需 weasyprint）
│   └── summary_structured.html      # 结构化折叠式 HTML
└── rendering_economic/              # 经济版渲染产物（同样四种文件）
```

`summary.md` 开头长这样：

```markdown
本文基于B站UP主**钱博士直播回放**的视频《**钱博士_直播回放 2026.8.6**》总结得来，网址：https://www.bilibili.com/video/BV1kCuH6dExG
免责声明：本文仅为内容总结，不构成投资建议。

<h1 align="center">精炼总结</h1>

## 大盘概览
今日市场：小票强于大票、五连阳价格未变、尾盘银行科技同步拉升、煤炭领涨、……
宏观事件：刚果金出口禁令、美囤铜支撑铜价、高温推升煤价、……
后市观点：A股双底概率偏大、二次回落看量能、……
仓位建议：年限以上不宜低于5成、年限以下不宜高于5成、……
```

> **完整版 vs 经济版**：默认跑**完整版**（走七步管线，含个股观点/事件速览、各主题分析、交易技巧，观点逐条带引用，产物落 `rendering/`）；加 `--economic` 跑**经济版**（跳过预处理与板块分析，只用原始字幕做宏观分析，产物落 `rendering_economic/`，更快更省 token）。

## 桌面 GUI

桌面程序（PySide6）面向不懂代码的新手：图形界面里粘贴视频链接 → 一键总结 → 实时显示当前阶段与日志 → 完成后弹窗提示，一键用系统默认程序打开 `summary.html` / `summary.pdf`，或打开输出文件夹。

开发态运行：

```bash
uv sync
uv run python desktop_app.py
```

首次启动会弹出设置页，在界面里填写 DEEPSEEK / 腾讯云密钥并保存到应用目录下的 `.env`（无需手动编辑文件）。`uv sync` 已默认装齐全部依赖（含 weasyprint，PDF 开箱即用）。

## 打包发布

桌面 GUI 可打包成 Windows `.exe`、macOS `.app` + `.dmg`（GUI 默认内置 PDF）。详细步骤、依赖分组与注意事项见 **[doc/打包发布.md](doc/打包发布.md)**。

## 配置说明

配置分 `config/llm_config.json`（LLM 与管线参数）、`config/video_url.json`（待总结链接）、`.env`（密钥）、`database/`（知识库）。字段与作用见 **[doc/配置说明.md](doc/配置说明.md)**。

## 费用说明

一次总结涉及 LLM（DeepSeek，本地有费用报告）、ASR 与 embedding（腾讯云，看控制台账单）三处计费。价格表与查看方式见 **[doc/费用说明.md](doc/费用说明.md)**。

## 测试

```bash
uv run pytest test/
```

测试全程 mock，不触网。

## 许可证

[MIT](LICENSE)
