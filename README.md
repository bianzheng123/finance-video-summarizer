# finance-video-summarizer

> 把中文财经视频变成结构化、可溯源的金融分析报告：ASR 字幕识别 + 多阶段 LLM 分析一站式管线。

**Turn Chinese financial videos into structured, citation-backed research reports via ASR transcription and multi-stage LLM analysis.**

给定一个 B 站 / YouTube / 抖音视频链接，自动完成：

1. 视频信息获取（作者 / 标题 / 时长 / 元数据）
2. ASR 语音转写（腾讯云 ASR，支持热词增强）
3. 多阶段 LLM 分析（去噪纠错 → 话题切分 → 宏观 / 交易技巧 / 主题分析）
4. 渲染结构化中文金融报告（`summary.md` + HTML / 公众号 / B 站等多种产物）

核心亮点是**可溯源**：报告中的每个观点、数据、判断都带原文引用区间，可在字幕中逐条核对。

## 特性

- 多平台视频：B 站 / YouTube / 抖音
- 七步分析管线：预处理（去噪、纠错、指代消解、黑话消歧）→ 板块分析（宏观 / 交易 / 主题）
- 引用校验：内置 `citation_check` 组件，确保结论有据可查
- 多产物渲染：Markdown、结构化 HTML、公众号文章、B 站投稿字幕

## 目录结构

```
.
├── pyproject.toml          # 依赖声明（uv 管理）
├── .env.example            # 环境变量模板
└── video_summarize/
    ├── url_video_summarize.py   # 入口：处理 URL 列表
    ├── src/                     # 核心实现
    │   ├── client/              # 多平台视频信息（bilibili/youtube/douyin）
    │   ├── recognize_subtitle/  # ASR 字幕识别
    │   ├── llm_infra/           # LLM 调用层（配置/落盘/重试/并发）
    │   ├── llm_summarize/       # 七步分析管线
    │   ├── rendering/           # 报告渲染
    │   └── ...
    ├── config/                  # 模型/阶段/管线参数
    ├── config_example/          # 配置模板
    ├── database/                # 黑话词表 / 板块映射 / 热词
    ├── assets/                  # 渲染用图片
    └── test/                    # 单元测试（全程 mock 不触网）
```

## 快速开始

### 前置条件

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（依赖管理）

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

### 3. 配置待处理 URL 列表

```bash
cd video_summarize
cp config_example/video_url.json config_local/video_url.json
# 编辑 config_local/video_url.json，填入要总结的视频链接数组
```

### 4. 运行

```bash
cd video_summarize
uv run python url_video_summarize.py
```

处理完成后，产物落在 `video_summarize/summary_data/<platform>/<author>/<title>/` 下，核心报告为 `summary.md`。

> **注意**：入口必须在 `video_summarize/` 目录下运行（`src.*` 与 `script.*` 为相对包导入）。

## 配置说明

| 文件 | 作用 |
|---|---|
| `config/llm_profiles.json` | LLM provider / 模型 / 阶段参数 |
| `config/parameter_config.json` | 管线批大小 / 阈值 |
| `config_local/` | 本地覆盖配置（个人数据，不入库） |
| `database/` | 黑话词表、板块映射、热词（金融领域知识库） |

## 测试

```bash
cd video_summarize
uv run pytest test/
```

测试全程 mock，不触网。

## 许可证

[MIT](LICENSE)
