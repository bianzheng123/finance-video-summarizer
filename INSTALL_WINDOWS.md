# Windows 安装教程（零基础版）

> 面向完全没有编程经验的新手。只要会「双击鼠标」和「复制粘贴」，就能装好并跑起来。

## 一、先说结论

**这个项目可以在 Windows 上运行。** 你只需要准备 4 样东西：

| 需要什么 | 是干嘛的 | 难不难 |
|---|---|---|
| Python 3.12 | 运行程序的「底座」 | 简单，自动装 |
| uv | 装依赖的小工具 | 简单，自动装 |
| ffmpeg | 把视频转成音频（字幕识别要用） | 简单，自动装 |
| 几个 API 密钥 | 调用大模型 / 语音识别 | 需要你自己去对应平台申请 |

> **更省事：桌面图形界面** —— 项目已内置一个图形界面程序（打包后是 `finance-video-summarizer.exe`），装好后**双击即可**在窗口里粘贴视频链接一键总结，无需记命令、无需编辑配置文件。详见下方「六、桌面图形界面」。

下面有**两种安装方式**，**强烈推荐第一种**——让 AI（CodeBuddy）帮你装，你基本不用动手。

---

## 二、方式一（推荐）：用 CodeBuddy 让 AI 帮你一键安装

### 第 1 步：安装 CodeBuddy

CodeBuddy 是腾讯出的 AI 代码编辑器，它能在你电脑上帮你运行命令、装环境。

1. 打开浏览器，访问官网：**https://www.codebuddy.ai**（国内可用 https://www.codebuddy.cn）
2. 点击首页的「下载」，选择 **Windows** 版本，下载安装包。
3. 下载完成后，**双击**安装程序（`CodeBuddy Setup x.x.x.exe`）。
4. 安装过程一路点「下一步」即可（勾选「我同意协议」→ 选安装位置 → 下一步 → 完成）。
5. 打开 CodeBuddy，点「登录」，用 **GitHub / Google / 邮箱** 任意一种方式登录（没有账号就按提示注册）。

### 第 2 步：打开本项目

1. 先在 GitHub 上把本项目下载到电脑：打开项目主页，点绿色「Code」按钮 → 「Download ZIP」，下载后解压。
   （如果你会用 git，也可以 `git clone git@github.com:bianzheng123/finance-video-summarizer.git`）
2. 打开 CodeBuddy，点「打开文件夹」（Open Folder），选中刚解压出来的 `finance-video-summarizer` 文件夹。

### 第 3 步：让 AI 一键安装

打开项目后，在 CodeBuddy 的**对话框里粘贴下面这段提示词**，然后回车，AI 就会自动帮你完成安装：

```
请帮我在这台 Windows 电脑上安装并运行这个项目（finance-video-summarizer）。请按顺序完成以下步骤，每完成一步就告诉我结果，遇到问题用中文提示我：

1. 检查是否已安装 Python 3.12。如果没有，用命令 `winget install Python.Python.3.12 --accept-source-agreements --accept-package-agreements` 安装。
2. 检查并安装 uv：如果 `uv --version` 报错，就执行 `pip install uv`。
3. 在项目根目录执行 `uv sync` 安装依赖。
4. 检查并安装 ffmpeg：如果 `ffmpeg -version` 报错，就执行 `winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements`。
5. 编辑 `config/video_url.json`，填入要总结的视频链接。
6. 把 `.env.example` 复制为 `.env`。
7. 最后告诉我：需要在 `.env` 里填哪些 API 密钥、`video_url.json` 里怎么填视频链接、以及运行项目的完整命令。

请一步一步来，不要跳过任何一步。
```

> 说明：如果某一步 AI 需要你确认（比如弹出安装授权），你点「同意 / 是」即可。

### 第 4 步：填密钥并运行

AI 装完后，会提示你填两个文件。填完后再让 AI 帮你跑（或者看下面「四、配置说明」和「五、运行」自己来）。

---

## 三、方式二：手动安装（双击脚本）

如果你不想用 AI，也可以自己双击脚本安装：

1. 下载并解压本项目，进入 `finance-video-summarizer` 文件夹。
2. **双击 `install.bat`**，脚本会自动帮你：检查/安装 Python → 装 uv → 装依赖 → 装 ffmpeg → 生成配置文件。
3. 按脚本最后的中文提示，用记事本打开 `.env` 填入 API 密钥。

> 脚本安装 Python / ffmpeg 用的是 Windows 自带的 `winget`（Win10/11 一般都有）。如果 winget 不可用，脚本会提示你手动去官网下载。

---

## 四、配置说明

装好后，需要填两个地方：

### 1. `.env`（API 密钥，在项目根目录）

用记事本打开 `.env`，把等号后面填上你的真实密钥（**不要加引号、不要有空格**）：

```
DEEPSEEK_API_KEY=你的DeepSeek密钥
TENCENT_SECRET_ID=你的腾讯云SecretId
TENCENT_SECRET_KEY=你的腾讯云SecretKey
TENCENT_COS_BUCKET=你的COS存储桶名
TENCENT_COS_REGION=ap-shanghai
TENCENT_LLM_API_KEY=你的腾讯云混元密钥
```

**必填项**：`DEEPSEEK_API_KEY`、`TENCENT_SECRET_ID`、`TENCENT_SECRET_KEY`、`TENCENT_COS_BUCKET`、`TENCENT_COS_REGION`、`TENCENT_LLM_API_KEY`。

> 密钥从哪来：DeepSeek 密钥去 https://platform.deepseek.com 申请；腾讯云密钥去 https://console.cloud.tencent.com/cam/capi 获取，COS 存储桶在腾讯云对象存储控制台创建。

### 2. `config/video_url.json`（要总结的视频链接）

用记事本打开 `config/video_url.json`，把数组里的链接换成你想总结的 B 站 / YouTube / 抖音视频链接，例如总结钱博士这条直播回放：

```json
[
  "https://www.bilibili.com/video/BV1kCuH6dExG"
]
```

> 链接就是视频页地址栏里那个 `https://www.bilibili.com/video/BVxxxx` 的完整链接，`BV` 后面那串是每个视频唯一的编号（bvid）。

---

## 五、运行

**最简单的方式：直接双击项目根目录下的 `run.bat`**——它已自动把控制台切成 UTF-8，避免中文和报告符号（如 █ ∞ ≈）在 Windows 默认 GBK 控制台下乱码或报错。

如果习惯用命令行，打开「命令提示符」（或 CodeBuddy 内置终端），在项目根目录执行：

```bash
uv run python url_video_summarize.py
```

> 注意：一定要在项目根目录（`finance-video-summarizer`）下运行，否则会报找不到模块。

运行完成后，结果保存在：

```
summary_data/<平台>/<作者>/<标题>/
```

以上面这条钱博士视频（`BV1kCuH6dExG`）为例，目录会长这样：

```
summary_data/bilibili/钱博士直播回放/钱博士_直播回放 2026.8.6/
├── metadata.json                    # 视频元信息
├── recognize_subtitle/              # ASR 字幕
├── llm_analysis/                    # 完整版中间产物
├── llm_analysis_economic/           # 经济版中间产物
├── rendering/                       # 完整版渲染产物（下面四种文件）
│   ├── summary.md                   # Markdown 报告（最重要）
│   ├── summary.html                 # 完整版 HTML
│   ├── summary.pdf                  # PDF
│   └── summary_structured.html      # 结构化折叠式 HTML
└── rendering_economic/              # 经济版渲染产物（同样四种文件）
```

`summary.md` 开头长这样：

```markdown
本文基于B站UP主**钱博士直播回放**的视频《**钱博士_直播回放 2026.8.6**》总结得来，网址：https://www.bilibili.com/video/BV1kCuH6dExG
免责声明：本文仅为内容总结，不构成投资建议。

<h1 align="center">精炼总结</h1>

## 大盘概览
今日市场：小票强于大票、五连阳价格未变、……
宏观事件：刚果金出口禁令、美囤铜支撑铜价、……
后市观点：A股双底概率偏大、……
仓位建议：年限以上不宜低于5成、……
```

> **完整版 vs 经济版**：默认跑**完整版**（七步管线，含个股/主题/交易技巧，观点带引用，产物落 `rendering/`）；加 `--economic` 跑**经济版**（只做宏观分析，产物落 `rendering_economic/`，更快更省 token）。

---

## 六、桌面图形界面（最省事）

不想记命令、不想手动编辑配置文件的，直接用图形界面：

### 运行图形界面

装好环境后（前面「二」或「三」任选一种装完），在项目根目录执行：

```bash
uv sync
uv run python desktop_app.py
```

打开后：

1. **首次启动**会弹出「配置 API Key」窗口，把 DEEPSEEK / 腾讯云密钥填进去点「保存」（保存到项目目录下的 `.env`，以后不用再填）。
2. 在主窗口**粘贴视频链接**（B 站 / YouTube / 抖音），点「开始总结」。
3. 下方日志区**实时显示**当前阶段和日志，耐心等几分钟。
4. 完成后弹窗提示，点「打开完整版 HTML」「打开 PDF」或「打开输出文件夹」用系统默认程序查看。

### 打包成 exe（分发给别人）

如果你想让**没有 Python 环境的人**也能用，双击项目根目录的 `script\build_win.bat`，脚本会自动打包，完成后生成 `dist\finance-video-summarizer.exe`。把这个 exe 发给对方，对方双击运行、首次启动填密钥即可（密钥保存在 exe 旁边的 `.env` 文件里）。

> 桌面版只是换了层「图形界面」皮，内部跑的还是同一套总结管线，产物和命令行完全一样。

---

## 七、常见问题

### 1. 没有生成 summary.pdf 怎么办？

`uv sync` 已默认装齐全部依赖（含 weasyprint），正常情况下 GUI 和命令行都能直接生成 `summary.pdf`。

如果确实没生成：先看运行日志里有没有「PDF导出失败」的告警；weasyprint 在 Windows 上还依赖 GTK3 运行时，若报缺 cairo / pango 库，需按 weasyprint 官方指引补装 GTK3 运行时。

### 2. 提示「ffmpeg 找不到」或「ffprobe 找不到」？

说明 ffmpeg / ffprobe 还没装好或没生效。装完后**关闭当前命令行窗口重新打开**再运行。仍不行就手动去 https://ffmpeg.org/download.html 下载，把 `ffmpeg.exe`、`ffprobe.exe` 所在目录加入系统 PATH。

### 3. YouTube 视频能处理吗？

能识别链接，但 YouTube 需要「翻墙」才能访问。代码里原本用 Linux 的 `proxychains` 工具翻墙，Windows 上没有这个工具，代码会**自动降级为直连**（可能被墙）。所以 B 站 / 抖音没问题，YouTube 取决于你的网络环境。

### 4. WorkBuddy 和 CodeBuddy 有什么区别？

- **CodeBuddy**：AI **代码编辑器**，能打开项目、运行命令、装依赖，适合装代码项目（本教程用它）。
- **WorkBuddy**：AI **办公工作台**，偏办公自动化，不适合装代码项目，本教程不用它。

### 5. 运行报「找不到模块 src.xxx」？

因为你没在项目根目录（`finance-video-summarizer`）下运行。先 `cd` 到项目根目录再运行即可。
