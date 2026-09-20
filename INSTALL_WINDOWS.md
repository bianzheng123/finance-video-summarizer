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
5. 把 `video_summarize/config_example/video_url.json` 复制为 `video_summarize/config_local/video_url.json`。
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

装好后，需要填两个文件（都在项目根目录或 `video_summarize` 下）：

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

### 2. `video_summarize/config_local/video_url.json`（要总结的视频链接）

用记事本打开，把数组里的链接换成你想总结的 B 站 / YouTube / 抖音视频链接，例如：

```json
[
  "https://www.bilibili.com/video/BVxxxxxxxxxx"
]
```

---

## 五、运行

打开「命令提示符」（或 CodeBuddy 内置终端），执行：

```bash
cd video_summarize
uv run python url_video_summarize.py
```

> 注意：一定要先 `cd video_summarize` 进入该目录再运行，否则会报找不到模块。

运行完成后，结果保存在：

```
video_summarize/summary_data/<平台>/<作者>/<标题>/
```

其中最重要的就是 `summary.md`（金融分析报告），以及 `rendering/` 目录下的 HTML / 公众号 / B 站等格式产物。

---

## 六、常见问题

### 1. 没有生成 summary.pdf 怎么办？

PDF 是「附件形态」的附加产物，需要额外的 `weasyprint` 库（在 Windows 上安装较麻烦，依赖 GTK3 运行时）。**本项目已把它设为可选**——不装也能正常生成 `summary.md`、HTML、公众号、B 站等所有主要产物，只是少了 PDF 这一份。

如果你确实需要 PDF，在项目根目录执行：

```bash
uv sync --extra pdf
```

（这会额外安装 weasyprint 及其 GTK3 依赖，可能需要较长时间。）

### 2. 提示「ffmpeg 找不到」？

说明 ffmpeg 还没装好或没生效。装完后**关闭当前命令行窗口重新打开**再运行。仍不行就手动去 https://ffmpeg.org/download.html 下载，把 `ffmpeg.exe` 所在目录加入系统 PATH。

### 3. YouTube 视频能处理吗？

能识别链接，但 YouTube 需要「翻墙」才能访问。代码里原本用 Linux 的 `proxychains` 工具翻墙，Windows 上没有这个工具，代码会**自动降级为直连**（可能被墙）。所以 B 站 / 抖音没问题，YouTube 取决于你的网络环境。

### 4. WorkBuddy 和 CodeBuddy 有什么区别？

- **CodeBuddy**：AI **代码编辑器**，能打开项目、运行命令、装依赖，适合装代码项目（本教程用它）。
- **WorkBuddy**：AI **办公工作台**，偏办公自动化，不适合装代码项目，本教程不用它。

### 5. 运行报「找不到模块 src.xxx」？

因为你没在 `video_summarize` 目录下运行。先 `cd video_summarize` 再运行即可。
