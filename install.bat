@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ==============================================
echo   finance-video-summarizer 一键安装脚本
echo ==============================================
echo.

REM ---------- 1. 检查 Python ----------
echo [1/6] 检查 Python ...
python --version >nul 2>&1
if errorlevel 1 (
    echo   未检测到 Python，正在通过 winget 自动安装 Python 3.12 ...
    winget install Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo.
        echo   [失败] Python 安装失败。请手动到 https://www.python.org/downloads/ 下载安装，
        echo          安装时务必勾选 "Add Python to PATH"。
        echo   安装完成后，重新双击运行本脚本即可。
        pause
        exit /b 1
    )
    echo.
    echo   Python 安装完成。请关闭本窗口，然后重新双击运行本脚本继续。
    pause
    exit /b 0
)
echo   已找到 Python:
python --version

REM ---------- 2. 检查 uv ----------
echo.
echo [2/6] 检查 uv ...
uv --version >nul 2>&1
if errorlevel 1 (
    echo   未检测到 uv，正在安装 ...
    pip install uv
    if errorlevel 1 (
        echo   [失败] uv 安装失败。请手动在命令行执行： pip install uv
        pause
        exit /b 1
    )
)
echo   已找到 uv:
uv --version

REM ---------- 3. 安装依赖 ----------
echo.
echo [3/6] 安装项目依赖（首次较慢，请耐心等待）...
uv sync
if errorlevel 1 (
    echo   [失败] 依赖安装失败，请检查网络后重新运行本脚本。
    pause
    exit /b 1
)

REM ---------- 4. 检查 ffmpeg ----------
echo.
echo [4/6] 检查 ffmpeg ...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo   未检测到 ffmpeg，正在通过 winget 自动安装 ...
    winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo   [失败] ffmpeg 安装失败。请手动到 https://ffmpeg.org/download.html 下载，
        echo          并把 ffmpeg.exe 所在目录加入系统 PATH 环境变量。
    ) else (
        echo   ffmpeg 安装完成。请关闭本窗口，重新打开后 ffmpeg 才会生效。
    )
) else (
    echo   ffmpeg 已安装
)

REM ---------- 5. 生成视频 URL 配置 ----------
echo.
echo [5/6] 生成视频 URL 配置文件 ...
if not exist "video_summarize\config_local" mkdir "video_summarize\config_local"
if not exist "video_summarize\config_local\video_url.json" (
    copy "video_summarize\config_example\video_url.json" "video_summarize\config_local\video_url.json" >nul
    echo   已生成 video_summarize\config_local\video_url.json，请填入要总结的视频链接
) else (
    echo   已存在，跳过
)

REM ---------- 6. 生成 .env ----------
echo.
echo [6/6] 配置 API 密钥 ...
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo   已生成 .env 文件
)
echo   请用记事本打开 .env，填入以下必填项：
echo     DEEPSEEK_API_KEY       （DeepSeek 大模型密钥）
echo     TENCENT_SECRET_ID      （腾讯云 SecretId）
echo     TENCENT_SECRET_KEY     （腾讯云 SecretKey）
echo     TENCENT_COS_BUCKET     （腾讯云 COS 存储桶名）
echo     TENCENT_COS_REGION     （COS 区域，如 ap-shanghai）
echo     TENCENT_LLM_API_KEY    （腾讯云混元大模型密钥）

echo.
echo ==============================================
echo   安装完成！
echo   下一步：
echo   1) 编辑 .env 填入 API 密钥
echo   2) 编辑 video_summarize\config_local\video_url.json 填入视频链接
echo   3) 运行：
echo        cd video_summarize
echo        uv run python url_video_summarize.py
echo ==============================================
pause
