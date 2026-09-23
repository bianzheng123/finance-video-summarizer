@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ==============================================
echo   finance-video-summarizer
echo ==============================================
echo.

REM 全程以 UTF-8 输出，避免中文 / 报告符号（如 █ ∞ ≈）在控制台乱码或报错
set PYTHONUTF8=1

echo 正在运行（首次会先加载依赖，请耐心等待）...
echo.
uv run python url_video_summarize.py

echo.
echo ==============================================
echo   运行结束。结果见 summary_data 目录。
echo ==============================================
pause
