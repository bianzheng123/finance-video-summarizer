@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

REM ==============================================
REM  公共环境准备：Python / uv / 依赖 / ffmpeg
REM  参数 %* 透传给 uv sync（默认无参数，一把装齐全部依赖）
REM  返回码：0=成功  1=失败  2=Python 刚装好需重启
REM ==============================================

echo ==============================================
echo   finance-video-summarizer 环境准备
echo ==============================================
echo.

REM ---------- 1. 检查 Python ----------
echo [1/4] 检查 Python ...
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
    exit /b 2
)
echo   已找到 Python:
python --version

REM ---------- 2. 检查 uv ----------
echo.
echo [2/4] 检查 uv ...
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
echo [3/4] 安装项目依赖（首次较慢，请耐心等待）...
uv sync %*
if errorlevel 1 (
    echo   [失败] 依赖安装失败，请检查网络后重新运行本脚本。
    pause
    exit /b 1
)

REM ---------- 4. 检查 ffmpeg / ffprobe ----------
echo.
echo [4/4] 检查 ffmpeg / ffprobe ...
set "NEED_INSTALL=0"
ffmpeg -version >nul 2>&1
if errorlevel 1 set "NEED_INSTALL=1"
ffprobe -version >nul 2>&1
if errorlevel 1 set "NEED_INSTALL=1"

if "%NEED_INSTALL%"=="1" (
    echo   未检测到 ffmpeg 或 ffprobe，正在通过 winget 自动安装 ...
    winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo   [失败] 安装失败。请手动到 https://ffmpeg.org/download.html 下载，
        echo          并把 ffmpeg.exe、ffprobe.exe 所在目录加入系统 PATH 环境变量。
    ) else (
        echo   安装完成。请关闭本窗口，重新打开后 ffmpeg / ffprobe 才会生效。
    )
) else (
    echo   ffmpeg 与 ffprobe 均已安装
)

exit /b 0
