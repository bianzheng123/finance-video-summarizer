@echo off
chcp 65001 >nul

REM 在 Windows 上打包桌面 GUI（产出 dist\finance-video-summarizer.exe）
REM 用法：在项目根目录双击或命令行运行 script\build_win.bat

REM 公共环境准备（内部执行 uv sync，一把装齐全部依赖）
call "%~dp0setup_env.bat"
if errorlevel 2 exit /b 0
if errorlevel 1 exit /b 1

echo.
echo [打包] PyInstaller 打包...
uv run pyinstaller --clean --noconfirm finance_video_summarizer.spec || goto :error

echo.
echo 打包完成：dist\finance-video-summarizer.exe
exit /b 0

:error
echo 打包失败，请检查上方报错。
exit /b 1
