@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==============================================
echo   finance-video-summarizer 一键安装脚本
echo ==============================================
echo.

REM 公共环境准备：Python / uv / 依赖 / ffmpeg（详见 script\setup_env.bat）
call "%~dp0script\setup_env.bat"
if errorlevel 2 exit /b 0
if errorlevel 1 exit /b 1

REM ---------- 生成 .env ----------
echo.
echo [5/5] 配置 API 密钥 ...
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
echo   2) 编辑 config\video_url.json 填入视频链接
echo   3) 运行：
echo        uv run python url_video_summarize.py
echo ==============================================
pause
