#!/usr/bin/env bash
# 在 macOS 上打包桌面 GUI（产出 dist/finance-video-summarizer.app 与 dist/*.dmg）
# 用法：bash script/build_mac.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/3] 安装依赖..."
uv sync

echo "[2/3] PyInstaller 打包..."
uv run pyinstaller --clean --noconfirm finance_video_summarizer.spec

echo "[3/3] 打包成 dmg..."
APP="dist/finance-video-summarizer.app"
if [ -d "$APP" ]; then
    hdiutil create -volname "财经视频总结" -srcfolder "$APP" -ov -format UDZO "dist/财经视频总结.dmg"
    echo "打包完成：dist/finance-video-summarizer.app 与 dist/财经视频总结.dmg"
else
    echo "未找到 $APP，请检查上方报错。"
    exit 1
fi
