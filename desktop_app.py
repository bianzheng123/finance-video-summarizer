"""桌面 GUI 启动器。

必须先加载 exe 旁 .env（或项目根 .env）再 import ``src.desktop.app``：src 各模块会在
import 时 ``load_dotenv``（路径指向 ``_MEIPASS/.env``，frozen 下不存在 → no-op），先注入
``os.environ`` 才能保证 API Key 就位、不被覆盖。随后 :func:`src.desktop.app.main` 内部会
再兜底调用一次 ``load_app_env``，双保险。
"""

import sys
from pathlib import Path

# 保证 src.* 可导入（开发态可直接运行本脚本；打包后由 PyInstaller 处理 paths）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.desktop.env_loader import load_app_env  # noqa: E402

load_app_env()

from src.desktop.app import main  # noqa: E402  # 延迟 import：确保 env 先就位

if __name__ == "__main__":
    main()
