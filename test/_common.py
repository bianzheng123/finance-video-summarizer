"""各步骤脚本共用的工具函数。"""

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

# 把项目根目录加入 sys.path，使 src.* 包可直接导入
_vs_root = Path(__file__).parent
if str(_vs_root) not in sys.path:
    sys.path.insert(0, str(_vs_root))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_subtitle_l(input_path: Path) -> list[dict]:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"输入文件应为字幕列表，实际类型：{type(data)}")
    return data


def save_preview(output_dir: Path, step: int, data: dict) -> Path:
    path = output_dir / f"step{step}_result_preview.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  详细结果已保存到 {path}")
    return path


def load_step_cache(output_dir: Path, step: int) -> dict | list | None:
    """读取某步骤的缓存，返回 None 表示不存在。

    step 1 读 `step1_result_preview.json`：第 1-3 步的落盘检查点
    （2_1/2_2/2_3/3_1_*.json）只存各阶段的判定结论，不存装配后的整篇字幕；
    带 annotation 的 `processed_subtitle_l` 只在 step1 的 preview 里，
    而下游 step 2/3/4 要的正是它。
    """
    cache_map = {
        1: (output_dir / "step1_result_preview.json", None),
        2: (output_dir / "4_3_keyword_extraction.json", "结果"),
        3: (output_dir / "4_topic_consolidated.json", None),
    }
    entry = cache_map.get(step)
    if entry is None:
        return None
    path, key = entry
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if key is not None and isinstance(data, dict):
        return data.get(key)
    return data


def require_cache(output_dir: Path, step: int, name: str) -> dict | list:
    """读取前序缓存，不存在则退出并提示先跑对应步骤。"""
    data = load_step_cache(output_dir, step)
    if data is None:
        print(f"错误：找不到 step {step}（{name}）的缓存，请先运行 step{step}_*.py", file=sys.stderr)
        sys.exit(1)
    return data
