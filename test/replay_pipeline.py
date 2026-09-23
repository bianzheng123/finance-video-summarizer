"""golden 全管线回放：对已录制日志的源目录重跑完整管线，零 token，对比 llm_analysis.json。

原理：把源视频目录复制到 workdir（llm_output_* 日志目录用 symlink 复用，只读），
设置 LLM_REPLAY=1 后跑 LLMAnalyzer.summarize——所有 LLM 调用从历史日志回放结果，
WSA 搜索一并绕开，全程零网络、零 token。跑完后对比 workdir 的 llm_analysis.json
与源的 llm_analysis.json：未改代码时应完全一致（基线验证回放保真），
改代码后的差异即本次改动的影响面。

用法（必须在项目根目录下运行）：
    uv run python test/replay_pipeline.py --source bilibili/钱博士直播回放/钱博士_直播回放 2026.8.6
    uv run python test/replay_pipeline.py --source <相对summary_data视频目录> --diff      # 不一致时打印结构差异
    uv run python test/replay_pipeline.py --source <相对summary_data视频目录> --render   # 再渲染并对比 md 产物

退出码：0 = llm_analysis.json 与源完全一致；2 = 不一致；1 = 运行错误（含 ReplayMissError）。

注意：golden 由"录制时运行的代码"决定——改了 prompt/schema/log_name/调用集合后，
旧日志可能无法命中（ReplayMissError），需要先跑一次 live 录制刷新 golden。

未命中的两类常见原因（都不是代码 bug，先排除再怀疑改动）：
  1. **log_name 改名**：录制用 `A1_粗筛_b0` / `7_2_..._b0_attempt1`，现行代码问
     `2_1_粗筛_b0` / `7_2_..._b0`。比对办法 `git show <录制commit>:./src/... | grep log_name=`；
     零 token 续用办法是在 `llm_output_N/` 下给旧目录建同名符号链接。
     引用校验（5_3/7_2）尤其要注意：批量名失配 → 被当成整批调用失败 → 拆回单跑问
     `_s{idx}`，报错落在 `_s{idx}` 上，真正的失配点在批量名。
     另：交易技巧的校验（旧 6_2）已整体取消，旧 golden 里那批日志现行代码不再问。
  2. **录制时带断点跑**（pending ⊊ 全量）：`b{idx}` 是 pending 排序后的序号，
     少一个单元后面全部左移。本仓 golden 的 5_3 就只录了 10 批（录制前已有 3 个主题
     落了 checkpoint），全量 13 主题回放必然缺 b10~b12——只能重录 golden 才能消除。
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# 必须在任何 src 导入之前设置：replay 开关决定 LLMClient 的行为。
# 不要把它写进 .env——client.py 的 load_dotenv(override=True) 会覆盖此值。
os.environ["LLM_REPLAY"] = "1"

from src.llm_analysis import LLMAnalyzer  # noqa: E402
from url_video_summarize import read_subtitle_file_to_dict  # noqa: E402

_SUMMARY_DATA = _ROOT / "summary_data"
_DIFF_LIMIT = 50  # --diff 最多打印的差异条数
_DIFF_VALUE_LIMIT = 120  # 差异值截断长度（引用原文动辄数千字，全量打印不可读）
_TEXT_RENDER_SUFFIXES = (".md", ".html")  # --render 只对比文本产物，跳过 png/pdf
# 历史 golden 渲染目录中的旧文件名 → 现行文件名（sanitize 机制与 mobile 命名已删除/改名）
# 注意：完整版 / 上传版拆分后产物集也变了——旧 golden 没有 summary.html，且
# summary_gongzhonghao.html / summary_bilibili.md 由完整版变成裁剪版。对旧 golden 跑
# --render 必然报这几处差异，属预期（渲染差异不影响退出码）。
_LEGACY_RENDER_NAMES = {"summary_mobile.html": "summary_structured.html"}


def _short(v: object) -> str:
    """差异值截断：长文本只留头尾片段。"""
    s = repr(v)
    if len(s) <= _DIFF_VALUE_LIMIT:
        return s
    return f"{s[:_DIFF_VALUE_LIMIT]}…({len(s)} 字符)"


def _prepare_workdir(src: Path, workdir: Path) -> None:
    """复制源目录到 workdir，llm_output_* 日志目录用 symlink 复用（replay 只读）。

    workdir/llm_analysis 只放 symlink，各步 checkpoint 天然不存在 → 全管线重跑；
    其余文件（字幕、incremental_database、metadata.json）用真实副本，
    因为步骤 2 会原子回写字幕、步骤 2/3 会写 incremental_database，不能污染源。
    """
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(
        src, workdir, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("llm_analysis", "rendering", "workflow.log"),
    )
    la = workdir / "llm_analysis"
    la.mkdir()
    for d in sorted((src / "llm_analysis").iterdir()):
        if d.name.startswith("llm_output_") and d.is_dir():
            os.symlink(d.resolve(), la / d.name, target_is_directory=True)


def _find_subtitle(workdir: Path) -> Path:
    """按约定探测字幕文件：recognize_subtitle/subtitle_tencent.txt → subtitle.txt。"""
    for candidate in (
        workdir / "recognize_subtitle" / "subtitle_tencent.txt",
        workdir / "subtitle.txt",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{workdir} 下未找到字幕文件（subtitle_tencent.txt / subtitle.txt）")


def _diff_dicts(old: dict, new: dict, path: str = "") -> list[str]:
    """递归比较两个 dict，返回 "路径: 旧值 -> 新值" 的差异列表。"""
    diffs: list[str] = []
    for key in sorted(set(old) | set(new)):
        p = f"{path}.{key}" if path else str(key)
        if key not in old:
            diffs.append(f"{p}: (源无) -> {_short(new[key])}")
        elif key not in new:
            diffs.append(f"{p}: {_short(old[key])} -> (新无)")
        elif isinstance(old[key], dict) and isinstance(new[key], dict):
            diffs.extend(_diff_dicts(old[key], new[key], p))
        elif old[key] != new[key]:
            diffs.append(f"{p}: {_short(old[key])} -> {_short(new[key])}")
    return diffs


def _compare_renderings(src: Path, workdir: Path) -> list[str]:
    """对比文本渲染产物（.md/.html），返回差异行，不影响退出码。"""
    diffs: list[str] = []
    src_dir, dst_dir = src / "rendering", workdir / "rendering"
    if not src_dir.exists() or not dst_dir.exists():
        return ["渲染目录缺失，跳过对比"]
    for f in sorted(src_dir.iterdir()):
        if not f.suffix.lower() in _TEXT_RENDER_SUFFIXES:
            continue
        if "_sanitize." in f.name:
            continue  # sanitize 机制已删除，历史 golden 的 sanitize 产物不再产出
        counterpart = dst_dir / _LEGACY_RENDER_NAMES.get(f.name, f.name)
        if not counterpart.exists():
            diffs.append(f"{f.name}: 源有、新无")
            continue
        if f.read_text(encoding="utf-8") != counterpart.read_text(encoding="utf-8"):
            diffs.append(f"{f.name}: 内容不一致")
    return diffs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--source", required=True, help="相对 summary_data 的视频目录")
    parser.add_argument("--workdir", default=None, help="工作目录（默认 summary_data/_replay/<目录名>）")
    parser.add_argument("--diff", action="store_true", help="llm_analysis.json 不一致时打印结构差异")
    parser.add_argument("--render", action="store_true", help="跑完后渲染并对比 md/html 产物")
    args = parser.parse_args(argv)

    src = (_SUMMARY_DATA / args.source).resolve()
    if not src.is_dir():
        print(f"错误：源目录不存在：{src}", file=sys.stderr)
        return 1
    workdir = (args.workdir and Path(args.workdir).resolve()) or (_SUMMARY_DATA / "_replay" / src.name).resolve()
    if workdir == src:
        print("错误：workdir 不能与源目录相同", file=sys.stderr)
        return 1

    _prepare_workdir(src, workdir)
    print(f"workdir: {workdir}")

    subtitle_file = _find_subtitle(workdir)
    subtitle_l = read_subtitle_file_to_dict(subtitle_file)
    print(f"字幕 {len(subtitle_l)} 行，开始回放（LLM_REPLAY=1，零 token）...")

    analysis = LLMAnalyzer().summarize(workdir, subtitle_l)

    new_path = workdir / "llm_analysis" / "llm_analysis.json"
    old_path = src / "llm_analysis" / "llm_analysis.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    diffs = _diff_dicts(old, new)
    if diffs:
        print(f"llm_analysis.json 不一致：{len(diffs)} 处差异", file=sys.stderr)
        if args.diff:
            for line in diffs[:_DIFF_LIMIT]:
                print(f"  {line}", file=sys.stderr)
            if len(diffs) > _DIFF_LIMIT:
                print(f"  ... 其余 {len(diffs) - _DIFF_LIMIT} 处省略", file=sys.stderr)
    else:
        print("llm_analysis.json 与源完全一致 ✓")

    if args.render:
        # 渲染模块较重，只在需要时导入
        from src.rendering_controller import SummaryRenderer

        meta = json.loads((workdir / "metadata.json").read_text(encoding="utf-8"))

        SummaryRenderer().render_save(
            workdir, analysis, author=meta.get("author"), title=meta.get("title"),
        )
        render_diffs = _compare_renderings(src, workdir)
        if render_diffs:
            print("渲染产物差异（不影响退出码）：", file=sys.stderr)
            for line in render_diffs:
                print(f"  {line}", file=sys.stderr)
        else:
            print("渲染产物（md/html）与源一致 ✓")

    return 0 if not diffs else 2


if __name__ == "__main__":
    sys.exit(main())
