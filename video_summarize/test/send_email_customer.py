"""指定博主名和日期，找到对应 summary_data 目录，给客户发送总结邮件（附 PDF + HTML）。

用法：
    cd video_summarize
    uv run python test/test_email_customer.py                      # 缺省 投机大拿 2026-06-18
    uv run python test/test_email_customer.py 钱博士 2026-05-10
    uv run python test/test_email_customer.py 钱博士 2026-05-10 下午场

参数：
    name      博主名（目录名中"{name}直播录像-"的 name）；缺省 投机大拿
    date      日期，格式 YYYY-MM-DD（如 2026-05-10）；缺省 2026-06-18
    session   场次（可选，适用于一天多场的博主），如"下午场"、"晚上场"；不传则匹配无场次目录

工作流程：
    1. 根据 (name, date, session) 拼出 summary_data 路径
    2. 调 EmailNotifier.send_customer_email：从 config_local/email_info.json 的 customers 中
       筛选「订阅了该博主（bloggers 含 name）且未过期（expire_date >= 今天）」的客户作收件人，
       把 rendering/ 下完整版 summary.pdf / summary.html 重命名为
       「{博主}直播总结-{日期}[ {场次}].(pdf/html)」作为附件发送，标题与正文均标注日期与场次

前置条件：
    - config_local/email_info.json 含 sender（smtp_host/smtp_port/sender/password）
      及 customers（[{"name","email","bloggers":[...],"expire_date":"YYYY-MM-DD"}]）
    - 目标目录下已生成 rendering/summary.pdf 与 summary.html（即先跑过总结渲染）
"""

import argparse
import sys
from pathlib import Path

import _common  # noqa: F401  —— 负责加载 .env 并把 video_summarize/ 加入 sys.path

from src.utils.email import EmailNotifier

_VS_ROOT = Path(__file__).parent.parent
_SUMMARY_DATA_BASE = _VS_ROOT / "summary_data"

# summary_data 下的可能平台目录
_PLATFORMS = ["bilibili", "douyin", "unknown"]


def _find_output_dir(name: str, date: str, session: str | None) -> Path | None:
    """根据 (博主名, 日期, 场次) 找到 summary_data 下对应的录像目录。

    目录命名规则（与 record_live_workflow.py 一致）：
      - 单场：{name}直播录像-{YYYY-MM-DD}
      - 多场：{name}直播录像-{YYYY-MM-DD}-{场次}
    先在各平台 live_record/ 下找，再在 platform/{author}/ 下找。
    """
    suffix = f"{date}-{session}" if session else date
    folder_name = f"{name}直播录像-{suffix}"

    for platform in _PLATFORMS:
        candidate = _SUMMARY_DATA_BASE / platform / "live_record" / folder_name
        if candidate.exists():
            return candidate

    for platform in _PLATFORMS:
        platform_dir = _SUMMARY_DATA_BASE / platform
        if not platform_dir.exists():
            continue
        for author_dir in platform_dir.iterdir():
            if not author_dir.is_dir() or author_dir.name == "live_record":
                continue
            candidate = author_dir / folder_name
            if candidate.exists():
                return candidate
    return None


def main() -> None:
    _common.setup_logging()

    parser = argparse.ArgumentParser(description="指定博主名和日期，给客户发送总结邮件")
    parser.add_argument("name", nargs="?", default="钱博士",
                        help="博主名（如 钱博士、深研一点）；缺省 投机大拿")
    parser.add_argument("date", nargs="?", default="2026-07-28",
                        help="日期，格式 YYYY-MM-DD（如 2026-05-10）；缺省 2026-06-18")
    parser.add_argument("session", nargs="?", default=None,
                        help="场次（可选，如 下午场、晚上场）；不传则匹配无场次目录")
    args = parser.parse_args()

    output_dir = _find_output_dir(args.name, args.date, args.session)
    if output_dir is None:
        suffix = f"{args.date}-{args.session}" if args.session else args.date
        print(f"错误：未找到目录 {args.name}直播录像-{suffix}（已在 summary_data 各平台下搜索）", file=sys.stderr)
        sys.exit(1)

    rendering_dir = output_dir / "rendering"
    pdf_path = rendering_dir / "summary.pdf"
    html_path = rendering_dir / "summary.html"
    if not pdf_path.exists():
        print(f"警告：未找到完整版 PDF: {pdf_path}", file=sys.stderr)
    if not html_path.exists():
        print(f"警告：未找到完整版 HTML: {html_path}", file=sys.stderr)

    date_text = f"{args.date}-{args.session}" if args.session else args.date
    session_label = args.session or ""

    notifier = EmailNotifier()
    if not notifier.config:
        print("错误：未找到 config_local/email_info.json，无法发送邮件", file=sys.stderr)
        sys.exit(1)

    date_str = notifier._extract_date(date_text)
    date_session = f"{date_str} {session_label}" if session_label else date_str

    # 预览将命中的收件人：订阅了该博主且未过期的客户
    targets = [
        c for c in notifier.config.get("customers", [])
        if c.get("email") and args.name in c.get("bloggers", [])
        and not notifier._is_expired(c.get("expire_date"))
    ]
    print(f"目录：{output_dir}")
    print(f"附件命名：{args.name}直播总结-{date_session}.(pdf/html)")
    if targets:
        print("收件人：" + ", ".join(f"{c.get('name', '?')} <{c['email']}>" for c in targets) + "\n")
    else:
        print(f"收件人：无（没有订阅「{args.name}」且未过期的客户）\n")

    ok = notifier.send_customer_email(args.name, date_text, output_dir, session_label)
    if ok:
        print("\n客户邮件发送成功！")
    else:
        print("\n客户邮件发送失败（见上方日志：可能无匹配客户、缺附件或 SMTP 异常）", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
