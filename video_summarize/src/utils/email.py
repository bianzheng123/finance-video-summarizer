import json
import logging
import re
import smtplib
from datetime import date, datetime, time, timedelta, timezone
from time import sleep
from pathlib import Path
from collections.abc import Callable
from typing import Optional
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from ..log_config import step_logger

logger = logging.getLogger(__name__)


class EmailNotifier:
    """邮件通知类，统一管理邮件发送功能

    配置文件 config_local/email_info.json 结构：
    {
      "sender":   {"smtp_host", "smtp_port", "sender", "password"},   # 发件账号 + SMTP
      "customers": [{"name", "email", "bloggers": [...], "expire_date": "YYYY-MM-DD"}],
      "staff":     [{"name", "email"}]
    }

    - staff：内部人员，收总结完成通知（无附件）
    - customers：客户，收带总结附件的邮件，仅在「未过期」且「该博主在其 bloggers 列表内」时发送
    """

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        """初始化邮件通知器

        Args:
            config_path: 邮件配置文件路径，默认为 project_root/config_local/email_info.json
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config_local" / "email_info.json"
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Optional[dict]:
        """加载邮件配置文件"""
        try:
            with open(self.config_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("未找到 email_info.json，邮件通知功能将跳过")
            return None
        except Exception as e:
            logger.error("加载邮件配置失败: %s", e)
            return None

    def _refresh_config(self) -> None:
        """热更新：每次发送前重新读取配置文件，无需重启监控进程即可生效。

        读取失败（文件被删、或正写到一半导致 JSON 损坏）时保留上一次的有效配置，
        避免把已有配置误覆盖为空。
        """
        new_config = self._load_config()
        if new_config is not None:
            self.config = new_config

    # 同一封信内部重试次数与两封信之间的最小间隔（秒）。
    # 邮件服务商对「短时间内多次登录/多次投递」会触发 550 Too many attempts，
    # 故所有投递复用同一条连接、同一次登录，并在信件之间留出间隔。
    _SEND_GAP_SECONDS = 3
    _MAX_RETRIES = 3
    _RETRY_BACKOFF_SECONDS = 15

    def _build_message(
        self,
        subject: str,
        body: str,
        recipients: list[str],
        attachments: Optional[list[tuple[Path, str]]] = None,
    ) -> MIMEMultipart:
        """构造一封 MIMEMultipart 邮件（不负责发送）。"""
        sender_cfg = self.config["sender"]
        msg = MIMEMultipart()
        msg["From"] = sender_cfg["sender"]
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        for file_path, display_name in attachments or []:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning("附件不存在，跳过：%s", file_path)
                continue
            with open(file_path, "rb") as fh:
                part = MIMEApplication(fh.read(), Name=display_name)
            # RFC 2231 编码，确保中文文件名在各邮件客户端正确显示
            part.add_header("Content-Disposition", "attachment", filename=display_name)
            msg.attach(part)
        return msg

    def _send_batch(self, messages: list[tuple[list[str], MIMEMultipart]]) -> int:
        """在「同一条 SMTP 连接、同一次登录」下批量投递多封邮件。

        这是规避 550 Too many attempts 的关键：逐个收件人各开一条连接 + 各登录一次，
        会被邮件服务商判定为频繁登录而限流。本方法只登录一次，信件之间留出
        _SEND_GAP_SECONDS 间隔；遇到临时性错误（限流 / 连接被关）时，对当前这封信
        重连并退避重试，最多 _MAX_RETRIES 次。

        Args:
            messages: [(recipients, msg), ...]，每项为一封已构造好的邮件及其收件人

        Returns:
            成功投递的邮件数量
        """
        if not self.config or not messages:
            return 0

        sender_cfg = self.config["sender"]
        sent_count = 0

        def _connect() -> smtplib.SMTP_SSL:
            server = smtplib.SMTP_SSL(sender_cfg["smtp_host"], sender_cfg["smtp_port"])
            server.login(sender_cfg["sender"], sender_cfg["password"])
            return server

        server: Optional[smtplib.SMTP_SSL] = None
        try:
            for idx, (recipients, msg) in enumerate(messages):
                if not recipients:
                    logger.warning("收件人列表为空，跳过发送：%s", msg["Subject"])
                    continue

                # 信件之间留间隔，避免投递过快触发限流（第一封不等待）。
                if idx > 0:
                    sleep(self._SEND_GAP_SECONDS)

                for attempt in range(1, self._MAX_RETRIES + 1):
                    try:
                        if server is None:
                            server = _connect()
                        server.sendmail(sender_cfg["sender"], recipients, msg.as_string())
                        logger.info("邮件已发送至 %s，标题：%s",
                                    ", ".join(recipients), msg["Subject"])
                        sent_count += 1
                        break
                    except Exception as e:
                        # 连接可能已被服务端关闭，下一次重试时重新建连。
                        try:
                            if server is not None:
                                server.quit()
                        except Exception:
                            pass
                        server = None

                        if attempt >= self._MAX_RETRIES:
                            logger.error("邮件发送失败（已重试 %d 次）: %s", attempt, e)
                        else:
                            backoff = self._RETRY_BACKOFF_SECONDS * attempt
                            logger.warning("邮件发送失败（第 %d 次，%s），%d 秒后重试",
                                           attempt, e, backoff)
                            sleep(backoff)
        finally:
            try:
                if server is not None:
                    server.quit()
            except Exception:
                pass

        return sent_count

    def _send_email(
        self,
        subject: str,
        body: str,
        recipients: list[str],
        attachments: Optional[list[tuple[Path, str]]] = None,
    ) -> bool:
        """发送单封邮件（一封信、一组收件人）。

        Args:
            subject: 邮件主题
            body: 邮件正文
            recipients: 收件人邮箱列表
            attachments: 附件列表，每项为 (文件路径, 附件展示文件名)

        Returns:
            是否发送成功
        """
        if not self.config:
            return False

        if not recipients:
            logger.warning("收件人列表为空，跳过发送：%s", subject)
            return False

        msg = self._build_message(subject, body, recipients, attachments)
        return self._send_batch([(recipients, msg)]) > 0

    def _staff_emails(self) -> list[str]:
        """收集所有内部人员（staff）的邮箱。"""
        if not self.config:
            return []
        return [s["email"] for s in self.config.get("staff", []) if s.get("email")]

    @staticmethod
    def _parse_hhmm(value: str) -> Optional[time]:
        """解析 "9:00" / "09:30" / "11:00" 为 time；解析失败返回 None。"""
        m = re.match(r"\s*(\d{1,2}):(\d{2})\s*$", value or "")
        if not m:
            return None
        hh, mm = int(m.group(1)), int(m.group(2))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
        return time(hh, mm)

    @classmethod
    def _in_send_window(cls, start: str, end: str) -> bool:
        """判断当前北京时间（UTC+8）是否落在 [start, end) 发送窗口内。

        - start / end 任一为 "all"（或缺失）→ 全天可发，返回 True。
        - 形如 "9:00" / "11:00"：窗口为左闭右开 [start, end)。
        - 跨零点（start > end，如 22:00-2:00）按跨天处理。
        - 解析失败按「可发」处理（返回 True），避免误拦截。
        """
        if not start or not end or start.strip().lower() == "all" or end.strip().lower() == "all":
            return True
        t_start, t_end = cls._parse_hhmm(start), cls._parse_hhmm(end)
        if t_start is None or t_end is None:
            logger.warning("无法解析 staff 发送窗口 %s-%s，按全天可发处理", start, end)
            return True
        now = datetime.now(timezone(timedelta(hours=8))).time()
        if t_start <= t_end:
            return t_start <= now < t_end
        # 跨零点窗口（如 22:00-02:00）
        return now >= t_start or now < t_end

    def _staff_emails_now(self) -> list[str]:
        """收集当前北京时间处于发送窗口内的 staff 邮箱。

        每个 staff 可配 send_time_start(UTC+8) / send_time_end(UTC+8)，
        值为 "all" 或 "HH:MM"；缺失视为全天可发。
        """
        if not self.config:
            return []
        emails = []
        for s in self.config.get("staff", []):
            if not s.get("email"):
                continue
            start = s.get("send_time_start(UTC+8)", "all")
            end = s.get("send_time_end(UTC+8)", "all")
            if self._in_send_window(start, end):
                emails.append(s["email"])
            else:
                logger.info("staff %s 当前不在发送窗口 %s-%s（北京时间），跳过",
                            s.get("name"), start, end)
        return emails

    def log_customers(self, log_fn: Callable[[str], object] | None = None) -> None:
        """打印 / 记录配置中每个客户的信息（名字、email、bloggers、过期时间），

        供监控服务启动时核对订阅信息是否准确。每个客户附「已过期 / 有效」状态。

        Args:
            log_fn: 输出函数，默认 print；传 logger.info 等可写入日志
        """
        self._refresh_config()
        emit = log_fn or print
        customers = (self.config or {}).get("customers", [])
        if not customers:
            emit("[邮件配置] 未配置任何客户（customers 为空或缺少 email_info.json）")
            return
        emit(f"[邮件配置] 共 {len(customers)} 个客户：")
        for c in customers:
            expire = c.get("expire_date", "")
            status = "已过期" if self._is_expired(expire) else "有效"
            bloggers = "、".join(c.get("bloggers", [])) or "无"
            emit(f"[邮件配置]   - {c.get('name', '?')} <{c.get('email', '无')}> "
                 f"| 订阅: {bloggers} | 到期: {expire or '未设置'}（{status}）")

    def log_staff(self, log_fn: Callable[[str], object] | None = None) -> None:
        """打印 / 记录配置中每个 staff 的信息（名字、email、发送时间窗口），

        供监控服务启动时核对通知配置是否准确。每个 staff 附「当前是否在窗口内」状态。

        Args:
            log_fn: 输出函数，默认 print；传 logger.info 等可写入日志
        """
        self._refresh_config()
        emit = log_fn or print
        staff = (self.config or {}).get("staff", [])
        if not staff:
            emit("[邮件配置] 未配置任何 staff（staff 为空或缺少 email_info.json）")
            return
        emit(f"[邮件配置] 共 {len(staff)} 个 staff（发送时间基准 UTC+8）：")
        for s in staff:
            start = s.get("send_time_start(UTC+8)", "all")
            end = s.get("send_time_end(UTC+8)", "all")
            is_all = str(start).strip().lower() == "all" or str(end).strip().lower() == "all"
            window = "全天" if is_all else f"{start}-{end}"
            now_status = "当前可发" if self._in_send_window(start, end) else "当前不在窗口"
            emit(f"[邮件配置]   - {s.get('name', '?')} <{s.get('email', '无')}> "
                 f"| 发送窗口: {window}（{now_status}）")

    def send_summary_email(self, up_name: str, video_title: str, output_dir: Optional[Path] = None) -> bool:
        """发送视频总结完成邮件（通知 staff）

        对应 monitor_video_workflow.py 中的邮件通知

        Args:
            up_name: UP主名称
            video_title: 视频标题
            output_dir: 输出目录路径

        Returns:
            是否发送成功
        """
        with step_logger("邮件"):
            subject = f"视频总结：{video_title}"
            output_dir_line = ""
            if output_dir:
                output_dir_line = f"总结文件夹：{output_dir}\n总结文件：{output_dir / 'summary.md'}"
            body = f"{up_name} 上传了新视频《{video_title}》，总结已生成。\n\n{output_dir_line}"
            self._refresh_config()
            return self._send_email(subject, body, self._staff_emails())

    def send_completion_email(self, name: str, bvid: Optional[str] = None, date_text: str = "",
                              session_label: str = "", draft_errors: Optional[dict] = None) -> bool:
        """发送直播总结完成、草稿待发布的通知邮件（通知 staff）。

        对应 record_live_workflow.py 中的邮件通知。仅发给当前处于发送窗口
        （send_time_start/end(UTC+8)）内的 staff，正文按收件人姓名个性化。

        Args:
            name: 主播 / 博主名称
            bvid: B站视频ID（保留备用，正文以草稿为主）
            date_text: 含日期的字符串（如 name_suffix），从中提取 YYYY-MM-DD
            session_label: 场次标签（如"下午场"），multiple_live_per_day=true 时给定；空则不显示
            draft_errors: 草稿保存失败信息 {渠道名: 报错}，如 {"B站专栏": "...", "公众号": "..."}；
                          为空表示均成功

        Returns:
            是否至少成功发送给了一个 staff
        """
        with step_logger("邮件"):
            self._refresh_config()
            date_str = self._extract_date(date_text) if date_text else ""
            # "{博主}-{日期}-{场次}"（场次可选）
            title_id = "-".join(p for p in (name, date_str, session_label) if p)
            subject = f"{title_id}直播总结已完成，请发布草稿"

            # 草稿失败信息：有则附在正文末尾提示人工处理
            error_block = ""
            if draft_errors:
                lines = "\n".join(f"  - {ch}：{msg}" for ch, msg in draft_errors.items())
                error_block = f"\n注意：以下渠道草稿保存失败，请手动处理：\n{lines}\n"

            recipients = self._staff_emails_now()
            if not recipients:
                logger.info("当前无处于发送窗口内的 staff，跳过完成通知邮件")
                return False

            # 按收件人姓名个性化，构造所有邮件后复用同一条连接批量发送
            staff_by_email = {s["email"]: s for s in self.config.get("staff", []) if s.get("email")}
            messages: list[tuple[list[str], MIMEMultipart]] = []
            for email in recipients:
                staff_name = staff_by_email.get(email, {}).get("name", "")
                body = (
                    f"亲爱的{staff_name}，\n\n"
                    f"{title_id}直播总结已完成并保存草稿，请进行发布。\n"
                    f"{error_block}\n"
                    f"祝好\n"
                )
                messages.append(([email], self._build_message(subject, body, [email])))
            return self._send_batch(messages) > 0

    def send_recording_only_email(self, name: str, bvid: Optional[str] = None, date_text: str = "",
                                  session_label: str = "", draft_errors: Optional[dict] = None) -> bool:
        """发送「录制已完成、按配置无需总结」的通知邮件（通知 staff）。

        对应 record_live_workflow.py 中 summarize=false 的分支：只上传录像，
        不做 ASR / 字幕 / LLM 总结 / 草稿，因此没有草稿待发布。仅发给当前处于
        发送窗口（send_time_start/end(UTC+8)）内的 staff，正文按收件人姓名个性化。

        Args:
            name: 主播 / 博主名称
            bvid: B站视频ID（保留备用）
            date_text: 含日期的字符串（如 name_suffix），从中提取 YYYY-MM-DD
            session_label: 场次标签（如"下午场"），multiple_live_per_day=true 时给定；空则不显示
            draft_errors: 上传失败信息 {渠道名: 报错}，如 {"B站投稿": "..."}；为空表示成功

        Returns:
            是否至少成功发送给了一个 staff
        """
        with step_logger("邮件"):
            self._refresh_config()
            date_str = self._extract_date(date_text) if date_text else ""
            # "{博主}-{日期}-{场次}"（场次可选）
            title_id = "-".join(p for p in (name, date_str, session_label) if p)
            subject = f"{title_id}录制已完成（配置无需总结）"

            # 失败信息：有则附在正文末尾提示人工处理
            error_block = ""
            if draft_errors:
                lines = "\n".join(f"  - {ch}：{msg}" for ch, msg in draft_errors.items())
                error_block = f"\n注意：以下渠道处理失败，请手动检查：\n{lines}\n"

            recipients = self._staff_emails_now()
            if not recipients:
                logger.info("当前无处于发送窗口内的 staff，跳过录制完成通知邮件")
                return False

            # 按收件人姓名个性化，构造所有邮件后复用同一条连接批量发送
            staff_by_email = {s["email"]: s for s in self.config.get("staff", []) if s.get("email")}
            messages: list[tuple[list[str], MIMEMultipart]] = []
            for email in recipients:
                staff_name = staff_by_email.get(email, {}).get("name", "")
                body = (
                    f"亲爱的{staff_name}，\n\n"
                    f"{name} 的录制已完成，配置文件设置不需要总结，无需发布草稿。\n"
                    f"{error_block}\n"
                    f"祝好\n"
                )
                messages.append(([email], self._build_message(subject, body, [email])))
            return self._send_batch(messages) > 0

    def send_custom_email(self, subject: str, body: str) -> bool:
        """发送自定义邮件（通知 staff）

        Args:
            subject: 邮件主题
            body: 邮件正文

        Returns:
            是否发送成功
        """
        with step_logger("邮件"):
            self._refresh_config()
            return self._send_email(subject, body, self._staff_emails())

    def send_plain(self, subject: str, body: str, recipients: list[str]) -> bool:
        """发送纯文本邮件给任意收件人（订阅通知等）。

        Args:
            subject: 邮件主题
            body: 邮件正文
            recipients: 收件人邮箱列表

        Returns:
            是否发送成功
        """
        with step_logger("邮件"):
            self._refresh_config()
            return self._send_email(subject, body, recipients)

    @staticmethod
    def _extract_date(text: str) -> str:
        """从含日期的字符串中提取并格式化为 YYYY-MM-DD；提取失败则原样返回。

        支持 2026-06-11 / 2026.06.11 / 2026-06-11-下午场 等形式。
        """
        m = re.search(r"(\d{4})[-.](\d{2})[-.](\d{2})", text)
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else text

    @staticmethod
    def _is_expired(expire_date: Optional[str]) -> bool:
        """判断客户订阅是否已过期（expire_date < 今天）。

        expire_date 缺失或无法解析时按「未过期」处理（返回 False），避免误拦截。
        """
        if not expire_date:
            return False
        m = re.search(r"(\d{4})[-.](\d{2})[-.](\d{2})", expire_date)
        if not m:
            logger.warning("无法解析 expire_date=%s，按未过期处理", expire_date)
            return False
        exp = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return date.today() > exp

    def send_customer_email(self, up_name: str, date_text: str, output_dir: Path, session_label: str = "", rendering_dir_name: str = "rendering") -> bool:
        """给订阅了该博主的客户发送总结邮件：附完整版 PDF + HTML + 字幕。

        附件：
          - output_dir/rendering/ 下完整版（未裁剪）summary.pdf / summary.html，
            重命名为「{博主}直播总结-{日期}[-{场次}].(pdf/html)」。
            邮件不受平台正文长度限制，故用完整版而非上传版
            （summary_gongzhonghao.html / summary_bilibili.md 是裁剪后的上传产物）
          - output_dir/recognize_subtitle/subtitle_tencent.txt，
            重命名为「{博主}直播字幕-{日期}[-{场次}].txt」

        对每个客户单独发送，仅当：
          1. 客户未过期（expire_date >= 今天）
          2. 该博主 up_name 在客户的 bloggers 列表内
        标题与正文均标注日期与场次。

        Args:
            up_name: 博主 / 主播名（取自 live_watch_list.json 的 entry["name"]）
            date_text: 含日期的字符串（如 name_suffix），从中提取 YYYY-MM-DD 用于命名
            output_dir: summary_data 下该场次目录（含 rendering/）
            session_label: 场次标签（如"下午场"），multiple_live_per_day=true 时给定；空则不显示

        Returns:
            是否至少成功发送给了一个客户
        """
        with step_logger("邮件"):
            self._refresh_config()
            if not self.config:
                return False

            date_str = self._extract_date(date_text)
            # 日期 + 场次：date_session 用空格（正文/标题展示），date_session_file 用「-」（附件文件名）
            date_session = f"{date_str} {session_label}" if session_label else date_str
            date_session_file = f"{date_str}-{session_label}" if session_label else date_str
            rendering_dir = Path(output_dir) / rendering_dir_name

            attachments: list[tuple[Path, str]] = []
            for ext in ("pdf", "html"):
                src = rendering_dir / f"summary.{ext}"
                if src.exists():
                    attachments.append((src, f"{up_name}直播总结-{date_session_file}.{ext}"))
                else:
                    logger.warning("客户邮件附件缺失，跳过：%s", src)

            # 字幕文件：recognize_subtitle/subtitle_tencent.txt → 「{博主}直播字幕-{日期}-{场次}.txt」
            subtitle_src = Path(output_dir) / "recognize_subtitle" / "subtitle_tencent.txt"
            if subtitle_src.exists():
                attachments.append((subtitle_src, f"{up_name}直播字幕-{date_session_file}.txt"))
            else:
                logger.warning("客户邮件字幕附件缺失，跳过：%s", subtitle_src)

            if not attachments:
                logger.warning("客户邮件无可用附件，取消发送：%s", rendering_dir)
                return False

            # 筛选订阅了该博主且未过期的客户
            targets = []
            for cust in self.config.get("customers", []):
                if not cust.get("email"):
                    continue
                if up_name not in cust.get("bloggers", []):
                    continue
                if self._is_expired(cust.get("expire_date")):
                    logger.info("客户 %s 订阅已于 %s 过期，跳过 %s 的总结邮件",
                                cust.get("name"), cust.get("expire_date"), up_name)
                    continue
                targets.append(cust)

            if not targets:
                logger.info("无订阅 %s 且未过期的客户，跳过客户邮件", up_name)
                return False

            subject = f"{up_name}直播总结-{date_session}"
            messages: list[tuple[list[str], MIMEMultipart]] = []
            for cust in targets:
                body = (
                    f"尊敬的{cust['name']}，\n\n"
                    f"{up_name} {date_session} 的直播总结已完成，详见附件。\n\n"
                    f"附件说明：\n"
                    f"1. HTML 文件可以通过（手机）浏览器打开，方便阅读。\n"
                    f"2. PDF 文件的内容与 HTML 文件相同。\n"
                    f"3. 字幕文件用于定位讲话内容。\n\n"
                    f"祝愿股市赚大钱！\n"
                    f"深研一点团队\n"
                )
                messages.append(([cust["email"]], self._build_message(subject, body, [cust["email"]], attachments)))

            return self._send_batch(messages) > 0
