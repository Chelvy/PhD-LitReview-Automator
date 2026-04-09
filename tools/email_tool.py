"""
Email tool for sending daily pipeline reports.
Supports: Gmail SMTP, SendGrid API, AWS SES (via SMTP relay).
Generates HTML emails with embedded report and plain-text fallback.
"""
from __future__ import annotations

import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import structlog
from jinja2 import Template

from config.settings import settings

logger = structlog.get_logger(__name__)

# ── HTML email template ─────────────────────────────────────────────────────

EMAIL_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PhD LitReview Daily Update</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f7f7f7; margin: 0; padding: 20px; color: #222; }
  .container { max-width: 750px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  .header { background: #1a237e; color: white; padding: 24px 32px; }
  .header h1 { margin: 0; font-size: 22px; font-weight: 600; }
  .header .subtitle { margin: 6px 0 0; font-size: 13px; opacity: 0.8; }
  .section { padding: 20px 32px; border-bottom: 1px solid #eee; }
  .section:last-child { border-bottom: none; }
  .section-title { font-size: 16px; font-weight: 700; color: #1a237e; margin: 0 0 12px; display: flex; align-items: center; gap: 8px; }
  .badge { display: inline-block; background: #e8eaf6; color: #1a237e; border-radius: 12px; padding: 2px 10px; font-size: 12px; font-weight: 600; }
  .badge.green { background: #e8f5e9; color: #2e7d32; }
  .badge.red { background: #ffebee; color: #c62828; }
  .badge.orange { background: #fff3e0; color: #e65100; }
  .paper-card { background: #f8f9ff; border-left: 3px solid #3f51b5; border-radius: 4px; padding: 12px 14px; margin: 10px 0; }
  .paper-title { font-weight: 600; font-size: 14px; color: #1a237e; margin: 0 0 4px; }
  .paper-meta { font-size: 12px; color: #666; margin: 0 0 6px; }
  .paper-contribution { font-size: 13px; color: #333; margin: 0; }
  .gap-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #f0f0f0; font-size: 13px; }
  .gap-name { font-weight: 600; color: #333; }
  .gap-status { font-size: 12px; }
  .update-item { padding: 6px 0; font-size: 13px; color: #333; }
  .update-item strong { color: #1a237e; }
  .action-item { background: #fff8e1; border-left: 3px solid #ffc107; padding: 10px 14px; border-radius: 4px; margin: 8px 0; font-size: 13px; }
  .contribution-box { background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 6px; padding: 14px; font-size: 14px; color: #1b5e20; }
  .footer { background: #f5f5f5; padding: 14px 32px; font-size: 11px; color: #888; text-align: center; }
  .no-papers { color: #888; font-style: italic; font-size: 13px; padding: 8px 0; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>📚 PhD LitReview Daily Update</h1>
    <div class="subtitle">{{ topic_short }} · {{ run_date }}</div>
  </div>

  <!-- New Papers Section -->
  <div class="section">
    <div class="section-title">
      📄 New Papers
      <span class="badge">{{ n_discovered }} discovered</span>
      <span class="badge green">{{ n_included }} included</span>
      {% if n_excluded > 0 %}<span class="badge red">{{ n_excluded }} excluded</span>{% endif %}
    </div>
    {% if included_papers %}
      {% for paper in included_papers %}
      <div class="paper-card">
        <div class="paper-title">{{ paper.title }}</div>
        <div class="paper-meta">
          {{ paper.authors[:3]|join(', ') }}{% if paper.authors|length > 3 %} et al.{% endif %} ·
          {{ paper.venue or 'Preprint' }} {{ paper.year }} ·
          Relevance: <strong>{{ "%.1f"|format(paper.relevance_score or 0) }}/10</strong>
        </div>
        <div class="paper-contribution">{{ paper.critical_notes or paper.actual_contribution or '' }}</div>
      </div>
      {% endfor %}
    {% else %}
      <div class="no-papers">No new papers met the inclusion criteria today.</div>
    {% endif %}
  </div>

  <!-- Gap Analysis -->
  <div class="section">
    <div class="section-title">🔍 Gap Analysis Update</div>
    {% for gap_id, gap_data in gap_status.items() %}
    <div class="gap-row">
      <span class="gap-name">{{ gap_id }}</span>
      <span class="gap-status">
        {{ gap_data.papers_addressing }} paper(s) ·
        <span class="badge {% if gap_data.trajectory == 'growing' %}green{% elif gap_data.trajectory == 'absent' %}red{% else %}orange{% endif %}">
          {{ gap_data.trajectory }}
        </span>
      </span>
    </div>
    {% if gap_data.today_update %}
    <div style="font-size:12px; color:#555; padding: 3px 0 8px 12px;">↳ {{ gap_data.today_update }}</div>
    {% endif %}
    {% endfor %}
  </div>

  <!-- Document Updates -->
  <div class="section">
    <div class="section-title">📝 Document Updates</div>
    {% if lit_review_updates %}
      {% for upd in lit_review_updates %}
      <div class="update-item">
        <strong>Lit Review {{ upd.section_path }}</strong>: {{ upd.action }} — {{ upd.justification }}
      </div>
      {% endfor %}
    {% else %}
      <div class="no-papers">No document updates today (no new included papers).</div>
    {% endif %}
    {% if doc_links %}
    <div style="margin-top:12px; font-size:13px;">
      📎 <a href="{{ doc_links.lit_review }}" style="color:#3f51b5;">Literature Review</a> ·
      <a href="{{ doc_links.proposal }}" style="color:#3f51b5;">Research Proposal</a>
    </div>
    {% endif %}
  </div>

  <!-- Human Action Required -->
  {% if human_action_items %}
  <div class="section">
    <div class="section-title">⚠️ Action Required</div>
    {% for item in human_action_items %}
    <div class="action-item">{{ item }}</div>
    {% endfor %}
  </div>
  {% endif %}

  <!-- Contribution Reminder -->
  <div class="section">
    <div class="section-title">🧭 Research Contribution Reminder</div>
    <div class="contribution-box">{{ contribution_reminder }}</div>
  </div>

  <div class="footer">
    PhD LitReview Automator · Powered by LangGraph + Claude claude-sonnet-4-6 · {{ run_date }}<br>
    <a href="{{ doc_links.lit_review if doc_links else '#' }}" style="color:#888;">Literature Review</a> ·
    <a href="{{ doc_links.proposal if doc_links else '#' }}" style="color:#888;">Research Proposal</a>
  </div>
</div>
</body>
</html>
"""


class EmailTool:
    """Send daily pipeline report emails via SMTP or SendGrid."""

    def __init__(self) -> None:
        self.enabled = settings.has_email
        if not self.enabled:
            logger.warning("email_disabled", reason="SMTP credentials not configured")

    def send(
        self,
        subject: str,
        html_body: str,
        text_body: str,
        to: Optional[str] = None,
        cc: Optional[str] = None,
    ) -> bool:
        """Send an HTML email with plain-text fallback."""
        if not self.enabled:
            logger.info("email_skipped_dry_run", subject=subject)
            return False

        recipient = to or settings.email_to
        if not recipient:
            logger.error("email_no_recipient")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = settings.email_from
            msg["To"] = recipient
            if cc or settings.email_cc:
                msg["Cc"] = cc or settings.email_cc

            msg.attach(MIMEText(text_body, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            if settings.email_backend == "sendgrid" and settings.sendgrid_api_key:
                return self._send_via_sendgrid(subject, html_body, text_body, recipient)
            else:
                return self._send_via_smtp(msg, recipient)

        except Exception as e:
            logger.error("email_send_failed", error=str(e))
            return False

    def _send_via_smtp(self, msg: MIMEMultipart, recipient: str) -> bool:
        ctx = ssl.create_default_context()
        try:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                server.ehlo()
                server.starttls(context=ctx)
                server.login(settings.smtp_user, settings.smtp_password)
                all_recipients = [r.strip() for r in recipient.split(",")]
                if settings.email_cc:
                    all_recipients += [r.strip() for r in settings.email_cc.split(",")]
                server.sendmail(settings.smtp_user, all_recipients, msg.as_string())
            logger.info("email_sent_smtp", to=recipient)
            return True
        except Exception as e:
            logger.error("smtp_send_failed", error=str(e))
            return False

    def _send_via_sendgrid(
        self, subject: str, html: str, text: str, recipient: str
    ) -> bool:
        try:
            import sendgrid
            from sendgrid.helpers.mail import Mail, Email, To, Content

            sg = sendgrid.SendGridAPIClient(api_key=settings.sendgrid_api_key)
            mail = Mail(
                from_email=Email(settings.smtp_user),
                to_emails=To(recipient),
                subject=subject,
                html_content=Content("text/html", html),
            )
            response = sg.send(mail)
            logger.info("email_sent_sendgrid", status=response.status_code)
            return response.status_code in (200, 202)
        except Exception as e:
            logger.error("sendgrid_send_failed", error=str(e))
            return False

    def render_daily_report(self, context: dict) -> tuple[str, str]:
        """
        Render the HTML and plain-text versions of the daily report.
        Returns: (html_body, text_body)
        """
        template = Template(EMAIL_HTML_TEMPLATE)
        html = template.render(**context)

        # Build simple plain-text fallback
        lines = [
            f"PhD LitReview Daily Update — {context.get('run_date', '')}",
            "=" * 60,
            "",
            f"NEW PAPERS: {context.get('n_discovered', 0)} discovered, "
            f"{context.get('n_included', 0)} included",
            "",
        ]
        for paper in context.get("included_papers", []):
            lines.append(f"  • {paper.get('title', 'Unknown')}")
            lines.append(f"    Relevance: {paper.get('relevance_score', 0):.1f}/10")
            lines.append(f"    {paper.get('critical_notes', '')}")
            lines.append("")

        lines += [
            "GAP ANALYSIS:",
            *[
                f"  {gid}: {gdata.get('papers_addressing', 0)} papers — {gdata.get('trajectory', '')}"
                for gid, gdata in context.get("gap_status", {}).items()
            ],
            "",
            "CONTRIBUTION REMINDER:",
            context.get("contribution_reminder", ""),
            "",
            "---",
            "PhD LitReview Automator | LangGraph + Claude claude-sonnet-4-6",
        ]

        return html, "\n".join(lines)

    def save_report_archive(self, html: str, run_date: str) -> str:
        """Save HTML report to local archive."""
        archive_dir = Path(settings.outputs_dir) / "daily_reports"
        archive_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{run_date}_daily_report.html"
        path = archive_dir / filename
        path.write_text(html, encoding="utf-8")
        logger.info("report_archived", path=str(path))
        return str(path)
