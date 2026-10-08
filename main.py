#!/usr/bin/env python3
"""
PhD LitReview Automator — Main Entry Point

Usage:
  python main.py run              # Single pipeline run
  python main.py run --dry-run    # Dry run (no email, no doc/database writes)
  python main.py run --human-checkpoint  # Pause before doc updates
  python main.py resume --thread-id ID   # Continue a run paused at the checkpoint
  python main.py schedule         # Start local scheduler (CRON_SCHEDULE, default Mondays 06:17)
  python main.py status           # Show database stats
  python main.py export-bibtex    # Export corpus BibTeX
  python main.py test-email       # Send a test email
  python main.py google-auth      # One-time Google OAuth sign-in (if not using a service account)
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

# ── Configure structured logging ────────────────────────────────────────────

def configure_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    Path("logs").mkdir(exist_ok=True)

    # structlog events are routed through stdlib logging so they reach both stdout
    # and the log file (uploaded as a GitHub Actions artifact).
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"),
        ],
        force=True,
    )
    # httpx logs every request URL at INFO; some carry e-mail addresses or tokens, and
    # GitHub Actions logs of a public repository are public.
    for noisy in ("httpx", "httpcore", "googleapiclient.discovery_cache"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if level <= logging.DEBUG else logging.WARNING)


# ── Commands ────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    """Execute a single full pipeline run."""
    from config.settings import settings

    # settings is instantiated at import time, so the flag must be applied to the
    # object itself (setting the DRY_RUN env var here would have no effect).
    if args.dry_run:
        settings.dry_run = True
    if args.days_lookback:
        settings.days_lookback = args.days_lookback
    if settings.dry_run:
        console.print("[yellow]DRY RUN MODE — no emails will be sent, no docs or databases will be updated[/yellow]")

    if not settings.anthropic_api_key:
        console.print("[red bold]ANTHROPIC_API_KEY is not set — add it to .env (local) or the repository secrets (GitHub Actions).[/red bold]")
        return 2

    settings.configure_langsmith()

    console.print(Panel(
        f"[bold blue]PhD LitReview Automator[/bold blue]\n"
        f"Research: {settings.research_topic[:80]}...\n"
        f"Model: {settings.anthropic_model} (fallback: {settings.anthropic_fallback_model or 'none'}) | "
        f"Days back: {settings.days_lookback}\n"
        f"Dry run: {settings.dry_run} | Human checkpoint: {args.human_checkpoint}",
        title="Pipeline Starting",
        border_style="blue",
    ))

    from pipeline.graph import run_pipeline

    try:
        final_state = run_pipeline(
            interrupt_before_docs=args.human_checkpoint
        )
        return _report_results(final_state)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red bold]Pipeline failed: {e}[/red bold]")
        return 1


def cmd_resume(args: argparse.Namespace) -> int:
    """Continue a run that paused at the human-review checkpoint."""
    from config.settings import settings
    from pipeline.graph import resume_pipeline

    settings.configure_langsmith()
    try:
        return _report_results(resume_pipeline(args.thread_id))
    except Exception as e:
        console.print(f"[red bold]Resume failed: {e}[/red bold]")
        return 1


def _report_results(final_state: dict) -> int:
    """Print the run summary. Exit code 1 means a pipeline stage failed outright."""
    if final_state.get("paused_at"):
        console.print(Panel(
            f"Paused before: {', '.join(final_state['paused_at'])}\n"
            f"{len(final_state.get('included_papers', []))} papers are waiting for document updates.\n"
            f"Review them, then run:\n  python main.py resume --thread-id {final_state['thread_id']}",
            title="Paused for human review",
            border_style="yellow",
        ))
        return 0

    # Display results
    included = len(final_state.get("included_papers", []))
    excluded = len(final_state.get("excluded_papers", []))
    email_sent = final_state.get("email_sent", False)
    errors = final_state.get("errors", [])

    table = Table(title="Pipeline Results", border_style="green")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")
    table.add_row("Papers discovered", str(len(final_state.get("discovered_papers", []))))
    table.add_row("New papers (after dedup)", str(len(final_state.get("new_papers", []))))
    table.add_row("Included in corpus", str(included))
    table.add_row("Excluded", str(excluded))
    table.add_row("Document updates (lit review)", str(len(final_state.get("lit_review_updates", []))))
    table.add_row("Document updates (proposal)", str(len(final_state.get("proposal_updates", []))))
    table.add_row("Email sent", "✓" if email_sent else "✗")
    table.add_row("Errors", str(len(errors)))
    console.print(table)

    if errors:
        console.print("[red]Errors encountered:[/red]")
        for err in errors[:5]:
            console.print(f"  [red]• {err.get('node', 'unknown')}: {err.get('error', '')}[/red]")
    for warning in final_state.get("warnings", [])[:5]:
        console.print(f"  [yellow]• {warning}[/yellow]")

    return 0 if not errors else 1


# (low, high) of each cron field: minute hour day-of-month month day-of-week (0/7 = Sunday)
_CRON_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))


def _cron_field_values(field: str, low: int, high: int) -> set[int]:
    """Expand one cron field: '*', numbers, 'a-b' ranges, ',' lists and '/n' steps."""
    values: set[int] = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, step_text = part.split("/", 1)
            step = int(step_text)
            if step < 1:
                raise ValueError(f"invalid step in cron field {field!r}")
        if part in ("*", ""):
            start, end = low, high
        elif "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
        else:
            start = end = int(part)
        if not low <= start <= end <= high:
            raise ValueError(f"cron field {field!r} is outside {low}-{high}")
        values.update(range(start, end + 1, step))
    return values


def parse_cron(expression: str) -> list[set[int]]:
    """Parse a 5-field cron expression; raises ValueError if any field is invalid."""
    fields = expression.split()
    if len(fields) != 5:
        raise ValueError(f"expected 5 cron fields, got {len(fields)}")
    parsed = [_cron_field_values(f, low, high) for f, (low, high) in zip(fields, _CRON_RANGES)]
    if 7 in parsed[4]:
        parsed[4].add(0)
    return parsed


def cron_matches(expression: str, moment: datetime) -> bool:
    """True if a 5-field cron expression (minute hour day month weekday) fires at `moment`."""
    minutes, hours, days, months, weekdays = parse_cron(expression)
    _, _, day_field, _, weekday_field = expression.split()
    day_ok = moment.day in days
    weekday_ok = (moment.weekday() + 1) % 7 in weekdays  # cron: 0 = Sunday, 1 = Monday
    # Standard cron: when both day fields are restricted, either one matching is enough
    if day_field != "*" and weekday_field != "*":
        date_ok = day_ok or weekday_ok
    else:
        date_ok = day_ok and weekday_ok
    return moment.minute in minutes and moment.hour in hours and moment.month in months and date_ok


def cmd_schedule(args: argparse.Namespace) -> int:
    """Run the pipeline locally on CRON_SCHEDULE in TIMEZONE (an alternative to GitHub Actions)."""
    from zoneinfo import ZoneInfo
    from config.settings import settings

    tz = ZoneInfo(settings.timezone)
    try:
        parse_cron(settings.cron_schedule)
    except ValueError as e:
        console.print(f"[red]Invalid CRON_SCHEDULE {settings.cron_schedule!r}: {e}[/red]")
        return 1
    if not settings.anthropic_api_key:
        console.print("[red bold]ANTHROPIC_API_KEY is not set — add it to .env.[/red bold]")
        return 2
    settings.configure_langsmith()
    console.print(f"[green]Scheduler started: '{settings.cron_schedule}' in {settings.timezone}[/green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    last_fired = None
    while True:
        now = datetime.now(tz).replace(second=0, microsecond=0)
        if now != last_fired and cron_matches(settings.cron_schedule, now):
            last_fired = now
            console.print(f"[blue]Scheduled run starting: {now.isoformat()}[/blue]")
            try:
                from pipeline.graph import run_pipeline
                _report_results(run_pipeline())
            except Exception as e:
                console.print(f"[red]Scheduled run failed: {e}[/red]")
        time.sleep(20)


def cmd_status(args: argparse.Namespace) -> int:
    """Display database and pipeline status."""
    from tools.paper_db import PaperDatabase
    from config.settings import settings

    db = PaperDatabase()
    stats = db.get_stats()
    recent_runs = db.get_recent_runs(5)

    # Stats table
    table = Table(title="Database Status", border_style="cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total papers discovered", str(stats.get("total_discovered", 0)))
    table.add_row("Active corpus (included)", str(stats.get("total_corpus", 0)))
    table.add_row("Excluded papers", str(stats.get("total_excluded", 0)))
    table.add_row("Total pipeline runs", str(stats.get("total_runs", 0)))
    table.add_row("Avg corpus relevance", f"{stats.get('avg_relevance', 0.0):.2f}/10")
    console.print(table)

    # Gap coverage
    gap_table = Table(title="Gap Coverage", border_style="yellow")
    gap_table.add_column("Gap ID")
    gap_table.add_column("Papers", justify="right")
    gap_table.add_column("Status")
    gaps = stats.get("gaps_coverage", {})
    for gap_id in [f"GAP-{i}" for i in range(1, 7)]:
        count = gaps.get(gap_id, 0)
        status = "🔴 Absent" if count == 0 else ("🟡 Nascent" if count < 3 else "🟢 Growing")
        gap_table.add_row(gap_id, str(count), status)
    console.print(gap_table)

    # Recent runs
    if recent_runs:
        runs_table = Table(title="Recent Pipeline Runs", border_style="dim")
        runs_table.add_column("Date")
        runs_table.add_column("Discovered", justify="right")
        runs_table.add_column("Included", justify="right")
        runs_table.add_column("Email", justify="center")
        runs_table.add_column("Errors", justify="right")
        for run in recent_runs[:5]:
            runs_table.add_row(
                run.get("run_date", ""),
                str(run.get("papers_discovered", 0)),
                str(run.get("papers_included", 0)),
                "✓" if run.get("email_sent") else "✗",
                str(run.get("errors", 0)),
            )
        console.print(runs_table)

    return 0


def cmd_export_bibtex(args: argparse.Namespace) -> int:
    """Export full corpus as BibTeX."""
    from tools.paper_db import PaperDatabase

    db = PaperDatabase()
    bibtex = db.export_corpus_bibtex()
    output_path = args.output or "outputs/references_export.bib"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(bibtex, encoding="utf-8")
    console.print(f"[green]BibTeX exported to: {output_path}[/green]")
    return 0


def cmd_google_auth(args: argparse.Namespace) -> int:
    """Run the one-time Google OAuth browser sign-in and save the token (local use only)."""
    from config.settings import settings
    from tools.google_docs import _get_docs_service

    try:
        _get_docs_service()
    except Exception as e:
        console.print(f"[red]Google sign-in failed: {e}[/red]")
        return 1
    if os.path.exists(settings.google_service_account_file):
        console.print(f"[green]Using service account {settings.google_service_account_file} — no sign-in needed[/green]")
    else:
        console.print(f"[green]Signed in; token saved to {settings.google_token_file}[/green]")
    return 0


def cmd_test_email(args: argparse.Namespace) -> int:
    """Send a test email to verify configuration."""
    from tools.email_tool import EmailTool
    from config.settings import settings

    email = EmailTool()
    if not email.enabled:
        console.print("[red]Email not configured. Set SMTP_USER, SMTP_PASSWORD, EMAIL_TO in .env[/red]")
        return 1

    sent = email.send(
        subject=f"PhD LitReview Automator — Test Email ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        html_body="<h1>Test Email</h1><p>PhD LitReview Automator is configured correctly!</p>",
        text_body="Test Email\n\nPhD LitReview Automator is configured correctly!",
    )
    if sent:
        console.print(f"[green]Test email sent to {settings.email_to}[/green]")
        return 0
    else:
        console.print("[red]Failed to send test email. Check SMTP settings.[/red]")
        return 1


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PhD LitReview Automator — Daily academic literature pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Execute a full pipeline run")
    run_parser.add_argument("--dry-run", action="store_true", help="Skip email and doc writes")
    run_parser.add_argument("--human-checkpoint", action="store_true",
                           help="Pause before document updates for human review")
    run_parser.add_argument("--log-level", default="INFO", help="Logging level (INFO/DEBUG/WARNING)")
    run_parser.add_argument("--days-lookback", type=int, default=None,
                           help="Override DAYS_LOOKBACK (how many days back to search)")

    # resume
    resume_parser = subparsers.add_parser("resume", help="Continue a run paused at the human-review checkpoint")
    resume_parser.add_argument("--thread-id", required=True, help="Thread ID printed when the run paused")
    resume_parser.add_argument("--log-level", default="INFO")

    # schedule
    sched_parser = subparsers.add_parser("schedule", help="Start the local scheduler (CRON_SCHEDULE)")
    sched_parser.add_argument("--log-level", default="INFO")

    # status
    status_parser = subparsers.add_parser("status", help="Show pipeline status and corpus stats")

    # export-bibtex
    export_parser = subparsers.add_parser("export-bibtex", help="Export corpus as BibTeX")
    export_parser.add_argument("--output", help="Output file path (default: outputs/references_export.bib)")

    # test-email
    subparsers.add_parser("test-email", help="Send a test email")

    # google-auth
    subparsers.add_parser("google-auth", help="One-time Google OAuth sign-in for Docs sync")

    args = parser.parse_args()
    if getattr(args, "output", None):
        args.output = os.path.abspath(args.output)  # relative to where the command was typed

    # Work from the project directory so .env, storage/ and outputs/ resolve the same
    # way under cron/launchd as in an interactive shell.
    os.chdir(Path(__file__).resolve().parent)

    # Setup
    log_level = getattr(args, "log_level", "INFO")
    configure_logging(log_level)

    # Route to command
    commands = {
        "run": cmd_run,
        "resume": cmd_resume,
        "schedule": cmd_schedule,
        "status": cmd_status,
        "export-bibtex": cmd_export_bibtex,
        "test-email": cmd_test_email,
        "google-auth": cmd_google_auth,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
