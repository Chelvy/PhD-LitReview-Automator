#!/usr/bin/env python3
"""
PhD LitReview Automator — Main Entry Point

Usage:
  python main.py run              # Single pipeline run
  python main.py run --dry-run    # Dry run (no email, no doc writes)
  python main.py run --human-checkpoint  # Pause before doc updates
  python main.py schedule         # Start local scheduler (daily at 6 AM UTC)
  python main.py status           # Show database stats
  python main.py export-bibtex    # Export corpus BibTeX
  python main.py test-email       # Send a test email
"""
import argparse
import logging
import os
import sys
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
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if os.getenv("TERM") else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
        ],
    )


# ── Commands ────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    """Execute a single full pipeline run."""
    from config.settings import settings

    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
        console.print("[yellow]DRY RUN MODE — no emails will be sent, no docs will be updated[/yellow]")

    settings.configure_langsmith()

    console.print(Panel(
        f"[bold blue]PhD LitReview Automator[/bold blue]\n"
        f"Research: {settings.research_topic[:80]}...\n"
        f"Model: {settings.anthropic_model} | Days back: {settings.days_lookback}\n"
        f"Dry run: {args.dry_run} | Human checkpoint: {args.human_checkpoint}",
        title="Pipeline Starting",
        border_style="blue",
    ))

    from pipeline.graph import run_pipeline

    try:
        final_state = run_pipeline(
            interrupt_before_docs=args.human_checkpoint
        )

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

        return 0 if not errors else 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red bold]Pipeline failed: {e}[/red bold]")
        return 1


def cmd_schedule(args: argparse.Namespace) -> int:
    """Start the local APScheduler for daily runs."""
    import schedule
    import time
    from config.settings import settings

    configure_logging(settings.log_level)
    console.print(f"[green]Starting scheduler: {settings.cron_schedule} ({settings.timezone})[/green]")

    def scheduled_run():
        console.print(f"[blue]Scheduled run starting: {datetime.now().isoformat()}[/blue]")
        try:
            from pipeline.graph import run_pipeline
            run_pipeline()
        except Exception as e:
            console.print(f"[red]Scheduled run failed: {e}[/red]")

    # Parse cron-like schedule (simplified: support "0 6 * * *" = daily at 6:00)
    cron = settings.cron_schedule.split()
    hour = int(cron[1]) if len(cron) > 1 and cron[1].isdigit() else 6
    minute = int(cron[0]) if cron[0].isdigit() else 0

    schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(scheduled_run)
    console.print(f"[green]Scheduled: daily at {hour:02d}:{minute:02d} UTC[/green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    while True:
        schedule.run_pending()
        time.sleep(60)


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

    # schedule
    sched_parser = subparsers.add_parser("schedule", help="Start local daily scheduler")
    sched_parser.add_argument("--log-level", default="INFO")

    # status
    status_parser = subparsers.add_parser("status", help="Show pipeline status and corpus stats")

    # export-bibtex
    export_parser = subparsers.add_parser("export-bibtex", help="Export corpus as BibTeX")
    export_parser.add_argument("--output", help="Output file path (default: outputs/references_export.bib)")

    # test-email
    subparsers.add_parser("test-email", help="Send a test email")

    args = parser.parse_args()

    # Setup
    Path("logs").mkdir(exist_ok=True)
    log_level = getattr(args, "log_level", "INFO")
    configure_logging(log_level)

    # Route to command
    commands = {
        "run": cmd_run,
        "schedule": cmd_schedule,
        "status": cmd_status,
        "export-bibtex": cmd_export_bibtex,
        "test-email": cmd_test_email,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
