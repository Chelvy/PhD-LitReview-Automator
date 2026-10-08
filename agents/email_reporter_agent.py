"""
Agent 7: Email Reporter & Archivist Agent

Generates a concise executive summary email for each run with:
  - New papers (included/excluded breakdown)
  - Gap analysis updates
  - Document changes
  - Pipeline problems (errors and warnings)
  - Human action items
  - Research contribution reminder

Sends via Gmail SMTP or SendGrid, archives to local HTML files.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import structlog
from langchain_core.messages import HumanMessage

from config.settings import settings
from state.pipeline_state import PipelineState, EmailReport
from tools.paper_db import PaperDatabase
from tools.email_tool import EmailTool
from agents.llm_utils import build_llm, list_field, message_text, text_field

logger = structlog.get_logger(__name__)

GOOGLE_DOCS_BASE = "https://docs.google.com/document/d"


def _build_llm():
    return build_llm(temperature=0.3, max_tokens=4000, effort="low")


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _generate_contribution_reminder(
    llm: Any,
    synthesis_report: dict[str, Any],
    db: PaperDatabase,
) -> str:
    """
    Generate a fresh contribution reminder statement.
    Emphasizes what our study contributes that the literature still lacks.
    """
    stats = db.get_stats()
    gap_coverage = stats.get("gaps_coverage", {})
    contribution_update = synthesis_report.get("contribution_update", "")

    prompt = f"""Write a 3-4 sentence "Research Contribution Reminder" for the PhD researcher.

Current corpus: {stats.get('total_corpus', 0)} papers analyzed.
Gap coverage: {json.dumps(gap_coverage)}
Synthesis contribution update: {contribution_update}

Research topic: {settings.research_topic}
Research questions:
{chr(10).join(f"  {rq}" for rq in settings.research_questions)}

The reminder should:
1. Note what the literature still lacks (referencing specific gaps still under-covered)
2. Articulate what our study uniquely contributes (the unified theory + SDLC practices)
3. Motivate continued work
4. Be encouraging but analytically grounded

Write only the reminder text, no JSON wrapper."""

    try:
        reminder = message_text(llm.invoke([HumanMessage(content=prompt)]))
        if reminder:
            return reminder
    except Exception as e:
        logger.warning("contribution_reminder_failed", error=str(e)[:300])

    uncovered = [f"GAP-{i}" for i in range(1, 7) if not gap_coverage.get(f"GAP-{i}")]
    return (
        f"{len(uncovered)} of our 6 research gaps ({', '.join(uncovered) or 'none'}) remain unaddressed by the "
        f"{stats.get('total_corpus', 0)}-paper corpus. The field lacks a unified, engineer-usable theory linking "
        f"AI architecture, interface contracts, and operational drift to system-level trust — our core contribution."
    )


def _collect_human_action_items(state: PipelineState) -> list[str]:
    """Collect items that require human attention."""
    items = []

    # Papers whose analysis failed are retried automatically next run
    analysis_errors = state.get("analysis_errors", [])
    if analysis_errors:
        items.append(
            f"{len(analysis_errors)} paper(s) could not be analyzed this run and will be retried "
            f"automatically next run (first error: {analysis_errors[0].get('error', '')[:200]})."
        )

    # Papers that failed extraction were analyzed from the abstract only
    extraction_errors = state.get("extraction_errors", [])
    if extraction_errors:
        items.append(
            f"Full extraction failed for {len(extraction_errors)} paper(s); they were assessed from "
            f"their abstracts only. Manual review recommended."
        )

    # High-relevance papers missing PDFs
    for paper in state.get("included_papers", []):
        if _score(paper.get("relevance_score")) >= 8.5 and not paper.get("pdf_url"):
            items.append(
                f"High-relevance paper '{text_field(paper, 'title', 'Unknown', limit=60)}' "
                f"(score: {_score(paper.get('relevance_score')):.1f}) lacks PDF access. "
                f"Consider manual retrieval."
            )

    # Contradictions found (the LLM may return objects or plain strings)
    for contradiction in list_field(state.get("synthesis_report") or {}, "contradictions"):
        if isinstance(contradiction, dict):
            items.append(
                f"CONTRADICTION detected: {contradiction.get('contradiction', '')} "
                f"→ Resolution needed: {contradiction.get('resolution', 'See synthesis report')}"
            )
        elif contradiction:
            items.append(f"CONTRADICTION detected: {contradiction}")

    # Human review checkpoint
    if state.get("human_review_needed"):
        items.extend(state.get("human_review_reasons", []))

    return items


def _paper_for_email(paper: dict[str, Any]) -> dict[str, Any]:
    """Normalize the fields the email template reads (explicit nulls break Jinja filters)."""
    return {
        "title": text_field(paper, "title", "Untitled"),
        "authors": [str(a) for a in list_field(paper, "authors")],
        "venue": paper.get("venue") or "",
        "year": paper.get("year") or "",
        "relevance_score": _score(paper.get("relevance_score")),
        "critical_notes": paper.get("critical_notes") or "",
        "actual_contribution": paper.get("actual_contribution") or "",
    }


def run_email_reporter(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: Email Reporter & Archivist.
    Generates, sends, and archives the pipeline report.
    """
    run_date = state.get("run_date", datetime.now().strftime("%Y-%m-%d"))
    included_papers = state.get("included_papers", [])
    excluded_papers = state.get("excluded_papers", [])
    discovered_papers = state.get("discovered_papers", [])
    synthesis_report = state.get("synthesis_report", {}) or {}
    lit_review_updates = state.get("lit_review_updates", [])
    proposal_updates = state.get("proposal_updates", [])
    errors = state.get("errors", [])
    warnings = state.get("warnings", [])

    logger.info("email_reporter_start", run_date=run_date, included=len(included_papers))

    llm = _build_llm()
    db = PaperDatabase()
    email_tool = EmailTool()

    # Collect all context for email
    gap_status = {
        gap_id: data for gap_id, data in (synthesis_report.get("gap_status_update") or {}).items()
        if isinstance(data, dict)
    }
    human_action_items = _collect_human_action_items(state)
    contribution_reminder = _generate_contribution_reminder(llm, synthesis_report, db)

    # Document links
    doc_links = {
        "lit_review": f"{GOOGLE_DOCS_BASE}/{settings.litreview_google_doc_id}/edit",
        "proposal": f"{GOOGLE_DOCS_BASE}/{settings.proposal_google_doc_id}/edit",
    }

    # Build email subject
    n_included = len(included_papers)
    n_discovered = len(discovered_papers)
    gaps_updated = sum(
        1 for g in gap_status.values()
        if g.get("today_update") and g.get("today_update") != "No new coverage"
    )
    subject = (
        f"PhD LitReview Update: {n_included} new papers, "
        f"{gaps_updated} gaps updated — {run_date}"
    )
    if errors:
        subject = f"[needs attention] {subject}"

    # Render email
    email_context = {
        "run_date": run_date,
        "topic_short": "Integration Paradox / AI System Reliability",
        "n_discovered": n_discovered,
        "n_included": n_included,
        "n_excluded": len(excluded_papers),
        "included_papers": [_paper_for_email(p) for p in included_papers[:10]],  # Show up to 10 in email
        "gap_status": gap_status,
        "lit_review_updates": lit_review_updates[:5],
        "proposal_updates": proposal_updates[:3],
        "doc_links": doc_links,
        "human_action_items": human_action_items,
        "contribution_reminder": contribution_reminder,
        "total_corpus": db.get_stats().get("total_corpus", 0),
        "errors": errors[:5],
        "warnings": warnings[:5],
    }

    html_body, text_body = email_tool.render_daily_report(email_context)

    # Archive report (dry runs too: reports are gitignored, and CI uploads them as an artifact)
    archive_path = email_tool.save_report_archive(html_body, f"{run_date}_dry_run" if settings.dry_run else run_date)

    # Send email
    email_sent = False
    if not settings.dry_run:
        email_sent = email_tool.send(
            subject=subject,
            html_body=html_body,
            text_body=text_body,
        )
        if email_sent:
            logger.info("email_sent_successfully", subject=subject)
    else:
        logger.info("email_skipped_dry_run", subject=subject)

    # Log pipeline run (once, with the final email status)
    db.log_run({
        "run_date": run_date,
        "run_id": state.get("run_id", ""),
        "papers_discovered": n_discovered,
        "papers_included": n_included,
        "papers_excluded": len(excluded_papers),
        "papers_failed": len(state.get("analysis_errors", [])),
        "gaps_updated": gaps_updated,
        "lit_review_updates": len(lit_review_updates),
        "proposal_updates": len(proposal_updates),
        "email_sent": email_sent,
        "errors": len(errors),
        "dry_run": settings.dry_run,
    })

    email_report: EmailReport = {
        "email_subject": subject,
        "email_html": html_body,
        "email_text": text_body,
        "attachments": [],
        "human_action_required": bool(human_action_items),
        "human_action_items": human_action_items,
        "archive_filename": archive_path,
    }

    logger.info(
        "email_reporter_complete",
        sent=email_sent,
        archived=bool(archive_path),
        action_items=len(human_action_items),
    )

    result: dict[str, Any] = {
        "email_report": email_report,
        "email_sent": email_sent,
    }
    if not settings.dry_run:
        if not settings.has_email:
            result["warnings"] = ["Email is not configured (EMAIL_TO and SMTP or SendGrid credentials) — no report was sent."]
        elif not email_sent:
            result["errors"] = [{
                "node": "email_reporter",
                "error": "The report email could not be sent — check the SMTP/SendGrid credentials (see the run log).",
                "timestamp": datetime.now().isoformat(),
            }]
    return result
