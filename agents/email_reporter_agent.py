"""
Agent 7: Email Reporter & Archivist Agent

Generates a concise daily executive summary email with:
  - New papers (included/excluded breakdown)
  - Gap analysis updates
  - Document changes
  - Visualizations
  - Human action items
  - Research contribution reminder

Sends via Gmail SMTP or SendGrid, archives to local HTML files.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from config.prompts import AgentPrompts
from state.pipeline_state import PipelineState, EmailReport
from tools.paper_db import PaperDatabase
from tools.email_tool import EmailTool

logger = structlog.get_logger(__name__)

GOOGLE_DOCS_BASE = "https://docs.google.com/document/d"


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0.3,
        max_tokens=4096,
    )


def _generate_contribution_reminder(
    llm: ChatAnthropic,
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
3. Motivate continued daily work
4. Be encouraging but analytically grounded

Write only the reminder text, no JSON wrapper."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return (
            f"Our study addresses {len([g for g, c in gap_coverage.items() if c == 0])} "
            f"gaps still unaddressed by the {stats.get('total_corpus', 0)}-paper corpus. "
            f"The field lacks a unified, engineer-usable theory linking AI architecture, "
            f"interface contracts, and operational drift to system-level trust — our core contribution."
        )


def _collect_human_action_items(state: PipelineState) -> list[str]:
    """Collect items that require human attention."""
    items = []

    # Papers that failed extraction
    for error in state.get("extraction_errors", []):
        items.append(
            f"Extraction failed for paper '{error.get('paper_id', 'unknown')}': "
            f"{error.get('error', '')}. Manual review recommended."
        )

    # High-relevance papers missing PDFs
    for paper in state.get("included_papers", []):
        if paper.get("relevance_score", 0) >= 8.5 and not paper.get("pdf_url"):
            items.append(
                f"High-relevance paper '{paper.get('title', 'Unknown')[:60]}' "
                f"(score: {paper.get('relevance_score', 0):.1f}) lacks PDF access. "
                f"Consider manual retrieval."
            )

    # Contradictions found
    for contradiction in state.get("synthesis_report", {}).get("contradictions", []):
        items.append(
            f"CONTRADICTION detected: {contradiction.get('contradiction', '')} "
            f"→ Resolution needed: {contradiction.get('resolution', 'See synthesis report')}"
        )

    # Human review checkpoint
    if state.get("human_review_needed"):
        items.extend(state.get("human_review_reasons", []))

    return items


def run_email_reporter(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: Email Reporter & Archivist.
    Generates, sends, and archives the daily pipeline report.
    """
    run_date = state.get("run_date", datetime.now().strftime("%Y-%m-%d"))
    included_papers = state.get("included_papers", [])
    excluded_papers = state.get("excluded_papers", [])
    discovered_papers = state.get("discovered_papers", [])
    synthesis_report = state.get("synthesis_report", {})
    lit_review_updates = state.get("lit_review_updates", [])
    proposal_updates = state.get("proposal_updates", [])
    errors = state.get("errors", [])

    logger.info("email_reporter_start", run_date=run_date, included=len(included_papers))

    llm = _build_llm()
    db = PaperDatabase()
    email_tool = EmailTool()

    # Collect all context for email
    gap_status = synthesis_report.get("gap_status_update", {})
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
        f"PhD LitReview Daily Update: {n_included} new papers, "
        f"{gaps_updated} gaps updated — {run_date}"
    )

    # Render email
    email_context = {
        "run_date": run_date,
        "topic_short": "Integration Paradox / AI System Reliability",
        "n_discovered": n_discovered,
        "n_included": n_included,
        "n_excluded": len(excluded_papers),
        "included_papers": included_papers[:10],  # Show up to 10 in email
        "gap_status": gap_status,
        "lit_review_updates": lit_review_updates[:5],
        "proposal_updates": proposal_updates[:3],
        "doc_links": doc_links,
        "human_action_items": human_action_items,
        "contribution_reminder": contribution_reminder,
        "total_corpus": db.get_stats().get("total_corpus", 0),
        "errors": errors[:3],
    }

    html_body, text_body = email_tool.render_daily_report(email_context)

    # Archive report
    archive_path = ""
    if not settings.dry_run:
        archive_path = email_tool.save_report_archive(html_body, run_date)

    # Log pipeline run
    db.log_run({
        "run_date": run_date,
        "run_id": state.get("run_id", ""),
        "papers_discovered": n_discovered,
        "papers_included": n_included,
        "papers_excluded": len(excluded_papers),
        "gaps_updated": gaps_updated,
        "lit_review_updates": len(lit_review_updates),
        "proposal_updates": len(proposal_updates),
        "email_sent": False,  # Will update below
        "errors": len(errors),
    })

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
            # Update run log
            db.log_run({
                "run_date": run_date,
                "email_sent": True,
            })
    else:
        logger.info("email_skipped_dry_run", subject=subject)

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

    return {
        "email_report": email_report,
        "email_sent": email_sent,
    }
