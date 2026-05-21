"""
Agent 5: Document Update Agent

Maintains two living PhD documents:
  1. Literature Review (LaTeX/Markdown)
  2. Research Proposal (LaTeX/Markdown)

Takes the synthesis report and produces precise, incremental, analytically
integrated document updates. Does NOT simply append paper summaries — it
synthesizes new content into the existing argumentative structure.

Primary storage: Google Docs (if configured) + local Markdown files (always)
Format: LaTeX for Overleaf export, Markdown for version control
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from config.prompts import AgentPrompts
from state.pipeline_state import PipelineState, DocumentUpdate, BibtexEntry
from tools.paper_db import PaperDatabase
from tools.google_docs import GoogleDocsTool
from agents.llm_utils import build_llm

logger = structlog.get_logger(__name__)

LIT_REVIEW_PATH = "outputs/literature_review.md"
PROPOSAL_PATH = "outputs/research_proposal.md"
BIBTEX_PATH = "outputs/references.bib"
LATEX_LIT_REVIEW_PATH = "outputs/literature_review.tex"


def _build_llm() -> ChatAnthropic:
    return build_llm(temperature=0.2, max_tokens=16000)


def _read_current_document(path: str) -> str:
    """Read the current document content, return empty string if not found."""
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def _write_document(path: str, content: str) -> None:
    """Write document content to file, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    logger.info("document_written", path=path, size=len(content))


def _generate_lit_review_update(
    llm: ChatAnthropic,
    synthesis_report: dict[str, Any],
    included_papers: list[dict[str, Any]],
    current_doc: str,
    run_date: str,
) -> tuple[str, list[DocumentUpdate], list[BibtexEntry]]:
    """
    Generate the updated literature review content.
    Returns: (updated_markdown, list_of_updates, bibtex_entries)
    """
    system = SystemMessage(content=AgentPrompts.DOCUMENT_UPDATE)

    # Prepare paper details for the LLM
    papers_for_update = []
    for p in included_papers:
        papers_for_update.append({
            "paper_id": p.get("paper_id"),
            "title": p.get("title"),
            "authors": p.get("authors", [])[:5],
            "year": p.get("year"),
            "venue": p.get("venue"),
            "relevance_score": p.get("relevance_score"),
            "gaps_addressed": p.get("gaps_addressed", []),
            "actual_contribution": p.get("actual_contribution", ""),
            "wagner_blank_spots": p.get("wagner_analysis", {}).get("blank_spots", []),
            "integration_paradox_evidence": p.get("wagner_analysis", {}).get("integration_paradox_evidence"),
            "critical_notes": p.get("critical_notes", ""),
            "synthesis_tags": p.get("synthesis_tags", []),
            "doi": p.get("doi", ""),
            "bibtex": p.get("bibtex", ""),
        })

    # Section update proposals from synthesis
    sections_to_update = synthesis_report.get("lit_review_sections_to_update", [])
    gap_updates = synthesis_report.get("gap_status_update", {})
    contribution_update = synthesis_report.get("contribution_update", "")

    prompt = f"""Update the PhD Literature Review document for run date: {run_date}

SYNTHESIS REPORT SUMMARY:
Historical Update: {synthesis_report.get('historical_narrative_update', {}).get('new_developments', 'N/A')}
Contribution Update: {contribution_update}
Key Debates Updates: {json.dumps(synthesis_report.get('key_debates_update', [])[:3])}

NEWLY INCLUDED PAPERS ({len(included_papers)}):
{json.dumps(papers_for_update, indent=2)[:8000]}

GAP STATUS:
{json.dumps(gap_updates, indent=2)}

PROPOSED SECTION UPDATES:
{json.dumps(sections_to_update, indent=2)[:3000]}

CURRENT DOCUMENT (first 3000 chars):
{current_doc[:3000]}

Your task:
1. For each proposed section update, generate the actual Markdown content to add/update
2. Produce NEW BibTeX entries for all newly included papers
3. Ensure all content is analytically integrated, not appended
4. Mark new additions with <!-- Updated: {run_date} -->
5. For each GAP that new papers address, update or create the corresponding gap analysis paragraph

CRITICAL REQUIREMENTS:
- Write in academic register (PhD level)
- Every claim must cite a paper
- Focus on analytical synthesis, not summaries
- Apply Wagner blank-spot analysis explicitly in gap discussion
- Strengthen the "necessity of present study" argument using blank spots found

Return a JSON object:
{{
  "document_updates": [
    {{
      "section_path": "2.3.1 Compositional Verification",
      "action": "create_subsection|append_paragraph|update_paragraph",
      "markdown_content": "### 2.3.1 Compositional Verification\\n\\n[Content here...]",
      "latex_content": "\\\\subsection{{Compositional Verification}}\\n\\n[LaTeX content...]",
      "citations_added": ["AuthorYear1", "AuthorYear2"],
      "justification": "Why this update is needed"
    }}
  ],
  "new_citations": [
    {{
      "key": "Author2025SystemReliability",
      "bibtex": "@article{{Author2025SystemReliability,\\n  author = {{...}},\\n  title = {{...}},\\n  year = {{2025}},\\n  journal = {{...}},\\n  doi = {{...}}\\n}}",
      "apa": "Author, F. (2025). Title. Journal, Vol(N), pages."
    }}
  ],
  "version_notes": "Summary of what was updated today"
}}"""

    try:
        response = llm.invoke([system, HumanMessage(content=prompt)])
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        updates: list[DocumentUpdate] = result.get("document_updates", [])
        citations: list[BibtexEntry] = result.get("new_citations", [])

        logger.info("lit_review_update_generated", updates=len(updates), citations=len(citations))
        return updates, citations

    except Exception as e:
        logger.error("lit_review_update_failed", error=str(e))
        return [], []


def _generate_proposal_update(
    llm: ChatAnthropic,
    synthesis_report: dict[str, Any],
    included_papers: list[dict[str, Any]],
    current_doc: str,
    run_date: str,
) -> list[DocumentUpdate]:
    """Generate research proposal updates from synthesis report."""
    system = SystemMessage(content=AgentPrompts.DOCUMENT_UPDATE)

    sections_to_update = synthesis_report.get("proposal_sections_to_update", [])
    if not sections_to_update and not included_papers:
        return []

    prompt = f"""Update the PhD Research Proposal document for run date: {run_date}

NEWLY INCLUDED PAPERS: {len(included_papers)} papers (see synthesis below)
SYNTHESIS INSIGHTS:
- New developments: {synthesis_report.get('historical_narrative_update', {}).get('new_developments', 'N/A')}
- Methodology recommendations: {json.dumps(synthesis_report.get('methodology_recommendations', [])[:5])}
- Contribution update: {synthesis_report.get('contribution_update', 'N/A')}

PROPOSED SECTION UPDATES:
{json.dumps(sections_to_update, indent=2)}

CURRENT PROPOSAL (first 2000 chars):
{current_doc[:2000]}

Update ONLY the sections identified in the proposal updates. Focus on:
1. Strengthening the literature gap argument with new evidence
2. Adding validation approaches inspired by newly found methodologies
3. Updating "Related Work" summary with key new papers
4. Refining "Expected Contributions" based on what we now know the literature lacks

Return JSON:
{{
  "document_updates": [
    {{
      "section_path": "3.2 Research Methodology",
      "action": "strengthen",
      "markdown_content": "...",
      "latex_content": "...",
      "citations_added": ["..."],
      "justification": "..."
    }}
  ]
}}"""

    try:
        response = llm.invoke([system, HumanMessage(content=prompt)])
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
        return result.get("document_updates", [])
    except Exception as e:
        logger.error("proposal_update_failed", error=str(e))
        return []


def _apply_updates_to_markdown(current_doc: str, updates: list[dict], run_date: str) -> str:
    """Apply document updates to Markdown content."""
    if not updates:
        return current_doc

    # Add a "Recent Updates" section at the bottom of the document
    update_block = f"\n\n---\n\n## Recent Updates — {run_date}\n\n"

    for update in updates:
        action = update.get("action", "append_paragraph")
        section = update.get("section_path", "General")
        content = update.get("markdown_content", "")
        justification = update.get("justification", "")

        if not content:
            continue

        update_block += f"<!-- Updated: {run_date} | {action} | {section} -->\n"
        update_block += f"<!-- Justification: {justification} -->\n"
        update_block += content + "\n\n"

    return current_doc + update_block


def _update_bibtex(bibtex_path: str, new_citations: list[dict]) -> None:
    """Append new BibTeX entries to the references file."""
    existing = ""
    if Path(bibtex_path).exists():
        existing = Path(bibtex_path).read_text(encoding="utf-8")

    new_entries = []
    for citation in new_citations:
        key = citation.get("key", "")
        bibtex = citation.get("bibtex", "")
        if bibtex and f"@article{{{key}" not in existing and f"@inproceedings{{{key}" not in existing:
            new_entries.append(bibtex)

    if new_entries:
        updated = existing + "\n\n" + "\n\n".join(new_entries)
        Path(bibtex_path).parent.mkdir(parents=True, exist_ok=True)
        Path(bibtex_path).write_text(updated, encoding="utf-8")
        logger.info("bibtex_updated", new_entries=len(new_entries))


def run_document_update(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: Document Update Agent.
    Applies synthesis findings to living literature review and research proposal.
    """
    synthesis_report = state.get("synthesis_report", {})
    included_papers = state.get("included_papers", [])
    run_date = state.get("run_date", datetime.now().strftime("%Y-%m-%d"))

    if not included_papers and not synthesis_report:
        logger.info("doc_update_skip", reason="no_included_papers_or_synthesis")
        return {
            "lit_review_updates": [],
            "proposal_updates": [],
            "new_citations": [],
            "documents_updated": False,
        }

    if settings.dry_run:
        logger.info("doc_update_dry_run", papers=len(included_papers))
        return {
            "lit_review_updates": [],
            "proposal_updates": [],
            "new_citations": [],
            "documents_updated": False,
        }

    logger.info("doc_update_start", papers=len(included_papers), run_date=run_date)

    llm = _build_llm()
    google_docs = GoogleDocsTool()

    # Read current documents
    current_lit_review = _read_current_document(LIT_REVIEW_PATH)
    current_proposal = _read_current_document(PROPOSAL_PATH)

    # Generate updates
    lit_review_updates, new_citations = _generate_lit_review_update(
        llm, synthesis_report, included_papers, current_lit_review, run_date
    )
    proposal_updates = _generate_proposal_update(
        llm, synthesis_report, included_papers, current_proposal, run_date
    )

    # Apply to local Markdown files
    if lit_review_updates:
        updated_lit_review = _apply_updates_to_markdown(
            current_lit_review, lit_review_updates, run_date
        )
        _write_document(LIT_REVIEW_PATH, updated_lit_review)

    if proposal_updates:
        updated_proposal = _apply_updates_to_markdown(
            current_proposal, proposal_updates, run_date
        )
        _write_document(PROPOSAL_PATH, updated_proposal)

    # Update BibTeX file
    if new_citations:
        _update_bibtex(BIBTEX_PATH, new_citations)

    # Also update database corpus with full BibTeX
    db = PaperDatabase()
    for citation in new_citations:
        key = citation.get("key", "")
        bibtex = citation.get("bibtex", "")
        apa = citation.get("apa", "")
        # Try to match to a paper in corpus by key pattern
        for paper in included_papers:
            paper_id = paper.get("paper_id", "")
            if key and (key[:10] in paper_id or paper_id[:10] in key):
                db.corpus.update(
                    {"bibtex": bibtex, "apa_citation": apa},
                    lambda p: p.get("paper_id") == paper_id
                )

    # Sync to Google Docs (if configured)
    if google_docs.enabled:
        try:
            for update in lit_review_updates:
                content = update.get("markdown_content", "")
                section = update.get("section_path", "")
                if content:
                    google_docs.append_text(
                        settings.litreview_google_doc_id,
                        f"\n[Auto-updated {run_date} — {section}]\n{content}",
                    )
            for update in proposal_updates:
                content = update.get("markdown_content", "")
                section = update.get("section_path", "")
                if content:
                    google_docs.append_text(
                        settings.proposal_google_doc_id,
                        f"\n[Auto-updated {run_date} — {section}]\n{content}",
                    )
            logger.info("google_docs_synced", lit_updates=len(lit_review_updates))
        except Exception as e:
            logger.warning("google_docs_sync_failed", error=str(e))

    documents_updated = bool(lit_review_updates or proposal_updates)
    logger.info(
        "doc_update_complete",
        lit_review_updates=len(lit_review_updates),
        proposal_updates=len(proposal_updates),
        new_citations=len(new_citations),
    )

    return {
        "lit_review_updates": lit_review_updates,
        "proposal_updates": proposal_updates,
        "new_citations": new_citations,
        "documents_updated": documents_updated,
    }
