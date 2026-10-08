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

Corpus papers carry a `documented` flag. A paper is marked documented only after
the LLM update covering it succeeded, so a failed update is retried next run.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from config.prompts import AgentPrompts
from state.pipeline_state import PipelineState, DocumentUpdate
from tools.bibtex import build_entries, existing_keys
from tools.paper_db import PaperDatabase
from tools.google_docs import GoogleDocsTool
from agents.llm_utils import build_llm, invoke_json, list_field, text_field

logger = structlog.get_logger(__name__)

LIT_REVIEW_FILE = "literature_review.md"
PROPOSAL_FILE = "research_proposal.md"
BIBTEX_FILE = "references.bib"
LATEX_UPDATES_FILE = "literature_review_updates.tex"

# Papers per literature-review LLM call; keeps every paper's details in the prompt.
PAPERS_PER_UPDATE = 6

# The document's own References section: a top-level (# or ##) heading. Updates go before
# the last one, so a "### References" list inside an earlier update block is never used.
_REFERENCES_HEADING_RE = re.compile(r"^#{1,2}\s+(?:\d+\.\s*)?References\b.*$", re.MULTILINE | re.IGNORECASE)
_HEADING_RE = re.compile(r"^#{1,4}\s+.+$", re.MULTILINE)
_RECENT_UPDATE_RE = re.compile(r"^## Recent Updates — .+$", re.MULTILINE)


def _output_path(filename: str) -> Path:
    return Path(settings.outputs_dir) / filename


def _build_llm():
    return build_llm(temperature=0.2, max_tokens=20000, effort="high")


def _read_current_document(path: Path) -> str:
    """Read the current document content, return empty string if not found."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _write_document(path: Path, content: str) -> None:
    """Write document content to file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("document_written", path=str(path), size=len(content))


def _document_outline(doc: str) -> str:
    """Headings plus the last few update stamps, so the LLM knows the structure and what exists."""
    headings = _HEADING_RE.findall(doc)
    recent = _RECENT_UPDATE_RE.findall(doc)[-3:]
    outline = "\n".join(headings[:80]) or "(empty document)"
    if recent:
        outline += "\n\nMOST RECENT AUTOMATED UPDATES:\n" + "\n".join(recent)
    return outline


def _updates_from(result: Any, *keys: str) -> list[DocumentUpdate]:
    """Accept the update list under any of the given keys (system vs user prompt wording)."""
    if isinstance(result, list):
        return [u for u in result if isinstance(u, dict)]
    if not isinstance(result, dict):
        return []
    for key in keys:
        value = result.get(key)
        if isinstance(value, list):
            return [u for u in value if isinstance(u, dict)]
    return []


def _paper_for_prompt(p: dict[str, Any]) -> dict[str, Any]:
    wagner = p.get("wagner_analysis") if isinstance(p.get("wagner_analysis"), dict) else {}
    return {
        "citation_key": p.get("citation_key"),
        "title": p.get("title"),
        "authors": list_field(p, "authors")[:5],
        "year": p.get("year"),
        "venue": p.get("venue"),
        "relevance_score": p.get("relevance_score"),
        "gaps_addressed": list_field(p, "gaps_addressed"),
        "actual_contribution": text_field(p, "actual_contribution", "", limit=600),
        "wagner_blank_spots": list_field(wagner, "blank_spots")[:4],
        "integration_paradox_evidence": wagner.get("integration_paradox_evidence"),
        "critical_notes": text_field(p, "critical_notes", "", limit=600),
        "synthesis_tags": list_field(p, "synthesis_tags"),
    }


def _generate_lit_review_update(
    llm: Any,
    synthesis_report: dict[str, Any],
    papers: list[dict[str, Any]],
    current_doc: str,
    run_date: str,
) -> list[DocumentUpdate]:
    """Generate literature review updates for one batch of papers. Raises on LLM/parse failure."""
    system = SystemMessage(content=AgentPrompts.DOCUMENT_UPDATE)

    historical = synthesis_report.get("historical_narrative_update") or {}
    new_developments = historical.get("new_developments", "N/A") if isinstance(historical, dict) else historical

    prompt = f"""Update the PhD Literature Review document for run date: {run_date}

SYNTHESIS REPORT SUMMARY:
Historical Update: {new_developments}
Contribution Update: {synthesis_report.get('contribution_update', '')}
Key Debates Updates: {json.dumps(list_field(synthesis_report, 'key_debates_update')[:3], ensure_ascii=False)}

PAPERS TO INTEGRATE ({len(papers)}), each with the citation key to use:
{json.dumps([_paper_for_prompt(p) for p in papers], indent=2, ensure_ascii=False)}

GAP STATUS:
{json.dumps(synthesis_report.get('gap_status_update', {}), indent=2, ensure_ascii=False)}

PROPOSED SECTION UPDATES (from synthesis):
{json.dumps(list_field(synthesis_report, 'lit_review_sections_to_update'), indent=2, ensure_ascii=False)[:4000]}

CURRENT DOCUMENT OUTLINE:
{_document_outline(current_doc)}

Your task:
1. Generate the actual Markdown (and matching LaTeX) content that integrates these papers
   into the relevant sections of the outline — synthesize, do not just summarize.
2. Do not repeat content already covered by earlier automated updates.
3. For each GAP these papers address, add analysis to the corresponding gap discussion,
   applying the Wagner blank-spot framework explicitly.
4. Cite every claim with the papers' citation keys ([@Key] in Markdown, \\citep{{Key}} in LaTeX).

Return JSON in the OUTPUT FORMAT from your instructions: an object with a
"document_updates" list (section_path, action, markdown_content, latex_content,
citations_added, justification) and "version_notes"."""

    result = invoke_json(llm, [system, HumanMessage(content=prompt)])
    updates = _updates_from(result, "document_updates", "literature_review_updates")
    logger.info("lit_review_update_generated", papers=len(papers), updates=len(updates))
    return updates


def _generate_proposal_update(
    llm: Any,
    synthesis_report: dict[str, Any],
    papers: list[dict[str, Any]],
    current_doc: str,
    run_date: str,
) -> list[DocumentUpdate]:
    """Generate research proposal updates from synthesis report. Raises on LLM/parse failure."""
    system = SystemMessage(content=AgentPrompts.DOCUMENT_UPDATE)

    sections_to_update = list_field(synthesis_report, "proposal_sections_to_update")
    if not sections_to_update and not papers:
        return []

    historical = synthesis_report.get("historical_narrative_update") or {}
    new_developments = historical.get("new_developments", "N/A") if isinstance(historical, dict) else historical
    paper_list = [
        {"citation_key": p.get("citation_key"), "title": p.get("title"), "year": p.get("year"),
         "actual_contribution": text_field(p, "actual_contribution", "", limit=300)}
        for p in papers
    ]

    prompt = f"""Update the PhD Research Proposal document for run date: {run_date}

NEWLY INCLUDED PAPERS ({len(papers)}), with citation keys:
{json.dumps(paper_list, indent=2, ensure_ascii=False)[:8000]}

SYNTHESIS INSIGHTS:
- New developments: {new_developments}
- Methodology recommendations: {json.dumps(list_field(synthesis_report, 'methodology_recommendations')[:5], ensure_ascii=False)}
- Contribution update: {synthesis_report.get('contribution_update', 'N/A')}

PROPOSED SECTION UPDATES:
{json.dumps(sections_to_update, indent=2, ensure_ascii=False)[:4000]}

CURRENT PROPOSAL OUTLINE:
{_document_outline(current_doc)}

Update ONLY the sections identified in the proposal updates. Focus on:
1. Strengthening the literature gap argument with new evidence
2. Adding validation approaches inspired by newly found methodologies
3. Updating "Related Work" summary with key new papers
4. Refining "Expected Contributions" based on what we now know the literature lacks

Return JSON in the OUTPUT FORMAT from your instructions (a "document_updates" list)."""

    result = invoke_json(llm, [system, HumanMessage(content=prompt)])
    return _updates_from(result, "document_updates", "research_proposal_updates")


def _apply_updates_to_markdown(current_doc: str, updates: list[dict], run_date: str) -> str:
    """Insert a dated update block before the References section (or at the end)."""
    blocks = []
    for update in updates:
        content = update.get("markdown_content") or ""
        if not content:
            continue
        action = update.get("action", "append_paragraph")
        section = update.get("section_path", "General")
        justification = str(update.get("justification") or "").replace("--", "—")
        blocks.append(
            f"<!-- Updated: {run_date} | {action} | {section} -->\n"
            f"<!-- Justification: {justification} -->\n"
            f"{content}\n"
        )
    if not blocks:
        return current_doc

    update_block = f"## Recent Updates — {run_date}\n\n" + "\n".join(blocks) + "\n---\n\n"
    matches = list(_REFERENCES_HEADING_RE.finditer(current_doc))
    if matches:
        start = matches[-1].start()
        return current_doc[:start] + update_block + current_doc[start:]
    return current_doc.rstrip() + "\n\n---\n\n" + update_block


def _append_latex(path: Path, updates: list[dict], run_date: str) -> None:
    """Append LaTeX versions of the updates for pasting into Overleaf."""
    chunks = [
        f"% {u.get('section_path', 'General')} ({u.get('action', '')})\n{u['latex_content']}\n"
        for u in updates if u.get("latex_content")
    ]
    if not chunks:
        return
    existing = _read_current_document(path)
    header = "" if existing else "% Automated literature review updates (LaTeX). Cite keys match references.bib.\n"
    _write_document(path, existing + header + f"\n% ===== Updates {run_date} =====\n" + "\n".join(chunks))


def _sync_google_doc(google_docs: GoogleDocsTool, doc_id: str, updates: list[dict], run_date: str) -> int:
    """Append updates to a Google Doc, skipping any whose marker is already present. Returns failures."""
    if not doc_id or not updates:
        return 0
    current_text = google_docs.get_document_text(doc_id)
    failures = 0
    for update in updates:
        content = update.get("markdown_content") or ""
        if not content:
            continue
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:8]
        marker = f"[Auto-updated {run_date} — {update.get('section_path', '')} — {digest}]"
        if marker in current_text:
            continue  # this exact update was already synced (e.g. by a retried run)
        plain = re.sub(r"^#{1,6}\s*", "", content, flags=re.MULTILINE).replace("**", "")
        if not google_docs.append_text(doc_id, f"\n{marker}\n{plain}"):
            failures += 1
    return failures


def run_document_update(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: Document Update Agent.
    Applies synthesis findings to living literature review and research proposal.
    """
    synthesis_report = state.get("synthesis_report", {}) or {}
    run_date = state.get("run_date", datetime.now().strftime("%Y-%m-%d"))

    if settings.dry_run:
        logger.info("doc_update_dry_run", papers=len(state.get("included_papers", [])))
        return {
            "lit_review_updates": [],
            "proposal_updates": [],
            "new_citations": [],
            "documents_updated": False,
        }

    db = PaperDatabase()
    # The database is the source of truth: it holds this run's included papers and any
    # earlier ones whose update failed, and skips papers another run already documented
    # (e.g. when a paused run is resumed after a later run).
    lit_pending = [p for p in db.get_undocumented_corpus() if p.get("paper_id")]
    proposal_pending = [p for p in db.get_undocumented_corpus("proposal_documented") if p.get("paper_id")]
    papers = list({p["paper_id"]: p for p in lit_pending + proposal_pending}.values())

    if not papers:
        logger.info("doc_update_skip", reason="no_papers_to_document")
        return {
            "lit_review_updates": [],
            "proposal_updates": [],
            "new_citations": [],
            "documents_updated": False,
        }

    logger.info("doc_update_start", papers=len(papers), run_date=run_date)

    # ── BibTeX: deterministic entries from metadata, written before any LLM call ──
    bib_path = _output_path(BIBTEX_FILE)
    bib_text = _read_current_document(bib_path)
    taken = existing_keys(bib_text)
    entries = build_entries(papers, taken)
    new_citations = []
    for paper in papers:
        if paper["paper_id"] in entries:
            key, bibtex = entries[paper["paper_id"]]
            paper["citation_key"], paper["bibtex"] = key, bibtex
            new_citations.append({"key": key, "bibtex": bibtex, "paper_id": paper["paper_id"]})
            db.update_corpus_paper(paper["paper_id"], {"citation_key": key, "bibtex": bibtex})
    if new_citations:
        header = "" if bib_text.strip() else "% Bibliography maintained by PhD LitReview Automator\n"
        _write_document(bib_path, bib_text.rstrip() + ("\n\n" if bib_text.strip() else header + "\n")
                        + "\n\n".join(c["bibtex"] for c in new_citations) + "\n")

    llm = _build_llm()
    google_docs = GoogleDocsTool()
    lit_path, proposal_path = _output_path(LIT_REVIEW_FILE), _output_path(PROPOSAL_FILE)
    current_lit_review = _read_current_document(lit_path)
    current_proposal = _read_current_document(proposal_path)

    by_id = {p["paper_id"]: p for p in papers}
    lit_papers = [by_id[p["paper_id"]] for p in lit_pending]
    proposal_papers = [by_id[p["paper_id"]] for p in proposal_pending]

    # ── Literature review, in batches so every paper's details fit in the prompt ──
    lit_review_updates: list[DocumentUpdate] = []
    documented_ids: list[str] = []
    failed_batches = 0
    for i in range(0, len(lit_papers), PAPERS_PER_UPDATE):
        batch = lit_papers[i:i + PAPERS_PER_UPDATE]
        try:
            updates = _generate_lit_review_update(llm, synthesis_report, batch, current_lit_review, run_date)
            updates = [u for u in updates if u.get("markdown_content")]
            if not updates:
                raise ValueError("reply contained no document updates")
        except Exception as e:
            failed_batches += 1
            logger.error("lit_review_update_failed", batch_start=i, papers=len(batch), error=str(e)[:300])
            continue
        lit_review_updates.extend(updates)
        documented_ids.extend(p["paper_id"] for p in batch)

    proposal_updates: list[DocumentUpdate] = []
    proposal_failed = False
    if proposal_papers:
        try:
            proposal_updates = _generate_proposal_update(llm, synthesis_report, proposal_papers, current_proposal, run_date)
            proposal_updates = [u for u in proposal_updates if u.get("markdown_content")]
        except Exception as e:
            proposal_failed = True
            logger.error("proposal_update_failed", error=str(e)[:300])

    # ── Apply to local files ──
    if lit_review_updates:
        _write_document(lit_path, _apply_updates_to_markdown(current_lit_review, lit_review_updates, run_date))
        _append_latex(_output_path(LATEX_UPDATES_FILE), lit_review_updates, run_date)
    if proposal_updates:
        _write_document(proposal_path, _apply_updates_to_markdown(current_proposal, proposal_updates, run_date))

    db.mark_documented(documented_ids)
    if not proposal_failed:
        # Also when the model chose not to change the proposal for these papers
        db.mark_documented([p["paper_id"] for p in proposal_papers], field="proposal_documented")

    # ── Sync to Google Docs (if configured) ──
    sync_failures = 0
    if google_docs.enabled:
        try:
            sync_failures += _sync_google_doc(google_docs, settings.litreview_google_doc_id, lit_review_updates, run_date)
            sync_failures += _sync_google_doc(google_docs, settings.proposal_google_doc_id, proposal_updates, run_date)
            logger.info("google_docs_synced", lit_updates=len(lit_review_updates), failures=sync_failures)
        except Exception as e:
            sync_failures += 1
            logger.warning("google_docs_sync_failed", error=str(e)[:300])

    logger.info(
        "doc_update_complete",
        lit_review_updates=len(lit_review_updates),
        proposal_updates=len(proposal_updates),
        new_citations=len(new_citations),
        papers_documented=len(documented_ids),
        papers_pending=len(lit_papers) - len(documented_ids),
    )

    result: dict[str, Any] = {
        "lit_review_updates": lit_review_updates,
        "proposal_updates": proposal_updates,
        "new_citations": new_citations,
        "documents_updated": bool(lit_review_updates or proposal_updates),
    }
    warnings = []
    if failed_batches:
        warnings.append(
            f"Literature review update failed for {len(lit_papers) - len(documented_ids)} paper(s); "
            f"they will be retried next run."
        )
    if proposal_failed:
        warnings.append(f"Research proposal update failed for {len(proposal_papers)} paper(s); it will be retried next run.")
    if sync_failures:
        warnings.append(
            f"{sync_failures} Google Docs update(s) failed — the documents must be native Google Docs "
            f"(not uploaded .docx files) shared with the service account as Editor; see the run log."
        )
    if warnings:
        result["warnings"] = warnings
    if failed_batches and not documented_ids:
        result["errors"] = [{
            "node": "document_update",
            "error": "Every literature review update failed; papers stay queued for the next run",
            "timestamp": datetime.now().isoformat(),
        }]
    return result
