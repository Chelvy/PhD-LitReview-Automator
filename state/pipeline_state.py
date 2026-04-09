"""
LangGraph state definitions for PhD LitReview Automator.
Uses TypedDict for LangGraph compatibility with full type annotations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from typing_extensions import TypedDict, Annotated
import operator


# ── Paper data model ──────────────────────────────────────────────────────────

class Paper(TypedDict, total=False):
    # Core metadata
    paper_id: str
    title: str
    authors: list[str]
    year: int
    abstract: str
    doi: str
    arxiv_id: str
    semantic_scholar_id: str
    openalex_id: str
    venue: str
    citation_count: int
    url: str
    pdf_url: str
    fields_of_study: list[str]
    tldr: str
    discovery_date: str            # ISO date when we first found it
    discovery_source: str          # "semantic_scholar", "arxiv", "crossref", etc.
    search_query: str              # Query that surfaced this paper

    # Extraction fields (from ExtractionAgent)
    extraction_complete: bool
    research_objectives: str
    methodology: str
    datasets_used: list[str]
    results: str
    conclusions: str
    key_claims: list[str]
    theoretical_contribution: str
    empirical_contribution: str
    formalisms_introduced: list[str]
    assumptions: list[str]
    strengths: list[str]
    blind_spots: list[str]
    blank_spots: list[str]
    key_contributions: list[str]
    related_anchor_papers: list[str]
    quotes: list[str]

    # Critical analysis fields (from CriticalAnalysisAgent)
    rq1_relevance: float           # 0.0–10.0
    rq2_relevance: float
    rq3_relevance: float
    relevance_score: float         # Weighted composite 0.0–10.0
    inclusion_decision: bool
    exclusion_reason: Optional[str]
    gaps_addressed: list[str]      # ["GAP-1", "GAP-3", ...]
    wagner_blank_spots: list[str]
    wagner_blind_spots: list[str]
    actual_contribution: str
    assumes_component_isolation: bool
    integration_paradox_evidence: Optional[str]
    methodological_rigor_score: float
    methodological_concerns: list[str]
    synthesis_tags: list[str]      # ["compositional_failure", "empirical", ...]
    citation_priority: str         # "high" | "medium" | "low"
    critical_notes: str
    preliminary_relevance: float   # Pre-extraction estimate from discovery

    # Synthesis fields (from SynthesisAgent)
    synthesis_notes: str
    related_papers: list[str]
    contradicts_papers: list[str]
    extends_papers: list[str]

    # BibTeX / citation
    bibtex: str
    apa_citation: str


class DocumentUpdate(TypedDict, total=False):
    section_path: str
    action: str   # "create_subsection" | "append_paragraph" | "update_paragraph" | "add_citation"
    latex_content: str
    markdown_content: str
    citations_added: list[str]
    justification: str


class BibtexEntry(TypedDict, total=False):
    key: str
    bibtex: str
    apa: str


class SynthesisReport(TypedDict, total=False):
    synthesis_date: str
    papers_synthesized: list[str]
    historical_narrative_update: dict[str, Any]
    key_debates_update: list[dict[str, Any]]
    gap_status_update: dict[str, dict[str, Any]]
    contradictions: list[dict[str, Any]]
    anchor_papers: list[str]
    methodology_recommendations: list[str]
    contribution_update: str
    lit_review_sections_to_update: list[dict[str, Any]]
    proposal_sections_to_update: list[dict[str, Any]]


class VisualizationData(TypedDict, total=False):
    visualization_type: str
    data: dict[str, Any]
    recommended_tool: str
    mermaid_diagram: str
    notes: str


class EmailReport(TypedDict, total=False):
    email_subject: str
    email_html: str
    email_text: str
    attachments: list[str]
    human_action_required: bool
    human_action_items: list[str]
    archive_filename: str


# ── Main pipeline state ───────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph nodes.
    Uses Annotated[list, operator.add] for lists that nodes append to
    (avoids overwrite conflicts in parallel nodes).
    """

    # ── Run metadata ──────────────────────────────────────────────────────
    run_date: str                            # ISO date "YYYY-MM-DD"
    run_id: str                              # UUID for this pipeline run
    days_lookback: int

    # ── Discovery phase ───────────────────────────────────────────────────
    search_queries: list[str]
    discovered_papers: Annotated[list[Paper], operator.add]
    known_paper_ids: list[str]               # IDs already in our corpus

    # ── Deduplication ─────────────────────────────────────────────────────
    new_papers: list[Paper]                  # After removing known IDs
    duplicate_count: int

    # ── Extraction phase ──────────────────────────────────────────────────
    extracted_papers: Annotated[list[Paper], operator.add]
    extraction_errors: Annotated[list[dict[str, Any]], operator.add]

    # ── Analysis phase ────────────────────────────────────────────────────
    analyzed_papers: Annotated[list[Paper], operator.add]
    included_papers: list[Paper]
    excluded_papers: list[Paper]

    # ── Synthesis ─────────────────────────────────────────────────────────
    synthesis_report: SynthesisReport

    # ── Document updates ──────────────────────────────────────────────────
    lit_review_updates: list[DocumentUpdate]
    proposal_updates: list[DocumentUpdate]
    new_citations: list[BibtexEntry]
    documents_updated: bool

    # ── Visualization ─────────────────────────────────────────────────────
    visualizations: list[VisualizationData]

    # ── Email ─────────────────────────────────────────────────────────────
    email_report: EmailReport
    email_sent: bool

    # ── Pipeline control ──────────────────────────────────────────────────
    errors: Annotated[list[dict[str, Any]], operator.add]
    warnings: Annotated[list[str], operator.add]
    human_review_needed: bool
    human_review_reasons: list[str]
    skip_reason: Optional[str]              # If run should be aborted early
