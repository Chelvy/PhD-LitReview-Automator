"""
LangGraph Pipeline Orchestration for PhD LitReview Automator.

Graph architecture:
  START
    → generate_run_metadata       (set run_date, run_id, load known IDs)
    → discovery                   (multi-source search + LLM relevance pre-filter)
    → deduplicate                 (remove known papers, apply strict uniqueness)
    → [conditional: any new papers?]
        ↓ no → skip_to_email
        ↓ yes
    → extraction                  (PDF + LLM structured extraction)
    → critical_analysis           (Wagner framework, inclusion/exclusion)
    → [conditional: any included?]
        ↓ no → generate_email
        ↓ yes
    → synthesis                   (cross-paper synthesis + gap update)
    → document_update             (update lit review + proposal)
    → visualization               (update all viz data files)
    → generate_email              (compose daily report)
    → send_email                  (dispatch + archive)
  END

Persistence: SQLite checkpointer (upgrade to PostgreSQL for cloud deployment).
Human-in-the-loop: optional interrupt_before=["document_update"] checkpoint.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Literal

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph

from config.settings import settings
from state.pipeline_state import PipelineState
from tools.paper_db import PaperDatabase

# Import agent node functions
from agents.discovery_agent import run_discovery
from agents.extraction_agent import run_extraction
from agents.critical_analysis_agent import run_critical_analysis
from agents.synthesis_agent import run_synthesis
from agents.document_update_agent import run_document_update
from agents.visualization_agent import run_visualization
from agents.email_reporter_agent import run_email_reporter

logger = structlog.get_logger(__name__)


# ── Node functions (thin wrappers with logging and error handling) ──────────

def generate_run_metadata(state: PipelineState) -> dict[str, Any]:
    """Initialize pipeline run with metadata and load known paper IDs."""
    run_date = datetime.now().strftime("%Y-%m-%d")
    run_id = str(uuid.uuid4())

    # Load known paper IDs from database to avoid reprocessing
    db = PaperDatabase()
    known_ids = list(db.get_known_ids())
    stats = db.get_stats()

    logger.info(
        "pipeline_started",
        run_date=run_date,
        run_id=run_id,
        known_papers=len(known_ids),
        corpus_size=stats.get("total_corpus", 0),
    )

    return {
        "run_date": run_date,
        "run_id": run_id,
        "days_lookback": settings.days_lookback,
        "known_paper_ids": known_ids,
        "errors": [],
        "warnings": [],
        "human_review_needed": False,
        "human_review_reasons": [],
    }


def discovery_node(state: PipelineState) -> dict[str, Any]:
    """Run the Literature Discovery Agent."""
    logger.info("node_start", node="discovery")
    try:
        result = run_discovery(state)
        logger.info(
            "node_complete",
            node="discovery",
            papers_found=len(result.get("discovered_papers", [])),
        )
        return result
    except Exception as e:
        logger.error("node_error", node="discovery", error=str(e))
        return {
            "discovered_papers": [],
            "search_queries": [],
            "errors": [{"node": "discovery", "error": str(e), "timestamp": datetime.now().isoformat()}],
        }


def deduplicate_node(state: PipelineState) -> dict[str, Any]:
    """
    Deduplicate discovered papers against known corpus.
    Also applies final title-based deduplication within the batch.
    """
    discovered = state.get("discovered_papers", [])
    known_ids = set(state.get("known_paper_ids", []))

    if not discovered:
        return {"new_papers": [], "duplicate_count": 0}

    seen_in_batch: set[str] = set()
    new_papers = []
    duplicate_count = 0

    for paper in discovered:
        paper_id = paper.get("paper_id", "")

        # Skip if already in corpus
        if paper_id in known_ids:
            duplicate_count += 1
            continue

        # Skip duplicates within this batch (same paper from different sources)
        if paper_id in seen_in_batch:
            duplicate_count += 1
            continue

        # DOI-based dedup (different IDs, same DOI)
        doi = paper.get("doi", "")
        if doi:
            doi_key = f"doi:{doi}"
            if doi_key in seen_in_batch or doi_key in known_ids:
                duplicate_count += 1
                continue
            seen_in_batch.add(doi_key)

        seen_in_batch.add(paper_id)
        new_papers.append(paper)

    logger.info(
        "deduplication_complete",
        discovered=len(discovered),
        new=len(new_papers),
        duplicates=duplicate_count,
    )

    return {
        "new_papers": new_papers,
        "duplicate_count": duplicate_count,
    }


def extraction_node(state: PipelineState) -> dict[str, Any]:
    """Run the PDF Extraction & Summarization Agent."""
    logger.info("node_start", node="extraction", papers=len(state.get("new_papers", [])))
    try:
        result = run_extraction(state)
        logger.info(
            "node_complete",
            node="extraction",
            extracted=len(result.get("extracted_papers", [])),
        )
        return result
    except Exception as e:
        logger.error("node_error", node="extraction", error=str(e))
        return {
            "extracted_papers": state.get("new_papers", []),  # Pass through unextracted
            "extraction_errors": [{"error": str(e)}],
            "errors": [{"node": "extraction", "error": str(e)}],
        }


def critical_analysis_node(state: PipelineState) -> dict[str, Any]:
    """Run the Critical Analysis Agent."""
    logger.info("node_start", node="critical_analysis")
    try:
        result = run_critical_analysis(state)
        logger.info(
            "node_complete",
            node="critical_analysis",
            included=len(result.get("included_papers", [])),
            excluded=len(result.get("excluded_papers", [])),
        )
        return result
    except Exception as e:
        logger.error("node_error", node="critical_analysis", error=str(e))
        return {
            "analyzed_papers": state.get("extracted_papers", []),
            "included_papers": [],
            "excluded_papers": state.get("extracted_papers", []),
            "errors": [{"node": "critical_analysis", "error": str(e)}],
        }


def synthesis_node(state: PipelineState) -> dict[str, Any]:
    """Run the Synthesis & Gap Identification Agent."""
    logger.info("node_start", node="synthesis", papers=len(state.get("included_papers", [])))
    try:
        result = run_synthesis(state)
        logger.info("node_complete", node="synthesis")
        return result
    except Exception as e:
        logger.error("node_error", node="synthesis", error=str(e))
        return {
            "synthesis_report": {
                "synthesis_date": state.get("run_date", ""),
                "papers_synthesized": [],
                "gap_status_update": {},
                "historical_narrative_update": {"new_developments": f"Synthesis error: {e}"},
                "key_debates_update": [],
                "contradictions": [],
                "lit_review_sections_to_update": [],
                "proposal_sections_to_update": [],
            },
            "errors": [{"node": "synthesis", "error": str(e)}],
        }


def document_update_node(state: PipelineState) -> dict[str, Any]:
    """Run the Document Update Agent."""
    logger.info("node_start", node="document_update")
    try:
        result = run_document_update(state)
        logger.info("node_complete", node="document_update")
        return result
    except Exception as e:
        logger.error("node_error", node="document_update", error=str(e))
        return {
            "lit_review_updates": [],
            "proposal_updates": [],
            "new_citations": [],
            "documents_updated": False,
            "errors": [{"node": "document_update", "error": str(e)}],
        }


def visualization_node(state: PipelineState) -> dict[str, Any]:
    """Run the Visualization & Mapping Agent."""
    logger.info("node_start", node="visualization")
    try:
        result = run_visualization(state)
        logger.info("node_complete", node="visualization")
        return result
    except Exception as e:
        logger.error("node_error", node="visualization", error=str(e))
        return {
            "visualizations": [],
            "errors": [{"node": "visualization", "error": str(e)}],
        }


def email_reporter_node(state: PipelineState) -> dict[str, Any]:
    """Run the Email Reporter & Archivist Agent."""
    logger.info("node_start", node="email_reporter")
    try:
        result = run_email_reporter(state)
        logger.info("node_complete", node="email_reporter", email_sent=result.get("email_sent"))
        return result
    except Exception as e:
        logger.error("node_error", node="email_reporter", error=str(e))
        return {
            "email_report": {"email_subject": "Pipeline Error", "human_action_required": True},
            "email_sent": False,
            "errors": [{"node": "email_reporter", "error": str(e)}],
        }


# ── Conditional routing ─────────────────────────────────────────────────────

def route_after_deduplication(
    state: PipelineState,
) -> Literal["extraction", "email_reporter"]:
    """After dedup: if new papers exist, extract them; else skip to email."""
    new_papers = state.get("new_papers", [])
    if new_papers:
        logger.info("routing", decision="extraction", new_papers=len(new_papers))
        return "extraction"
    logger.info("routing", decision="email_reporter", reason="no_new_papers")
    return "email_reporter"


def route_after_analysis(
    state: PipelineState,
) -> Literal["synthesis", "email_reporter"]:
    """After critical analysis: if papers were included, synthesize; else email."""
    included = state.get("included_papers", [])
    if included:
        logger.info("routing", decision="synthesis", included_papers=len(included))
        return "synthesis"
    logger.info("routing", decision="email_reporter", reason="no_included_papers")
    return "email_reporter"


def route_after_synthesis(
    state: PipelineState,
) -> Literal["document_update", "email_reporter"]:
    """After synthesis: update docs unless dry_run or error."""
    if settings.dry_run:
        logger.info("routing", decision="email_reporter", reason="dry_run")
        return "email_reporter"
    return "document_update"


# ── Human-in-the-loop interrupt check ──────────────────────────────────────

def should_interrupt_for_human(state: PipelineState) -> bool:
    """Determine if we should pause for human review."""
    return settings.human_review_checkpoint and bool(state.get("included_papers"))


# ── Graph construction ───────────────────────────────────────────────────────

def build_pipeline(
    checkpointer_path: str | None = None,
    interrupt_before_docs: bool = False,
) -> CompiledGraph:
    """
    Build and compile the LangGraph pipeline.

    Args:
        checkpointer_path: SQLite DB path for persistence. Defaults to settings value.
        interrupt_before_docs: If True, pause before document updates for human review.

    Returns:
        Compiled LangGraph application ready to invoke.
    """
    graph = StateGraph(PipelineState)

    # ── Add all nodes ───────────────────────────────────────────────────
    graph.add_node("generate_run_metadata", generate_run_metadata)
    graph.add_node("discovery", discovery_node)
    graph.add_node("deduplicate", deduplicate_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("critical_analysis", critical_analysis_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("document_update", document_update_node)
    graph.add_node("visualization", visualization_node)
    graph.add_node("email_reporter", email_reporter_node)

    # ── Define edges ────────────────────────────────────────────────────
    graph.add_edge(START, "generate_run_metadata")
    graph.add_edge("generate_run_metadata", "discovery")
    graph.add_edge("discovery", "deduplicate")

    # Route after deduplication
    graph.add_conditional_edges(
        "deduplicate",
        route_after_deduplication,
        {
            "extraction": "extraction",
            "email_reporter": "email_reporter",
        },
    )

    graph.add_edge("extraction", "critical_analysis")

    # Route after critical analysis
    graph.add_conditional_edges(
        "critical_analysis",
        route_after_analysis,
        {
            "synthesis": "synthesis",
            "email_reporter": "email_reporter",
        },
    )

    # Route after synthesis (handles dry_run)
    graph.add_conditional_edges(
        "synthesis",
        route_after_synthesis,
        {
            "document_update": "document_update",
            "email_reporter": "email_reporter",
        },
    )

    graph.add_edge("document_update", "visualization")
    graph.add_edge("visualization", "email_reporter")
    graph.add_edge("email_reporter", END)

    # ── Compile with persistence ─────────────────────────────────────────
    db_path = checkpointer_path or settings.sqlite_db_path

    import os
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    checkpointer = SqliteSaver.from_conn_string(db_path)

    # Interrupt before document updates for human review (optional)
    interrupt_nodes = ["document_update"] if (
        interrupt_before_docs or settings.human_review_checkpoint
    ) else []

    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes if interrupt_nodes else None,
    )

    logger.info(
        "pipeline_compiled",
        checkpoint_db=db_path,
        human_checkpoint=bool(interrupt_nodes),
    )
    return app


def run_pipeline(
    config: dict | None = None,
    interrupt_before_docs: bool = False,
) -> dict[str, Any]:
    """
    Run a complete pipeline execution.

    Args:
        config: Optional LangGraph run config (thread_id, etc.)
        interrupt_before_docs: Pause for human review before document updates.

    Returns:
        Final pipeline state.
    """
    app = build_pipeline(interrupt_before_docs=interrupt_before_docs)

    run_config: RunnableConfig = {
        "configurable": {
            "thread_id": f"daily-run-{datetime.now().strftime('%Y-%m-%d')}",
            **(config or {}),
        },
        "recursion_limit": 50,
    }

    logger.info("pipeline_run_starting", config=run_config)

    initial_state: PipelineState = {}  # All defaults applied in generate_run_metadata

    try:
        final_state = app.invoke(initial_state, config=run_config)
        logger.info(
            "pipeline_run_complete",
            included=len(final_state.get("included_papers", [])),
            email_sent=final_state.get("email_sent", False),
            errors=len(final_state.get("errors", [])),
        )
        return final_state
    except Exception as e:
        logger.error("pipeline_run_failed", error=str(e))
        raise
