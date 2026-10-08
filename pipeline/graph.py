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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
try:
    from langgraph.graph.graph import CompiledGraph
except ImportError:
    from langgraph.graph.state import CompiledStateGraph as CompiledGraph  # type: ignore

from config.settings import settings
from state.pipeline_state import PipelineState
from tools.paper_db import PaperDatabase, dedup_keys
from agents.llm_utils import reset_failure_counts

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
    known_keys = sorted(db.get_known_keys())
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
        "known_paper_keys": known_keys,
        "errors": [],
        "warnings": [],
        "synthesis_failed": False,
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
    Remove papers already analyzed in earlier runs, and the same paper surfaced by
    several sources in this run. Papers are matched on ID, DOI (case-insensitive),
    arXiv ID and normalized title, since each source assigns its own paper_id.
    """
    discovered = state.get("discovered_papers", [])
    known_keys = set(state.get("known_paper_keys", [])) | {
        f"id:{pid}" for pid in state.get("known_paper_ids", [])
    }

    if not discovered:
        return {"new_papers": [], "duplicate_count": 0}

    seen_in_batch: set[str] = set()
    new_papers = []
    duplicate_count = 0

    for paper in discovered:
        keys = dedup_keys(paper)
        if not keys or keys & known_keys or keys & seen_in_batch:
            duplicate_count += 1
            continue
        seen_in_batch |= keys
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
            "synthesis_failed": True,  # skip document updates; papers are retried next run
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

def _has_pending_documentation() -> bool:
    """Corpus papers from earlier runs whose document update failed and must be retried."""
    if settings.dry_run:
        return False
    try:
        db = PaperDatabase()
        return bool(db.get_undocumented_corpus() or db.get_undocumented_corpus("proposal_documented"))
    except Exception as e:
        logger.warning("pending_documentation_check_failed", error=str(e))
        return False


def route_after_deduplication(
    state: PipelineState,
) -> Literal["extraction", "synthesis", "email_reporter"]:
    """After dedup: extract new papers; else retry pending document updates; else email."""
    new_papers = state.get("new_papers", [])
    if new_papers:
        logger.info("routing", decision="extraction", new_papers=len(new_papers))
        return "extraction"
    if _has_pending_documentation():
        logger.info("routing", decision="synthesis", reason="pending_document_updates")
        return "synthesis"
    logger.info("routing", decision="email_reporter", reason="no_new_papers")
    return "email_reporter"


def route_after_analysis(
    state: PipelineState,
) -> Literal["synthesis", "email_reporter"]:
    """After critical analysis: synthesize if papers were included or documents are pending."""
    included = state.get("included_papers", [])
    if included:
        logger.info("routing", decision="synthesis", included_papers=len(included))
        return "synthesis"
    if _has_pending_documentation():
        logger.info("routing", decision="synthesis", reason="pending_document_updates")
        return "synthesis"
    logger.info("routing", decision="email_reporter", reason="no_included_papers")
    return "email_reporter"


def route_after_synthesis(
    state: PipelineState,
) -> Literal["document_update", "email_reporter"]:
    """After synthesis: update docs unless dry_run or the synthesis failed."""
    if settings.dry_run:
        logger.info("routing", decision="email_reporter", reason="dry_run")
        return "email_reporter"
    if state.get("synthesis_failed"):
        # Papers stay undocumented in the corpus and are retried next run
        logger.info("routing", decision="email_reporter", reason="synthesis_failed")
        return "email_reporter"
    return "document_update"


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
            "synthesis": "synthesis",
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

    import os, sqlite3
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    # SqliteSaver requires a direct sqlite3 connection (not from_conn_string context manager).
    # Dry runs keep checkpoints in memory so they write nothing under storage/.
    try:
        if settings.dry_run:
            checkpointer = InMemorySaver()
        else:
            _conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(_conn)
    except Exception:
        # Fallback to in-memory saver for environments where SQLite write fails
        logger.warning("sqlite_checkpointer_fallback", reason="Using InMemorySaver")
        checkpointer = InMemorySaver()

    # Interrupt before document updates for human review (optional). Never in
    # GitHub Actions: nobody is there to resume the run, so it would just stop.
    interrupt_nodes = ["document_update"] if (
        interrupt_before_docs or settings.human_review_checkpoint
    ) else []
    if interrupt_nodes and os.environ.get("GITHUB_ACTIONS") == "true":
        logger.warning("human_checkpoint_ignored", reason="unattended GitHub Actions run")
        interrupt_nodes = []

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
    reset_failure_counts()  # a long-running `schedule` process must not carry them across runs

    # Each run gets its own checkpoint thread. Reusing a thread (e.g. one per date)
    # makes LangGraph resume the previous run's state, and the operator.add reducers
    # then accumulate its errors and paper lists into the new run.
    thread_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            **(config or {}),
        },
        "recursion_limit": 50,
    }

    logger.info("pipeline_run_starting", config=run_config)

    initial_state: PipelineState = {}  # All defaults applied in generate_run_metadata

    try:
        final_state = app.invoke(initial_state, config=run_config)
    except Exception as e:
        logger.error("pipeline_run_failed", error=str(e))
        raise
    return _finish(app, run_config, final_state)


def resume_pipeline(thread_id: str) -> dict[str, Any]:
    """Resume a run paused at the human-review checkpoint (after reviewing it)."""
    app = build_pipeline(interrupt_before_docs=False)
    run_config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    snapshot = app.get_state(run_config)
    if not snapshot.next:
        raise ValueError(f"Run {thread_id!r} is not paused (unknown thread or already finished)")
    logger.info("pipeline_resuming", thread_id=thread_id, next=list(snapshot.next))
    final_state = app.invoke(None, config=run_config)
    return _finish(app, run_config, final_state)


def _finish(app: Any, run_config: RunnableConfig, final_state: dict[str, Any]) -> dict[str, Any]:
    """Annotate the final state with the thread ID and, if paused, where the run stopped."""
    thread_id = run_config["configurable"]["thread_id"]
    final_state = {**final_state, "thread_id": thread_id}
    paused_at = list(app.get_state(run_config).next)
    if paused_at:
        final_state["paused_at"] = paused_at
        logger.info("pipeline_paused_for_review", thread_id=thread_id, next=paused_at)
    else:
        logger.info(
            "pipeline_run_complete",
            included=len(final_state.get("included_papers", [])),
            email_sent=final_state.get("email_sent", False),
            errors=len(final_state.get("errors", [])),
        )
    return final_state
