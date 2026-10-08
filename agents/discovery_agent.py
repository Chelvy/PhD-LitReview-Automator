"""
Agent 1: Literature Discovery & Monitoring Agent

Runs daily searches across Semantic Scholar, ArXiv, CrossRef, OpenAlex,
and Elicit. Generates targeted queries, merges results, deduplicates,
and applies preliminary relevance filtering.

Uses Claude claude-sonnet-4-6 for:
  1. Generating intelligent, targeted search queries
  2. Preliminary relevance scoring from titles + abstracts
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from config.prompts import AgentPrompts
from state.pipeline_state import PipelineState
from tools.semantic_scholar import SemanticScholarTool
from tools.arxiv_tool import ArxivTool
from tools.crossref_tool import CrossRefTool
from tools.openalex_tool import OpenAlexTool
from tools.elicit_tool import ElicitTool
from tools.paper_db import PaperDatabase, dedup_keys
from agents.llm_utils import build_llm, invoke_json, list_field, text_field

logger = structlog.get_logger(__name__)

# Integration Paradox focused search queries (baseline — LLM augments these)
BASELINE_QUERIES = [
    "system-level AI reliability integration failure composition",
    "trustworthy AI pipeline compositional failure multi-component",
    "AI ML system assurance end-to-end reliability",
    "interface misalignment machine learning data pipeline",
    "multi-agent AI system failure modes reliability",
    "human-AI handoff coupled failure socio-technical",
    "runtime monitoring AI system trust reliability",
    "distribution shift feedback loop AI deployed system",
    "EU AI Act system-level risk high-risk AI compliance",
    "assurance case compositional verification AI software",
    "integration testing AI systems error propagation",
    "trust composition AI components system architecture",
]

# Key anchor paper IDs for citation tracking
ANCHOR_PAPER_SS_IDS = [
    # Will be populated as corpus grows — examples:
    # "204e3073870fae3d05bcbc2f6a8e263d55571600",  # Attention Is All You Need
]

# Tracked authors in AI systems reliability space
TRACKED_AUTHORS = [
    # Format: (Semantic Scholar author ID, name)
    # Populate as you identify key researchers in this space
]


def _build_llm():
    return build_llm(temperature=0.2, max_tokens=8000, effort="low")


def _generate_search_queries(llm: ChatAnthropic, run_date: str) -> list[str]:
    """
    Use Claude to generate fresh, targeted search queries for today's run.
    Combines baseline queries with LLM-generated variations.
    """
    system = SystemMessage(content=AgentPrompts.DISCOVERY.replace("{days_lookback}", str(settings.days_lookback)))

    prompt = f"""Today is {run_date}. Generate 12 highly targeted search queries for
academic databases (Semantic Scholar, ArXiv, OpenAlex) to discover papers published
in the last {settings.days_lookback} days that are relevant to our research.

Focus especially on papers about:
- Compositional failure in AI/ML systems (not just single models)
- System-level trust and reliability in multi-model pipelines
- Interface contracts and mismatch detection in AI systems
- SDLC practices for AI system-level assurance
- EU AI Act implementation and system-level requirements
- Runtime monitoring and assurance evidence for AI
- Empirical studies of AI integration failures
- Agentic AI system reliability and multi-agent failures
- Human-AI workflow coupling and failure propagation

Return ONLY a JSON array of query strings, no explanation:
["query 1", "query 2", ...]"""

    try:
        queries = invoke_json(llm, [system, HumanMessage(content=prompt)])
        # Claude sometimes wraps the array in an object like {"search_queries": [...]}
        if isinstance(queries, dict):
            queries = next((v for v in queries.values() if isinstance(v, list)), [])
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()] if isinstance(queries, list) else []
        if queries:
            logger.info("queries_generated_by_llm", count=len(queries))
            return queries
        logger.warning("query_generation_empty", fallback="baseline queries")
    except Exception as e:
        logger.warning("query_generation_failed", error=str(e), fallback="baseline queries")

    return BASELINE_QUERIES


def _preliminary_relevance_score(
    llm: ChatAnthropic,
    papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Use Claude to score preliminary relevance of discovered papers
    based on title + abstract only. Batch process for efficiency.
    """
    if not papers:
        return []

    # Process in batches of 10
    scored = []
    batch_size = 10

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]

        try:
            # Build batch prompt
            papers_text = "\n\n".join(
                f"PAPER {j+1}:\nTitle: {text_field(p, 'title')}\n"
                f"Authors: {', '.join(str(a) for a in list_field(p, 'authors')[:3] if a)}\n"
                f"Year: {text_field(p, 'year')}\n"
                f"Venue: {text_field(p, 'venue')}\n"
                f"Abstract: {text_field(p, 'abstract', limit=600)}"
                for j, p in enumerate(batch)
            )

            prompt = f"""Score the relevance of each paper to our research on:
"The Integration Paradox: Why Reliable AI/ML Components Compose into Unreliable Systems"

Our key research questions:
- RQ1: How should system-level trust be represented and composed across the AI SDLC?
- RQ2: Which integration-failure mechanisms recur and what indicators predict them?
- RQ3: Which SDLC practices reduce integration risk and provide auditable assurance?

SCORING RULES:
- 9-10: Directly addresses integration paradox or a core gap (must include)
- 7-8: Strong relevance to one+ RQ, high quality
- 5-6: Moderate relevance (conditional include)
- 3-4: Peripheral relevance (likely exclude)
- 0-2: Not relevant (exclude)

EXCLUDE (score ≤3):
- Papers purely about improving a single model's accuracy/robustness without system context
- Purely algorithmic papers without system-integration implications

PAPERS TO SCORE:
{papers_text}

Return ONLY a JSON array of objects (one per paper, in order):
[{{"paper_number": 1, "score": 7.5, "reason": "Brief justification", "include": true}}, ...]"""

            scores = invoke_json(llm, [HumanMessage(content=prompt)])
            # Handle dict-wrapped response e.g. {"scores": [...]}
            if isinstance(scores, dict):
                scores = next((v for v in scores.values() if isinstance(v, list)), [])
            for item in scores:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("paper_number", 0)) - 1
                    score = float(item.get("score", 5.0))
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(batch):
                    batch[idx]["preliminary_relevance"] = score
                    batch[idx]["preliminary_relevance_reason"] = item.get("reason") or ""
                    batch[idx]["preliminary_include"] = bool(item.get("include", True))

        except Exception as e:
            logger.warning("batch_relevance_scoring_failed", batch_start=i, error=str(e))
            for p in batch:
                p.setdefault("preliminary_relevance", 5.0)
                p.setdefault("preliminary_include", True)

        scored.extend(batch)

    return scored


def _record_screened_out(papers: list[dict[str, Any]], run_date: str) -> None:
    """
    Remember papers the pre-screen rejected, so the overlapping search windows of
    later runs skip them instead of scoring them again.
    """
    if not papers:
        return
    db = PaperDatabase()
    for p in papers:
        record = {k: p.get(k) for k in (
            "paper_id", "title", "authors", "year", "doi", "arxiv_id", "venue", "url",
            "publication_date", "discovery_source", "preliminary_relevance",
        )}
        record.update({"screened_out": True, "inclusion_decision": False, "analysis_date": run_date})
        db.save_paper(record)
        db.add_to_excluded(record, f"Pre-screen: {p.get('preliminary_relevance_reason') or 'not relevant'}")


def run_discovery(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: Literature Discovery & Monitoring.
    Searches multiple academic APIs, merges, deduplicates, and pre-scores papers.
    """
    run_date = state.get("run_date", datetime.now().strftime("%Y-%m-%d"))
    days_back = state.get("days_lookback", settings.days_lookback)
    known_ids = set(state.get("known_paper_ids", []))

    logger.info("discovery_start", run_date=run_date, days_back=days_back, known_papers=len(known_ids))

    llm = _build_llm()

    # Step 1: Generate search queries
    queries = _generate_search_queries(llm, run_date)
    logger.info("using_queries", count=len(queries))

    # Step 2: Search all sources in parallel (sequential here for simplicity)
    ss_tool = SemanticScholarTool()
    arxiv_tool = ArxivTool()
    crossref_tool = CrossRefTool()
    openalex_tool = OpenAlexTool()
    elicit_tool = ElicitTool()

    # Skip papers analyzed in earlier runs and the same paper found by several sources
    # (each source assigns its own paper_id, so match on DOI / arXiv ID / title too).
    seen_keys = set(state.get("known_paper_keys", [])) | {f"id:{pid}" for pid in known_ids}
    discovered: list[dict[str, Any]] = []

    def add_papers(papers: list[dict[str, Any]]) -> None:
        for p in papers:
            keys = dedup_keys(p)
            if not keys or keys & seen_keys:
                continue
            seen_keys.update(keys)
            discovered.append(p)

    sources = [
        # (name, queries, search function) — query counts respect each API's rate limits
        ("semantic_scholar", queries[:6], lambda q: ss_tool.search_recent(query=q, days_back=days_back, limit=15)),
        ("arxiv", queries[:8], lambda q: arxiv_tool.search(query=q, days_back=days_back, max_results=10)),
        ("openalex", queries[:4], lambda q: openalex_tool.search(query=q, days_back=days_back, limit=10)),
        ("crossref", queries[:3], lambda q: crossref_tool.search(query=q, days_back=days_back, limit=10)),
    ]
    if elicit_tool.enabled:
        sources.append(("elicit", queries[:3], lambda q: elicit_tool.search(query=q, num_papers=10)))

    warnings: list[str] = []
    total_found = total_failed = 0
    for name, source_queries, search in sources:
        found = failed = 0
        for query in source_queries:
            try:
                results = search(query)
                found += len(results)
                add_papers(results)
            except Exception as e:
                failed += 1
                logger.warning("source_search_failed", source=name, query=query[:50], error=str(e)[:200])
        total_found += found
        total_failed += failed
        logger.info("source_searched", source=name, queries=len(source_queries), found=found, failed=failed)
        if source_queries and failed * 2 >= len(source_queries):
            warnings.append(f"Discovery source '{name}' failed for {failed} of {len(source_queries)} queries this run.")

    # Citation tracking on anchor papers
    for ss_id in ANCHOR_PAPER_SS_IDS:
        try:
            add_papers(ss_tool.get_citations(ss_id, limit=20))
        except Exception as e:
            logger.warning("citation_tracking_failed", paper_id=ss_id, error=str(e))

    logger.info("discovery_raw", total_discovered=len(discovered))

    # Step 3: Preliminary relevance scoring by LLM
    if discovered:
        scored = _preliminary_relevance_score(llm, discovered)
        # Filter out obviously irrelevant papers (and ones the scorer explicitly excluded)
        filtered = [
            p for p in scored
            if p.get("preliminary_relevance", 5.0) >= 4.0 and p.get("preliminary_include", True) is not False
        ]
        logger.info(
            "discovery_filtered",
            total=len(scored),
            passed_filter=len(filtered),
            filtered_out=len(scored) - len(filtered),
        )
        _record_screened_out([p for p in scored if p not in filtered], run_date)
    else:
        filtered = []

    # Cap at max papers per run
    if len(filtered) > settings.max_papers_per_run:
        # Sort by preliminary relevance, take top N
        filtered.sort(key=lambda p: p.get("preliminary_relevance", 5.0), reverse=True)
        dropped = len(filtered) - settings.max_papers_per_run
        filtered = filtered[:settings.max_papers_per_run]
        logger.info("capped_at_max", max_papers=settings.max_papers_per_run, dropped=dropped)
        warnings.append(
            f"{dropped} lower-relevance papers were skipped (MAX_PAPERS_PER_RUN={settings.max_papers_per_run})."
        )

    logger.info(
        "discovery_complete",
        total_discovered=len(discovered),
        after_filter=len(filtered),
        queries_used=len(queries),
    )

    result: dict[str, Any] = {
        "search_queries": queries,
        "discovered_papers": filtered,
    }
    if warnings:
        result["warnings"] = warnings
    if total_found == 0 and total_failed > 0:
        result["errors"] = [{
            "node": "discovery",
            "error": f"No papers found and {total_failed} source searches failed — check network/API access",
            "timestamp": datetime.now().isoformat(),
        }]
    return result
