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
from tools.paper_db import PaperDatabase
from agents.llm_utils import build_llm

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


def _build_llm() -> ChatAnthropic:
    return build_llm(temperature=0.2, max_tokens=4096)


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
        response = llm.invoke([system, HumanMessage(content=prompt)])
        content = response.content.strip()
        # Extract JSON array
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        queries = json.loads(content)
        # Claude sometimes wraps the array in an object like {"search_queries": [...]}
        if isinstance(queries, dict):
            queries = queries.get("search_queries", queries.get("queries", next(iter(queries.values()), [])))
        if isinstance(queries, list) and queries:
            logger.info("queries_generated_by_llm", count=len(queries))
            return queries
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

        # Build batch prompt
        papers_text = "\n\n".join(
            f"PAPER {j+1}:\nTitle: {p.get('title', 'N/A')}\n"
            f"Authors: {', '.join(p.get('authors', [])[:3])}\n"
            f"Year: {p.get('year', 'N/A')}\n"
            f"Venue: {p.get('venue', 'N/A')}\n"
            f"Abstract: {p.get('abstract', 'N/A')[:600]}"
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

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            scores = json.loads(content)
            # Handle dict-wrapped response e.g. {"scores": [...]}
            if isinstance(scores, dict):
                scores = next((v for v in scores.values() if isinstance(v, list)), [])
            for item in scores:
                idx = item.get("paper_number", 1) - 1
                if 0 <= idx < len(batch):
                    batch[idx]["preliminary_relevance"] = item.get("score", 5.0)
                    batch[idx]["preliminary_relevance_reason"] = item.get("reason", "")
                    batch[idx]["preliminary_include"] = item.get("include", True)

        except Exception as e:
            logger.warning("batch_relevance_scoring_failed", batch_start=i, error=str(e))
            for p in batch:
                p.setdefault("preliminary_relevance", 5.0)
                p.setdefault("preliminary_include", True)

        scored.extend(batch)

    return scored


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

    all_papers: dict[str, dict[str, Any]] = {}

    # Semantic Scholar
    for query in queries[:6]:  # Top 6 queries for S2 (rate limit aware)
        try:
            results = ss_tool.search_recent(query=query, days_back=days_back, limit=15)
            for p in results:
                pid = p["paper_id"]
                if pid not in all_papers and pid not in known_ids:
                    all_papers[pid] = p
        except Exception as e:
            logger.warning("ss_search_failed", query=query[:50], error=str(e))

    # ArXiv — especially good for preprints and cutting-edge work
    for query in queries[:8]:
        try:
            results = arxiv_tool.search(query=query, days_back=days_back, max_results=10)
            for p in results:
                pid = p["paper_id"]
                if pid not in all_papers and pid not in known_ids:
                    all_papers[pid] = p
        except Exception as e:
            logger.warning("arxiv_search_failed", query=query[:50], error=str(e))

    # OpenAlex
    for query in queries[:4]:
        try:
            results = openalex_tool.search(query=query, days_back=days_back, limit=10)
            for p in results:
                pid = p["paper_id"]
                if pid not in all_papers and pid not in known_ids:
                    all_papers[pid] = p
        except Exception as e:
            logger.warning("openalex_search_failed", query=query[:50], error=str(e))

    # CrossRef — good for catching published journal papers
    for query in queries[:3]:
        try:
            results = crossref_tool.search(query=query, days_back=days_back, limit=10)
            for p in results:
                pid = p["paper_id"]
                if pid not in all_papers and pid not in known_ids:
                    all_papers[pid] = p
        except Exception as e:
            logger.warning("crossref_search_failed", query=query[:50], error=str(e))

    # Elicit (if configured)
    for query in queries[:3]:
        try:
            results = elicit_tool.search(query=query, num_papers=10)
            for p in results:
                pid = p["paper_id"]
                if pid not in all_papers and pid not in known_ids:
                    all_papers[pid] = p
        except Exception as e:
            logger.warning("elicit_search_failed", query=query[:50], error=str(e))

    # Citation tracking on anchor papers
    for ss_id in ANCHOR_PAPER_SS_IDS:
        try:
            citations = ss_tool.get_citations(ss_id, limit=20)
            for p in citations:
                pid = p["paper_id"]
                if pid not in all_papers and pid not in known_ids:
                    all_papers[pid] = p
        except Exception as e:
            logger.warning("citation_tracking_failed", paper_id=ss_id, error=str(e))

    discovered = list(all_papers.values())
    logger.info("discovery_raw", total_discovered=len(discovered))

    # Step 3: Preliminary relevance scoring by LLM
    if discovered:
        scored = _preliminary_relevance_score(llm, discovered)
        # Filter out obviously irrelevant papers
        filtered = [
            p for p in scored
            if p.get("preliminary_relevance", 5.0) >= 4.0
        ]
        logger.info(
            "discovery_filtered",
            total=len(scored),
            passed_filter=len(filtered),
            filtered_out=len(scored) - len(filtered),
        )
    else:
        filtered = []

    # Cap at max papers per run
    if len(filtered) > settings.max_papers_per_run:
        # Sort by preliminary relevance, take top N
        filtered.sort(key=lambda p: p.get("preliminary_relevance", 5.0), reverse=True)
        filtered = filtered[:settings.max_papers_per_run]
        logger.info("capped_at_max", max_papers=settings.max_papers_per_run)

    logger.info(
        "discovery_complete",
        total_discovered=len(discovered),
        after_filter=len(filtered),
        queries_used=len(queries),
    )

    return {
        "search_queries": queries,
        "discovered_papers": filtered,
    }
