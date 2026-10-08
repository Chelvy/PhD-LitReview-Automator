"""
Agent 3: Critical Analysis Agent

Applies Jon Wagner's blank-spot framework to every extracted paper.
Makes definitive inclusion/exclusion decisions with full justification.
Scores relevance against each RQ and overall.
Identifies gaps addressed and produces synthesis-ready tags.

This is the intellectual core of the pipeline — every decision must be
analytically grounded, not just keyword-matched.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from config.prompts import AgentPrompts
from state.pipeline_state import PipelineState
from tools.paper_db import PaperDatabase
from tools.scite_tool import SciteTool
from agents.llm_utils import build_llm, invoke_json, is_systemic_error, list_field, text_field

logger = structlog.get_logger(__name__)


# The LLM echoes paper_id back; never let analysis output overwrite identity fields.
IDENTITY_FIELDS = ("paper_id", "title", "authors", "year", "doi", "venue", "citation_count")


def _build_llm():
    return build_llm(temperature=0.15, max_tokens=16000, effort="high")


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_list(paper: dict[str, Any], key: str, n: int = 5) -> str:
    return json.dumps(list_field(paper, key)[:n], ensure_ascii=False)


def _analyze_paper(
    llm: ChatAnthropic,
    paper: dict[str, Any],
    corpus_context: str,
) -> dict[str, Any]:
    """
    Apply Wagner framework analysis to a single paper.
    Returns structured analysis with inclusion decision and relevance scores.
    """
    system = SystemMessage(content=AgentPrompts.CRITICAL_ANALYSIS)

    # Build paper summary for the LLM
    objectives = text_field(paper, "research_objectives", "") or text_field(paper, "abstract")
    paper_text = f"""PAPER TO ANALYZE:

Title: {text_field(paper, 'title')}
Authors: {', '.join(str(a) for a in list_field(paper, 'authors')[:5])}
Year: {text_field(paper, 'year')}
Venue: {text_field(paper, 'venue')} | Citations: {paper.get('citation_count') or 0}

Preliminary Relevance: {text_field(paper, 'preliminary_relevance')}/10
Pre-screening Reason: {text_field(paper, 'preliminary_relevance_reason')}

EXTRACTED DATA:
Research Objectives: {objectives[:800]}
Methodology: {text_field(paper, 'methodology', limit=600)}
Results: {text_field(paper, 'results', limit=600)}
Conclusions: {text_field(paper, 'conclusions', limit=400)}
Key Claims: {_json_list(paper, 'key_claims')}
Theoretical Contribution: {text_field(paper, 'theoretical_contribution', limit=600)}
Empirical Contribution: {text_field(paper, 'empirical_contribution', limit=600)}
Assumptions: {_json_list(paper, 'assumptions')}
Author-Acknowledged Limitations (Blind Spots): {_json_list(paper, 'blind_spots')}
Pre-extracted Blank Spots: {_json_list(paper, 'blank_spots')}
Key Contributions: {_json_list(paper, 'key_contributions')}

EXISTING CORPUS CONTEXT (for avoiding redundancy):
{corpus_context[:1000]}
"""

    prompt = f"""{paper_text}

Apply the Wagner blank-spot critical analysis framework and produce a complete
critical analysis. Be ANALYTICAL and SPECIFIC — avoid vague assessments.

Key questions to answer:
1. Does this paper study system-level reliability, or does it stay at the component level?
2. What does it ACTUALLY contribute to our understanding of the Integration Paradox?
3. What are the BLANK SPOTS — what should it have studied but didn't?
4. Which of our 6 research gaps does it address (GAP-1 through GAP-6)?
5. Does it inadvertently demonstrate the integration paradox? (Papers showing system
   degradation despite component quality are especially valuable evidence)

Return ONLY a JSON object:
{{
  "paper_id": "{paper.get('paper_id', '')}",
  "rq1_relevance": <0.0-10.0>,
  "rq2_relevance": <0.0-10.0>,
  "rq3_relevance": <0.0-10.0>,
  "relevance_score": <0.0-10.0 weighted composite>,
  "inclusion_decision": <true|false>,
  "exclusion_reason": <"reason" or null>,
  "gaps_addressed": ["GAP-1", ...],
  "wagner_analysis": {{
    "blank_spots": ["Specific analytical insight: what they could have done but didn't", ...],
    "blind_spots": ["What authors themselves acknowledge as limitations", ...],
    "actual_contribution": "Precise statement of what this paper actually contributes to our field",
    "assumes_component_isolation": <true|false>,
    "integration_paradox_evidence": <"Description of paradox evidence found" or null>
  }},
  "methodological_assessment": {{
    "rigor_score": <0.0-10.0>,
    "sample_size": "Description of empirical scope",
    "replication_possible": <true|false>,
    "generalizability": "Assessment of how broadly findings apply",
    "concerns": ["Specific methodological concern 1", ...]
  }},
  "synthesis_tags": ["tag1", "tag2", ...],
  "citation_priority": "high|medium|low",
  "critical_notes": "2-3 sentence synthesis of why this paper matters (or doesn't) for our research"
}}"""

    analysis = invoke_json(llm, [system, HumanMessage(content=prompt)])
    if not isinstance(analysis, dict):
        raise ValueError(f"Expected a JSON object from analysis, got {type(analysis).__name__}")
    logger.info(
        "paper_analyzed",
        paper_id=paper.get("paper_id"),
        score=analysis.get("relevance_score"),
        include=analysis.get("inclusion_decision"),
    )
    return analysis


def _build_corpus_context(db: PaperDatabase) -> str:
    """
    Build a brief context string summarizing the existing corpus
    to help the LLM avoid redundant inclusions.
    """
    corpus = db.get_corpus()
    if not corpus:
        return "No existing corpus. This is the first run."

    anchor_papers = [
        f"  - {p.get('title', 'Unknown')} ({p.get('year', 'N/A')}) — "
        f"Score: {_as_float(p.get('relevance_score'), 0.0):.1f}, Gaps: {list_field(p, 'gaps_addressed')}"
        for p in corpus[:15]  # Show top 15
    ]

    stats = db.get_stats()
    gaps = stats.get("gaps_coverage", {})

    return (
        f"EXISTING CORPUS ({len(corpus)} papers):\n"
        f"Gap coverage: {json.dumps(gaps)}\n"
        f"Sample papers:\n" + "\n".join(anchor_papers)
    )


def run_critical_analysis(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: Critical Analysis Agent.
    Analyzes each extracted paper and makes inclusion/exclusion decisions.
    """
    extracted_papers = state.get("extracted_papers", [])
    if not extracted_papers:
        logger.info("analysis_skip", reason="no_extracted_papers")
        return {
            "analyzed_papers": [],
            "included_papers": [],
            "excluded_papers": [],
        }

    logger.info("analysis_start", papers_to_analyze=len(extracted_papers))

    llm = _build_llm()
    db = PaperDatabase()
    scite = SciteTool()

    corpus_context = _build_corpus_context(db)

    analyzed = []
    included = []
    excluded = []
    analysis_errors = []

    def analyze(paper: dict[str, Any]) -> dict[str, Any]:
        """Scite signals + LLM analysis for one paper (no database access, safe to run in parallel)."""
        doi = paper.get("doi", "")
        if scite.enabled and doi:
            tally = scite.get_tally(doi)
            if tally:
                paper["scite_supporting"] = tally.get("supporting", 0)
                paper["scite_contrasting"] = tally.get("contrasting", 0)
                paper["scite_mentioning"] = tally.get("mentioning", 0)
        return _analyze_paper(llm, paper, corpus_context)

    def analyze_safely(paper: dict[str, Any]) -> Any:
        try:
            return analyze(paper)
        except Exception as e:  # handled per paper below
            return e

    with ThreadPoolExecutor(max_workers=max(1, settings.llm_concurrency)) as pool:
        outcomes = list(pool.map(analyze_safely, extracted_papers))

    for paper, outcome in zip(extracted_papers, outcomes):
        paper_id = paper.get("paper_id", "unknown")

        try:
            if isinstance(outcome, Exception):
                raise outcome
            analysis = {k: v for k, v in outcome.items() if k not in IDENTITY_FIELDS}

            # Apply citation boost to relevance score
            citation_count = _as_float(paper.get("citation_count"), 0.0)
            citation_boost = min(
                settings.relevance_citation_boost * (citation_count / 100),
                2.0
            )
            adjusted_score = min(
                _as_float(analysis.get("relevance_score"), 0.0) + citation_boost,
                10.0
            )
            analysis["relevance_score"] = round(adjusted_score, 2)
            analysis["inclusion_decision"] = bool(analysis.get("inclusion_decision"))
            analysis["gaps_addressed"] = [g for g in list_field(analysis, "gaps_addressed") if isinstance(g, str)]

            # Downstream stages (synthesis, documents, email) read these at top level
            wagner = analysis.get("wagner_analysis") or {}
            if isinstance(wagner, dict):
                analysis.setdefault("actual_contribution", wagner.get("actual_contribution") or "")
                analysis.setdefault("integration_paradox_evidence", wagner.get("integration_paradox_evidence"))

            # Override inclusion decision based on adjusted score
            if analysis["relevance_score"] < settings.min_relevance_score:
                analysis["inclusion_decision"] = False
                if not analysis.get("exclusion_reason"):
                    analysis["exclusion_reason"] = (
                        f"Relevance score {analysis['relevance_score']:.1f} below threshold {settings.min_relevance_score}"
                    )

            # Merge analysis into paper
            full_paper = {**paper, **analysis, "analysis_date": datetime.now().strftime("%Y-%m-%d")}
            analyzed.append(full_paper)

            # Persist to database — only successfully analyzed papers, so that a paper
            # whose analysis failed is picked up again by the next run.
            db.save_paper(full_paper)
            if full_paper["inclusion_decision"]:
                included.append(full_paper)
                db.add_to_corpus(full_paper)
            else:
                excluded.append(full_paper)
                db.add_to_excluded(full_paper, full_paper.get("exclusion_reason") or "Below threshold")

        except Exception as e:
            logger.error("paper_analysis_error", paper_id=paper_id, error=str(e))
            analysis_errors.append({
                "paper_id": paper_id,
                "title": text_field(paper, "title", "", limit=120),
                "error": str(e)[:500],
                "timestamp": datetime.now().isoformat(),
            })

    logger.info(
        "analysis_complete",
        analyzed=len(analyzed),
        included=len(included),
        excluded=len(excluded),
        failed=len(analysis_errors),
    )

    result: dict[str, Any] = {
        "analyzed_papers": analyzed,
        "included_papers": included,
        "excluded_papers": excluded,
        "analysis_errors": analysis_errors,
    }
    # Every paper failing points at a systemic problem (API key, model, outage), not bad papers
    if analysis_errors and not analyzed and (
        len(extracted_papers) >= 3 or is_systemic_error(analysis_errors[0]["error"])
    ):
        result["errors"] = [{
            "node": "critical_analysis",
            "error": f"All {len(extracted_papers)} analyses failed; first error: {analysis_errors[0]['error']}",
            "timestamp": datetime.now().isoformat(),
        }]
    return result
