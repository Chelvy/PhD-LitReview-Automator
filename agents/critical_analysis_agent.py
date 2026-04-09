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

logger = structlog.get_logger(__name__)


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0.15,
        max_tokens=8096,
    )


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
    paper_text = f"""PAPER TO ANALYZE:

Title: {paper.get('title', 'N/A')}
Authors: {', '.join(paper.get('authors', [])[:5])}
Year: {paper.get('year', 'N/A')}
Venue: {paper.get('venue', 'N/A')} | Citations: {paper.get('citation_count', 0)}

Preliminary Relevance: {paper.get('preliminary_relevance', 'N/A')}/10
Pre-screening Reason: {paper.get('preliminary_relevance_reason', 'N/A')}

EXTRACTED DATA:
Research Objectives: {paper.get('research_objectives', paper.get('abstract', 'N/A'))[:800]}
Methodology: {paper.get('methodology', 'N/A')[:600]}
Results: {paper.get('results', 'N/A')[:600]}
Conclusions: {paper.get('conclusions', 'N/A')[:400]}
Key Claims: {json.dumps(paper.get('key_claims', [])[:5])}
Theoretical Contribution: {paper.get('theoretical_contribution', 'N/A')}
Empirical Contribution: {paper.get('empirical_contribution', 'N/A')}
Assumptions: {json.dumps(paper.get('assumptions', [])[:5])}
Author-Acknowledged Limitations (Blind Spots): {json.dumps(paper.get('blind_spots', [])[:5])}
Pre-extracted Blank Spots: {json.dumps(paper.get('blank_spots', [])[:5])}
Key Contributions: {json.dumps(paper.get('key_contributions', [])[:5])}

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

    try:
        response = llm.invoke([system, HumanMessage(content=prompt)])
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        analysis = json.loads(content)
        logger.info(
            "paper_analyzed",
            paper_id=paper.get("paper_id"),
            score=analysis.get("relevance_score"),
            include=analysis.get("inclusion_decision"),
        )
        return analysis

    except json.JSONDecodeError as e:
        logger.warning("analysis_parse_failed", paper_id=paper.get("paper_id"), error=str(e))
        # Fallback: use preliminary score
        prelim = paper.get("preliminary_relevance", 5.0)
        return {
            "paper_id": paper.get("paper_id", ""),
            "rq1_relevance": prelim * 0.9,
            "rq2_relevance": prelim * 0.9,
            "rq3_relevance": prelim * 0.8,
            "relevance_score": prelim,
            "inclusion_decision": prelim >= settings.min_relevance_score,
            "exclusion_reason": None if prelim >= settings.min_relevance_score else "Below relevance threshold (parse error fallback)",
            "gaps_addressed": [],
            "wagner_analysis": {
                "blank_spots": [],
                "blind_spots": paper.get("blind_spots", []),
                "actual_contribution": paper.get("abstract", "")[:200],
                "assumes_component_isolation": True,
                "integration_paradox_evidence": None,
            },
            "synthesis_tags": [],
            "citation_priority": "medium" if prelim >= 7 else "low",
            "critical_notes": "Analysis parse failed — manual review recommended",
        }
    except Exception as e:
        logger.error("analysis_error", paper_id=paper.get("paper_id"), error=str(e))
        raise


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
        f"Score: {p.get('relevance_score', 0):.1f}, Gaps: {p.get('gaps_addressed', [])}"
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

    for paper in extracted_papers:
        paper_id = paper.get("paper_id", "unknown")

        try:
            # Get Scite citation quality signals if available
            doi = paper.get("doi", "")
            if scite.enabled and doi:
                tally = scite.get_tally(doi)
                if tally:
                    paper["scite_supporting"] = tally.get("supporting", 0)
                    paper["scite_contrasting"] = tally.get("contrasting", 0)
                    paper["scite_mentioning"] = tally.get("mentioning", 0)

            # Run critical analysis
            analysis = _analyze_paper(llm, paper, corpus_context)

            # Apply citation boost to relevance score
            citation_count = paper.get("citation_count", 0)
            citation_boost = min(
                settings.relevance_citation_boost * (citation_count / 100),
                2.0
            )
            adjusted_score = min(
                analysis.get("relevance_score", 5.0) + citation_boost,
                10.0
            )
            analysis["relevance_score"] = round(adjusted_score, 2)

            # Override inclusion decision based on adjusted score
            if analysis["relevance_score"] < settings.min_relevance_score:
                analysis["inclusion_decision"] = False
                if not analysis.get("exclusion_reason"):
                    analysis["exclusion_reason"] = (
                        f"Relevance score {analysis['relevance_score']:.1f} below threshold {settings.min_relevance_score}"
                    )

            # Merge analysis into paper
            full_paper = {**paper, **analysis}
            analyzed.append(full_paper)

            # Persist to database
            db.save_paper(full_paper)
            if full_paper.get("inclusion_decision"):
                included.append(full_paper)
                db.add_to_corpus(full_paper)
            else:
                excluded.append(full_paper)
                db.add_to_excluded(full_paper, full_paper.get("exclusion_reason", "Below threshold"))

        except Exception as e:
            logger.error("paper_analysis_error", paper_id=paper_id, error=str(e))
            analyzed.append({**paper, "analysis_error": str(e), "inclusion_decision": False})
            excluded.append(paper)

    logger.info(
        "analysis_complete",
        analyzed=len(analyzed),
        included=len(included),
        excluded=len(excluded),
    )

    return {
        "analyzed_papers": analyzed,
        "included_papers": included,
        "excluded_papers": excluded,
    }
