"""
Agent 4: Synthesis & Gap Identification Agent

Synthesizes across all newly included papers:
  - Updates historical narrative
  - Maps active debates
  - Quantifies gap coverage (GAP-1 through GAP-6)
  - Detects contradictions between papers
  - Proposes specific document updates

This agent produces the synthesis_report used by the Document Update Agent.
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
from state.pipeline_state import PipelineState, SynthesisReport
from tools.paper_db import Paper, PaperDatabase
from tools.scite_tool import SciteTool
from agents.llm_utils import build_llm, invoke_json, list_field, text_field

logger = structlog.get_logger(__name__)


def _build_llm():
    return build_llm(temperature=0.2, max_tokens=20000, effort="high")


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_papers_for_synthesis(papers: list[dict[str, Any]]) -> str:
    """Format included papers as a structured text block for the synthesis LLM."""
    blocks = []
    for p in papers:
        gaps = ", ".join(str(g) for g in list_field(p, "gaps_addressed"))
        tags = ", ".join(str(t) for t in list_field(p, "synthesis_tags"))
        wagner = p.get("wagner_analysis") if isinstance(p.get("wagner_analysis"), dict) else {}
        contribution = text_field(p, "actual_contribution", "") or text_field(p, "conclusions")
        block = f"""
PAPER: {text_field(p, 'title')} ({text_field(p, 'year')})
Authors: {', '.join(str(a) for a in list_field(p, 'authors')[:4])}
Venue: {text_field(p, 'venue')} | Citations: {p.get('citation_count') or 0}
Relevance Score: {_score(p.get('relevance_score')):.1f}/10 | Gaps: {gaps}
Tags: {tags}

Research Objectives: {text_field(p, 'research_objectives', limit=300)}
Methodology: {text_field(p, 'methodology', limit=250)}
Results: {text_field(p, 'results', limit=300)}
Actual Contribution: {contribution[:300]}
Blank Spots (our analysis): {json.dumps(list_field(wagner, 'blank_spots')[:3], ensure_ascii=False)}
Integration Paradox Evidence: {wagner.get('integration_paradox_evidence') or 'None'}
Critical Notes: {text_field(p, 'critical_notes', limit=600)}
"""
        blocks.append(block)
    return "\n---\n".join(blocks)


def _get_gap_status(db: PaperDatabase) -> dict[str, dict[str, Any]]:
    """Calculate current gap coverage from corpus."""
    corpus = db.get_corpus()
    gap_counts: dict[str, int] = {f"GAP-{i}": 0 for i in range(1, 7)}

    for paper in corpus:
        for gap in (paper.get("gaps_addressed") or []):
            if gap in gap_counts:
                gap_counts[gap] += 1

    # Trajectory assessment (simplified heuristic)
    trajectories = {
        "GAP-1": "emerging" if gap_counts["GAP-1"] < 5 else "growing",
        "GAP-2": "nascent" if gap_counts["GAP-2"] < 3 else "emerging",
        "GAP-3": "growing" if gap_counts["GAP-3"] > 5 else "emerging",
        "GAP-4": "static",
        "GAP-5": "nascent" if gap_counts["GAP-5"] < 3 else "emerging",
        "GAP-6": "absent" if gap_counts["GAP-6"] == 0 else "nascent",
    }

    return {
        gap: {
            "papers_addressing": count,
            "trajectory": trajectories.get(gap, "static"),
            "today_update": "",  # Will be filled by LLM
        }
        for gap, count in gap_counts.items()
    }


def run_synthesis(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: Synthesis & Gap Identification.
    Synthesizes newly included papers and produces the synthesis report.
    """
    included_papers = state.get("included_papers", [])
    run_date = state.get("run_date", datetime.now().strftime("%Y-%m-%d"))

    if not included_papers:
        logger.info("synthesis_skip", reason="no_included_papers")
        return {
            "synthesis_report": {
                "synthesis_date": run_date,
                "papers_synthesized": [],
                "gap_status_update": _get_gap_status(PaperDatabase()),
                "historical_narrative_update": {"new_developments": "No new papers today."},
                "key_debates_update": [],
                "contradictions": [],
                "anchor_papers": [],
                "methodology_recommendations": [],
                "contribution_update": settings.research_questions[0][:200],
                "lit_review_sections_to_update": [],
                "proposal_sections_to_update": [],
            }
        }

    logger.info("synthesis_start", papers=len(included_papers))

    llm = _build_llm()
    db = PaperDatabase()
    scite = SciteTool()

    # Get current gap status
    gap_status = _get_gap_status(db)
    gap_json = json.dumps(gap_status, indent=2)

    # Format papers for synthesis
    papers_text = _format_papers_for_synthesis(included_papers)

    # Get corpus context for contradiction detection
    existing_corpus = db.get_corpus()
    corpus_titles = [p.get("title", "") for p in existing_corpus[:30]]

    # Find Scite contradictions for key anchor papers
    contradiction_hints = []
    if scite.enabled:
        anchor_papers = db.get_anchor_papers()
        for anchor in anchor_papers[:5]:
            doi = anchor.get("doi", "")
            if doi:
                contrasting = scite.get_citations_for_doi(doi, citation_type="contrasting", limit=10)
                for c in contrasting:
                    new_id = c.get("paper_id", "")
                    if any(p.get("paper_id") == new_id for p in included_papers):
                        contradiction_hints.append(
                            f"Paper '{c.get('title', 'N/A')}' has a CONTRASTING citation to anchor paper '{anchor.get('title', 'N/A')}'"
                        )

    system = SystemMessage(content=AgentPrompts.SYNTHESIS)

    prompt = f"""Today is {run_date}. Synthesize these {len(included_papers)} newly included papers.

EXISTING CORPUS SIZE: {len(existing_corpus)} papers
CURRENT GAP STATUS:
{gap_json}

EXISTING CORPUS SAMPLE TITLES (for context):
{json.dumps(corpus_titles[:20])}

SCITE CONTRADICTION HINTS:
{json.dumps(contradiction_hints)}

TODAY'S NEWLY INCLUDED PAPERS:
{papers_text}

Produce a comprehensive synthesis report. For each gap, update the 'today_update' field
with what today's papers contribute. Identify any contradictions between today's papers
and the existing corpus. Propose specific document updates (section by section).

Remember:
- Our study's contribution is a UNIFIED, ENGINEER-USABLE THEORY linking architecture +
  interfaces + operational drift to system-level trust, then to actionable SDLC practices
- Focus on what the literature STILL lacks (strengthens our justification)
- Identify empirical evidence of the integration paradox (component OK → system fails)

Return the full synthesis report as JSON matching this schema:
{{
  "synthesis_date": "{run_date}",
  "papers_synthesized": {json.dumps([p.get('paper_id') for p in included_papers])},
  "historical_narrative_update": {{
    "new_developments": "...",
    "timeline_update": "...",
    "dominant_paradigm_shifts": [...]
  }},
  "key_debates_update": [
    {{
      "debate": "...",
      "current_state": "...",
      "new_evidence": "...",
      "our_position": "..."
    }}
  ],
  "gap_status_update": {{
    "GAP-1": {{"papers_addressing": <N>, "trajectory": "...", "today_update": "..."}},
    "GAP-2": {{"papers_addressing": <N>, "trajectory": "...", "today_update": "..."}},
    "GAP-3": {{"papers_addressing": <N>, "trajectory": "...", "today_update": "..."}},
    "GAP-4": {{"papers_addressing": <N>, "trajectory": "...", "today_update": "..."}},
    "GAP-5": {{"papers_addressing": <N>, "trajectory": "...", "today_update": "..."}},
    "GAP-6": {{"papers_addressing": <N>, "trajectory": "...", "today_update": "..."}}
  }},
  "contradictions": [...],
  "anchor_papers": ["Title of must-cite foundational paper", ...],
  "methodology_recommendations": ["Method to consider for our research", ...],
  "contribution_update": "Updated statement of what our study contributes that the corpus still lacks",
  "lit_review_sections_to_update": [
    {{
      "section_path": "2.3.1 Subsection Name",
      "action": "create_subsection|append_paragraph|update_paragraph",
      "content_summary": "What to write/update",
      "papers_to_cite": ["paper_id1", ...]
    }}
  ],
  "proposal_sections_to_update": [
    {{
      "section_path": "3.2 Methodology",
      "action": "strengthen|add_validation",
      "content_summary": "What to add/strengthen",
      "papers_to_cite": ["paper_id1", ...]
    }}
  ]
}}"""

    try:
        synthesis_report: SynthesisReport = invoke_json(llm, [system, HumanMessage(content=prompt)])
        if not isinstance(synthesis_report, dict):
            raise ValueError(f"Expected a JSON object from synthesis, got {type(synthesis_report).__name__}")
        # Keep the computed gap counts if the LLM omitted or mangled the gap table
        if not isinstance(synthesis_report.get("gap_status_update"), dict):
            synthesis_report["gap_status_update"] = gap_status
        _record_contradictions(db, synthesis_report)
        logger.info(
            "synthesis_complete",
            papers_synthesized=len(synthesis_report.get("papers_synthesized", [])),
            debates=len(synthesis_report.get("key_debates_update", [])),
            contradictions=len(synthesis_report.get("contradictions", [])),
        )

    except Exception as e:
        logger.error("synthesis_failed", error=str(e))
        return {
            "synthesis_report": _fallback_report(run_date, included_papers, gap_status, e),
            "synthesis_failed": True,
            "errors": [{
                "node": "synthesis",
                "error": f"Synthesis failed ({str(e)[:300]}); document updates postponed to the next run",
                "timestamp": datetime.now().isoformat(),
            }],
        }

    return {"synthesis_report": synthesis_report}


def _record_contradictions(db: PaperDatabase, report: dict[str, Any]) -> None:
    """Store detected contradictions on the corpus papers (edges of the citation network)."""
    for contradiction in list_field(report, "contradictions"):
        if not isinstance(contradiction, dict):
            continue
        ids = [i for i in list_field(contradiction, "papers") if isinstance(i, str)]
        for paper_id in ids:
            others = [o for o in ids if o != paper_id]
            existing = db.corpus.get(Paper.paper_id == paper_id)
            if existing and others:
                merged = sorted(set(existing.get("contradicts_papers") or []) | set(others))
                db.update_corpus_paper(paper_id, {"contradicts_papers": merged})


def _fallback_report(run_date: str, included_papers: list[dict[str, Any]], gap_status: dict[str, Any], error: Exception) -> dict[str, Any]:
    """Minimal synthesis report so the email can still be produced."""
    return {
        "synthesis_date": run_date,
        "papers_synthesized": [p.get("paper_id") for p in included_papers],
        "gap_status_update": gap_status,
        "historical_narrative_update": {"new_developments": f"Synthesis failed: {str(error)[:300]}"},
        "key_debates_update": [],
        "contradictions": [],
        "anchor_papers": [],
        "methodology_recommendations": [],
        "contribution_update": "See research questions for contribution statement",
        "lit_review_sections_to_update": [],
        "proposal_sections_to_update": [],
    }
