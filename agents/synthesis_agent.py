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
from tools.paper_db import PaperDatabase
from tools.scite_tool import SciteTool
from agents.llm_utils import build_llm

logger = structlog.get_logger(__name__)


def _build_llm() -> ChatAnthropic:
    return build_llm(temperature=0.2, max_tokens=16000)


def _format_papers_for_synthesis(papers: list[dict[str, Any]]) -> str:
    """Format included papers as a structured text block for the synthesis LLM."""
    blocks = []
    for p in papers:
        gaps = ", ".join(p.get("gaps_addressed", []))
        tags = ", ".join(p.get("synthesis_tags", []))
        block = f"""
PAPER: {p.get('title', 'N/A')} ({p.get('year', 'N/A')})
Authors: {', '.join(p.get('authors', [])[:4])}
Venue: {p.get('venue', 'N/A')} | Citations: {p.get('citation_count', 0)}
Relevance Score: {p.get('relevance_score', 0):.1f}/10 | Gaps: {gaps}
Tags: {tags}

Research Objectives: {p.get('research_objectives', 'N/A')[:300]}
Methodology: {p.get('methodology', 'N/A')[:250]}
Results: {p.get('results', 'N/A')[:300]}
Actual Contribution: {p.get('actual_contribution', p.get('conclusions', 'N/A'))[:300]}
Blank Spots (our analysis): {json.dumps(p.get('wagner_analysis', {}).get('blank_spots', [])[:3])}
Integration Paradox Evidence: {p.get('wagner_analysis', {}).get('integration_paradox_evidence', 'None')}
Critical Notes: {p.get('critical_notes', 'N/A')}
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
  "papers_synthesized": [{json.dumps([p.get('paper_id') for p in included_papers])}],
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
        response = llm.invoke([system, HumanMessage(content=prompt)])
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        synthesis_report: SynthesisReport = json.loads(content)
        logger.info(
            "synthesis_complete",
            papers_synthesized=len(synthesis_report.get("papers_synthesized", [])),
            debates=len(synthesis_report.get("key_debates_update", [])),
            contradictions=len(synthesis_report.get("contradictions", [])),
        )

    except Exception as e:
        logger.error("synthesis_failed", error=str(e))
        synthesis_report = {
            "synthesis_date": run_date,
            "papers_synthesized": [p.get("paper_id") for p in included_papers],
            "gap_status_update": gap_status,
            "historical_narrative_update": {"new_developments": f"Synthesis failed: {e}"},
            "key_debates_update": [],
            "contradictions": [],
            "anchor_papers": [],
            "methodology_recommendations": [],
            "contribution_update": "See research questions for contribution statement",
            "lit_review_sections_to_update": [],
            "proposal_sections_to_update": [],
        }

    return {"synthesis_report": synthesis_report}
