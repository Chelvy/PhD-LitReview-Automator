"""
Agent 6: Visualization & Mapping Agent

Generates structured data files for literature maps, citation networks,
and conceptual diagrams. Produces:
  - Citation network (JSON for Gephi/VOSviewer/D3.js)
  - Concept/topic map (Mermaid diagram)
  - Gap coverage matrix (Markdown table)
  - Research timeline (Mermaid Gantt / timeline)
  - Contradiction map (papers that contradict each other)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from config.settings import settings
from state.pipeline_state import PipelineState, VisualizationData
from tools.paper_db import PaperDatabase

logger = structlog.get_logger(__name__)

VIZ_OUTPUT_DIR = Path("outputs/visualizations")


def _build_citation_network(
    corpus: list[dict[str, Any]], new_papers: list[dict[str, Any]]
) -> VisualizationData:
    """Build a citation network JSON for Gephi/D3.js visualization."""
    nodes = []
    edges = []
    seen_ids = set()

    all_papers = corpus
    new_ids = {p.get("paper_id") for p in new_papers}

    for paper in all_papers:
        pid = paper.get("paper_id", "")
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        nodes.append({
            "id": pid,
            "label": paper.get("title", "Unknown")[:50],
            "year": paper.get("year", 0),
            "citation_count": paper.get("citation_count", 0),
            "relevance_score": paper.get("relevance_score", 0.0),
            "gaps_addressed": paper.get("gaps_addressed", []),
            "citation_priority": paper.get("citation_priority", "medium"),
            "is_new": pid in new_ids,
            "size": max(5, min(50, paper.get("citation_count", 0) // 10 + 10)),
            "group": paper.get("synthesis_tags", ["untagged"])[0] if paper.get("synthesis_tags") else "untagged",
        })

        # Add edges from extend/contradict relationships
        for related_id in (paper.get("related_papers") or []):
            if related_id != pid:
                edges.append({
                    "source": pid,
                    "target": related_id,
                    "type": "related",
                    "weight": 1,
                })
        for contradicts_id in (paper.get("contradicts_papers") or []):
            if contradicts_id != pid:
                edges.append({
                    "source": pid,
                    "target": contradicts_id,
                    "type": "contradicts",
                    "weight": 2,
                })

    return {
        "visualization_type": "citation_network",
        "data": {"nodes": nodes, "edges": edges},
        "recommended_tool": "Gephi or D3.js",
        "notes": f"Network of {len(nodes)} papers, {len(edges)} connections. "
                 f"Highlight nodes where is_new=true for today's additions.",
    }


def _build_concept_map_mermaid(synthesis_report: dict[str, Any]) -> VisualizationData:
    """Build a Mermaid concept map of the research landscape."""
    debates = synthesis_report.get("key_debates_update", [])
    gaps = synthesis_report.get("gap_status_update", {})

    mermaid_lines = [
        "graph TD",
        '    IP["🔬 Integration Paradox"]',
        '    SLR["System-Level Reliability"]',
        '    CLR["Component-Level Reliability"]',
        '    TC["Trust Composition"]',
        '    IM["Interface Misalignment"]',
        '    HAI["Human-AI Handoff"]',
        '    RM["Runtime Monitoring"]',
        '    SDLC["SDLC Practices"]',
        '    REG["Regulation (EU AI Act)"]',
        "",
        "    IP --> SLR",
        "    IP --> CLR",
        "    SLR --> TC",
        "    SLR --> IM",
        "    SLR --> HAI",
        "    SLR --> RM",
        "    TC --> SDLC",
        "    IM --> SDLC",
        "    REG --> SDLC",
        "    CLR -.->|'Insufficient alone'| SLR",
        "",
    ]

    # Add gap nodes
    gap_colors = {
        "absent": "fill:#ffcccc",
        "nascent": "fill:#ffe0cc",
        "static": "fill:#fff3cc",
        "emerging": "fill:#d4f0cc",
        "growing": "fill:#ccf0d4",
    }
    for gap_id, gap_data in gaps.items():
        trajectory = gap_data.get("trajectory", "static")
        color = gap_colors.get(trajectory, "fill:#eee")
        count = gap_data.get("papers_addressing", 0)
        mermaid_lines.append(
            f'    {gap_id.replace("-", "")}["{gap_id}\\n{count} papers\\n{trajectory}"]'
        )
        mermaid_lines.append(f'    style {gap_id.replace("-", "")} {color}')

    mermaid = "\n".join(mermaid_lines)

    return {
        "visualization_type": "concept_map",
        "data": {"mermaid_source": mermaid},
        "recommended_tool": "Mermaid.js (https://mermaid.live)",
        "mermaid_diagram": mermaid,
        "notes": "Paste into https://mermaid.live to render. Red=absent gap, green=growing.",
    }


def _build_gap_matrix(
    corpus: list[dict[str, Any]], run_date: str
) -> VisualizationData:
    """Build a gap coverage matrix as Markdown table."""
    gap_ids = [f"GAP-{i}" for i in range(1, 7)]
    gap_descriptions = {
        "GAP-1": "Component-centric dominance",
        "GAP-2": "No trust-composition calculus",
        "GAP-3": "Underdeveloped predictors",
        "GAP-4": "Non-monotone reliability",
        "GAP-5": "Assurance evidence composition",
        "GAP-6": "Socio-technical formalization",
    }

    # Count papers per gap and collect key papers
    gap_papers: dict[str, list[str]] = {g: [] for g in gap_ids}
    for paper in corpus:
        for gap in (paper.get("gaps_addressed") or []):
            if gap in gap_papers:
                gap_papers[gap].append(
                    f"{paper.get('authors', ['?'])[0].split(',')[0]} ({paper.get('year', '?')})"
                )

    # Build Markdown table
    rows = [
        f"| {g} | {gap_descriptions.get(g, '')} | {len(gap_papers[g])} | "
        f"{', '.join(gap_papers[g][:3])}{'...' if len(gap_papers[g]) > 3 else ''} |"
        for g in gap_ids
    ]

    table = (
        f"# Gap Coverage Matrix — {run_date}\n\n"
        "| Gap ID | Description | Papers | Key Authors |\n"
        "|--------|-------------|--------|-------------|\n"
        + "\n".join(rows)
    )

    return {
        "visualization_type": "gap_matrix",
        "data": {"table_markdown": table, "gap_counts": {g: len(gap_papers[g]) for g in gap_ids}},
        "recommended_tool": "Markdown renderer or LaTeX longtable",
        "notes": f"Total corpus: {len(corpus)} papers. Updated {run_date}.",
    }


def _build_timeline(corpus: list[dict[str, Any]]) -> VisualizationData:
    """Build a research timeline Mermaid diagram."""
    # Group papers by year
    by_year: dict[int, list[str]] = {}
    for paper in corpus:
        year = paper.get("year")
        if year and isinstance(year, int) and 2010 <= year <= 2030:
            title = paper.get("title", "Unknown")[:40]
            by_year.setdefault(year, []).append(title)

    if not by_year:
        return {
            "visualization_type": "timeline",
            "data": {},
            "mermaid_diagram": "timeline\n    title Research Timeline\n    section 2024\n        No papers yet",
            "recommended_tool": "Mermaid.js",
            "notes": "Empty corpus — timeline will populate as papers are added.",
        }

    mermaid_lines = ["timeline", "    title Integration Paradox Research Timeline"]

    for year in sorted(by_year.keys()):
        mermaid_lines.append(f"    section {year}")
        for title in by_year[year][:3]:  # Max 3 per year for readability
            mermaid_lines.append(f"        {title}")

    mermaid = "\n".join(mermaid_lines)

    return {
        "visualization_type": "timeline",
        "data": {"by_year": {str(k): v for k, v in by_year.items()}},
        "recommended_tool": "Mermaid.js",
        "mermaid_diagram": mermaid,
        "notes": "Timeline of corpus papers by publication year.",
    }


def run_visualization(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: Visualization & Mapping Agent.
    Generates/updates all visualization data files.
    """
    synthesis_report = state.get("synthesis_report", {})
    included_papers = state.get("included_papers", [])
    run_date = state.get("run_date", datetime.now().strftime("%Y-%m-%d"))

    logger.info("visualization_start", new_papers=len(included_papers))

    db = PaperDatabase()
    corpus = db.get_corpus()

    # Generate all visualizations
    visualizations: list[VisualizationData] = []

    # 1. Citation network
    network_viz = _build_citation_network(corpus, included_papers)
    visualizations.append(network_viz)

    # 2. Concept map (Mermaid)
    concept_viz = _build_concept_map_mermaid(synthesis_report)
    visualizations.append(concept_viz)

    # 3. Gap coverage matrix
    gap_viz = _build_gap_matrix(corpus, run_date)
    visualizations.append(gap_viz)

    # 4. Timeline
    timeline_viz = _build_timeline(corpus)
    visualizations.append(timeline_viz)

    # Save to files
    if not settings.dry_run:
        VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Citation network JSON
        network_path = VIZ_OUTPUT_DIR / "citation_network.json"
        network_path.write_text(
            json.dumps(network_viz["data"], indent=2), encoding="utf-8"
        )

        # Concept map Mermaid
        concept_path = VIZ_OUTPUT_DIR / "concept_map.mmd"
        concept_path.write_text(concept_viz.get("mermaid_diagram", ""), encoding="utf-8")

        # Gap matrix Markdown
        gap_path = VIZ_OUTPUT_DIR / "gap_matrix.md"
        gap_path.write_text(gap_viz["data"].get("table_markdown", ""), encoding="utf-8")

        # Timeline Mermaid
        timeline_path = VIZ_OUTPUT_DIR / "timeline.mmd"
        timeline_path.write_text(timeline_viz.get("mermaid_diagram", ""), encoding="utf-8")

        logger.info("visualizations_saved", dir=str(VIZ_OUTPUT_DIR))

    return {"visualizations": visualizations}
