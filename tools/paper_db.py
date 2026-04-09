"""
Local paper database using TinyDB for lightweight persistence.
Tracks all discovered, analyzed, and included papers across runs.
Prevents re-processing papers already in the corpus.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog
from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

from config.settings import settings

logger = structlog.get_logger(__name__)

Paper = Query()


class PaperDatabase:
    """
    Persistent local database for paper tracking across daily runs.

    Tables:
      - papers: All discovered papers with full metadata
      - corpus: Papers included in the literature review (analyzed + included)
      - excluded: Papers explicitly excluded with reasons
      - runs: Pipeline run history
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.papers_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db = TinyDB(
            self.db_path,
            storage=CachingMiddleware(JSONStorage),
            indent=2,
            sort_keys=True,
        )
        self.papers = self.db.table("papers")
        self.corpus = self.db.table("corpus")
        self.excluded = self.db.table("excluded")
        self.runs = self.db.table("runs")

    # ── Paper operations ────────────────────────────────────────────────────

    def get_known_ids(self) -> set[str]:
        """Return set of all paper IDs already in our database."""
        all_papers = self.papers.all()
        return {p["paper_id"] for p in all_papers if "paper_id" in p}

    def get_corpus_ids(self) -> set[str]:
        """Return set of paper IDs in our active corpus (included papers)."""
        return {p["paper_id"] for p in self.corpus.all() if "paper_id" in p}

    def paper_exists(self, paper_id: str) -> bool:
        """Check if a paper is already in the database."""
        return bool(self.papers.get(Paper.paper_id == paper_id))

    def save_paper(self, paper: dict[str, Any]) -> None:
        """Insert or update a paper record."""
        paper_id = paper.get("paper_id")
        if not paper_id:
            logger.warning("save_paper_no_id", title=paper.get("title", "unknown"))
            return

        existing = self.papers.get(Paper.paper_id == paper_id)
        if existing:
            self.papers.update(paper, Paper.paper_id == paper_id)
        else:
            self.papers.insert(paper)

    def save_papers_batch(self, papers: list[dict[str, Any]]) -> int:
        """Save multiple papers, returning count of new papers added."""
        new_count = 0
        for paper in papers:
            if not self.paper_exists(paper.get("paper_id", "")):
                self.save_paper(paper)
                new_count += 1
            else:
                self.save_paper(paper)  # Update existing with new fields
        return new_count

    def add_to_corpus(self, paper: dict[str, Any]) -> None:
        """Add paper to the active corpus (included in literature review)."""
        paper_id = paper.get("paper_id")
        if not self.corpus.get(Paper.paper_id == paper_id):
            self.corpus.insert({
                **paper,
                "corpus_added_date": datetime.now().strftime("%Y-%m-%d"),
            })
        else:
            self.corpus.update(paper, Paper.paper_id == paper_id)

    def add_to_excluded(self, paper: dict[str, Any], reason: str) -> None:
        """Record an excluded paper so we don't re-analyze it."""
        paper_id = paper.get("paper_id")
        if not self.excluded.get(Paper.paper_id == paper_id):
            self.excluded.insert({
                **paper,
                "exclusion_reason": reason,
                "excluded_date": datetime.now().strftime("%Y-%m-%d"),
            })

    def get_corpus(self) -> list[dict[str, Any]]:
        """Return all papers in the active corpus."""
        return self.corpus.all()

    def get_papers_by_gap(self, gap_id: str) -> list[dict[str, Any]]:
        """Get corpus papers addressing a specific gap (e.g., 'GAP-1')."""
        return [
            p for p in self.corpus.all()
            if gap_id in (p.get("gaps_addressed") or [])
        ]

    def get_high_relevance_papers(self, min_score: float = 8.0) -> list[dict[str, Any]]:
        """Get corpus papers above a relevance threshold."""
        return [
            p for p in self.corpus.all()
            if (p.get("relevance_score") or 0.0) >= min_score
        ]

    def get_anchor_papers(self) -> list[dict[str, Any]]:
        """Get papers marked as anchor/foundation papers."""
        return [
            p for p in self.corpus.all()
            if p.get("citation_priority") == "high"
        ]

    def search_corpus(self, keywords: list[str]) -> list[dict[str, Any]]:
        """Simple keyword search across corpus titles and abstracts."""
        results = []
        for paper in self.corpus.all():
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            if any(kw.lower() in text for kw in keywords):
                results.append(paper)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Return database statistics."""
        corpus = self.corpus.all()
        gaps_coverage: dict[str, int] = {}
        for paper in corpus:
            for gap in (paper.get("gaps_addressed") or []):
                gaps_coverage[gap] = gaps_coverage.get(gap, 0) + 1

        return {
            "total_discovered": len(self.papers.all()),
            "total_corpus": len(corpus),
            "total_excluded": len(self.excluded.all()),
            "gaps_coverage": gaps_coverage,
            "total_runs": len(self.runs.all()),
            "avg_relevance": (
                sum(p.get("relevance_score", 0) for p in corpus) / len(corpus)
                if corpus else 0.0
            ),
        }

    # ── Run tracking ────────────────────────────────────────────────────────

    def log_run(self, run_data: dict[str, Any]) -> None:
        """Log a pipeline run record."""
        self.runs.insert({
            **run_data,
            "logged_at": datetime.now().isoformat(),
        })

    def get_recent_runs(self, n: int = 10) -> list[dict[str, Any]]:
        """Get the N most recent pipeline runs."""
        all_runs = self.runs.all()
        return sorted(all_runs, key=lambda r: r.get("run_date", ""), reverse=True)[:n]

    def close(self) -> None:
        """Flush cache and close the database."""
        self.db.storage.flush()
        self.db.close()

    def export_corpus_bibtex(self) -> str:
        """Export all corpus papers as a BibTeX file."""
        lines = []
        for paper in self.corpus.all():
            bibtex = paper.get("bibtex", "")
            if bibtex:
                lines.append(bibtex)
            else:
                # Generate minimal BibTeX
                authors = paper.get("authors", [])
                author_str = " and ".join(authors[:5])
                year = paper.get("year", "YYYY")
                key = paper.get("paper_id", "unknown").replace(":", "_")[:20]
                title = paper.get("title", "Unknown Title")
                venue = paper.get("venue", "")
                doi = paper.get("doi", "")
                lines.append(
                    f"@article{{{key},\n"
                    f"  author = {{{author_str}}},\n"
                    f"  title = {{{title}}},\n"
                    f"  year = {{{year}}},\n"
                    f"  journal = {{{venue}}},\n"
                    f"  doi = {{{doi}}}\n"
                    f"}}"
                )
        return "\n\n".join(lines)
