"""
Local paper database using TinyDB for lightweight persistence.
Tracks all discovered, analyzed, and included papers across runs.
Prevents re-processing papers already in the corpus.
"""
from __future__ import annotations

import atexit
import html
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog
from tinydb import TinyDB, Query

from config.settings import settings

logger = structlog.get_logger(__name__)

Paper = Query()


def normalize_doi(doi: Any) -> str:
    """Lower-case a DOI and strip URL/'doi:' prefixes (DOIs are case-insensitive)."""
    if not doi:
        return ""
    d = str(doi).strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", d)


# A markup tag: '<' immediately followed by a tag name (e.g. <i>, </jats:p>, <sup/>), so
# comparisons such as "latency <10 ms for inputs >1 MB" are left alone.
_TAG_RE = re.compile(r"</?[A-Za-z][\w:.-]*(?:\s[^<>]*)?/?>")


def strip_markup(text: Any) -> str:
    """Remove HTML/JATS tags and entities that some APIs leave in titles and abstracts."""
    if not text:
        return ""
    cleaned = _TAG_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", str(text))))
    return re.sub(r"\s+", " ", cleaned).strip()


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _first_author_surname(paper: dict[str, Any]) -> str:
    authors = paper.get("authors") or []
    if not authors or not authors[0]:
        return ""
    name = str(authors[0]).strip()
    surname = name.split(",")[0] if "," in name else (name.split() or [""])[-1]
    return re.sub(r"[^a-z]", "", surname.lower())


def normalize_title(title: Any) -> str:
    """Comparable form of a title: HTML stripped, lower-case, alphanumerics only."""
    if not title:
        return ""
    t = strip_markup(title)
    t = re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    return t if len(t) >= 20 else ""  # very short titles are not distinctive


def dedup_keys(paper: dict[str, Any]) -> set[str]:
    """Keys that identify the same paper across sources (S2, arXiv, OpenAlex, CrossRef)."""
    keys = set()
    if paper.get("paper_id"):
        keys.add(f"id:{paper['paper_id']}")
    doi = normalize_doi(paper.get("doi"))
    if doi:
        keys.add(f"doi:{doi}")
        if doi.startswith("10.48550/arxiv."):
            keys.add(f"arxiv:{doi.split('arxiv.', 1)[1]}")
    arxiv_id = str(paper.get("arxiv_id") or "").strip().lower()
    if arxiv_id:
        keys.add(f"arxiv:{re.sub(r'v[0-9]+$', '', arxiv_id)}")
    title = normalize_title(paper.get("title"))
    if title:
        # Qualified by the first author's surname so two different papers that share a
        # generic title ("... : A Systematic Literature Review") are not merged.
        surname = _first_author_surname(paper)
        keys.add(f"title:{title}|{surname}" if surname else f"title:{title}")
    return keys


_dry_run_copy: Optional[str] = None


def _resolve_db_path(db_path: Optional[str]) -> str:
    """
    In dry-run mode the default database is replaced by a temporary copy, so a dry
    run behaves like a real one but never changes storage/papers_db.json.
    """
    global _dry_run_copy
    path = db_path or settings.papers_db_path
    if db_path is not None or not settings.dry_run:
        return path
    if _dry_run_copy is None:
        copy_dir = tempfile.mkdtemp(prefix="litreview-dryrun-")
        atexit.register(shutil.rmtree, copy_dir, ignore_errors=True)
        _dry_run_copy = os.path.join(copy_dir, "papers_db.json")
        if os.path.exists(path):
            shutil.copyfile(path, _dry_run_copy)
        logger.info("dry_run_database_copy", source=path, copy=_dry_run_copy)
    return _dry_run_copy


# One open database per file path. Separate TinyDB handles on the same file each keep
# their own view of it, so a second handle's write can silently overwrite the first's.
_open_databases: dict[str, "PaperDatabase"] = {}


class PaperDatabase:
    """
    Persistent local database for paper tracking across runs.

    Tables:
      - papers: All analyzed papers with full metadata (their IDs are skipped by later runs)
      - corpus: Papers included in the literature review (analyzed + included)
      - excluded: Papers explicitly excluded with reasons
      - runs: Pipeline run history

    PaperDatabase(path) returns the already-open instance for that path, and every
    write goes straight to disk, so nothing is lost if the process exits early.
    """

    def __new__(cls, db_path: Optional[str] = None) -> "PaperDatabase":
        key = os.path.abspath(_resolve_db_path(db_path))
        existing = _open_databases.get(key)
        if existing is not None:
            return existing
        return super().__new__(cls)

    def __init__(self, db_path: Optional[str] = None) -> None:
        if getattr(self, "_initialized", False):
            return
        self.db_path = _resolve_db_path(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db = TinyDB(self.db_path, indent=2, sort_keys=True, ensure_ascii=False)
        self.papers = self.db.table("papers")
        self.corpus = self.db.table("corpus")
        self.excluded = self.db.table("excluded")
        self.runs = self.db.table("runs")
        self._initialized = True
        _open_databases[os.path.abspath(self.db_path)] = self

    # ── Paper operations ────────────────────────────────────────────────────

    def get_known_ids(self) -> set[str]:
        """Return set of all paper IDs already in our database."""
        all_papers = self.papers.all()
        return {p["paper_id"] for p in all_papers if "paper_id" in p}

    def get_known_keys(self) -> set[str]:
        """Return de-duplication keys (IDs, DOIs, arXiv IDs, titles) of all analyzed papers."""
        keys: set[str] = set()
        for paper in self.papers.all():
            keys |= dedup_keys(paper)
        return keys

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
                "documented": False,           # written into the literature review
                "proposal_documented": False,  # reflected in the research proposal
                **paper,
                "corpus_added_date": datetime.now().strftime("%Y-%m-%d"),
            })
        else:
            self.corpus.update(paper, Paper.paper_id == paper_id)

    def get_undocumented_corpus(self, field: str = "documented") -> list[dict[str, Any]]:
        """
        Corpus papers not yet written into a document (e.g. a doc update failed).
        field: "documented" (literature review) or "proposal_documented".
        Records from before this tracking existed have no flag and are not returned.
        """
        return [p for p in self.corpus.all() if p.get(field) is False]

    def update_corpus_paper(self, paper_id: str, fields: dict[str, Any]) -> None:
        """Update fields of one corpus paper."""
        self.corpus.update(fields, Paper.paper_id == paper_id)

    def mark_documented(self, paper_ids: list[str], field: str = "documented") -> None:
        """Mark corpus papers as written into a document (see get_undocumented_corpus)."""
        for paper_id in paper_ids:
            self.corpus.update({field: True}, Paper.paper_id == paper_id)

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
                sum(_as_float(p.get("relevance_score")) for p in corpus) / len(corpus)
                if corpus else 0.0
            ),
        }

    # ── Run tracking ────────────────────────────────────────────────────────

    def log_run(self, run_data: dict[str, Any]) -> int:
        """Log a pipeline run record. Returns its document ID for later updates."""
        return self.runs.insert({
            **run_data,
            "logged_at": datetime.now().isoformat(),
        })

    def update_run(self, doc_id: int, fields: dict[str, Any]) -> None:
        """Update fields of a previously logged run record."""
        self.runs.update(fields, doc_ids=[doc_id])

    def get_recent_runs(self, n: int = 10) -> list[dict[str, Any]]:
        """Get the N most recent pipeline runs."""
        all_runs = self.runs.all()
        return sorted(all_runs, key=lambda r: r.get("run_date", ""), reverse=True)[:n]

    def close(self) -> None:
        """Close the database file. Writes are already on disk."""
        _open_databases.pop(os.path.abspath(self.db_path), None)
        self._initialized = False
        self.db.close()

    def export_corpus_bibtex(self) -> str:
        """Export all corpus papers as BibTeX (stored entries, generated ones for the rest)."""
        from tools.bibtex import build_entries, existing_keys

        corpus = self.corpus.all()
        taken = {p["citation_key"] for p in corpus if p.get("citation_key")}
        for paper in corpus:
            taken |= existing_keys(paper.get("bibtex") or "")
        generated = build_entries([p for p in corpus if not p.get("bibtex")], taken)
        entries = []
        for paper in corpus:
            if paper.get("bibtex"):
                entries.append(paper["bibtex"])
            elif paper.get("paper_id") in generated:
                entries.append(generated[paper["paper_id"]][1])
        return "\n\n".join(entries)
