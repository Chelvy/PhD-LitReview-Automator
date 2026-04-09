"""
ArXiv API integration using the official `arxiv` Python library.
Targets: cs.AI, cs.LG, cs.SE, cs.SY, cs.CR, stat.ML, cs.RO
Docs: https://info.arxiv.org/help/api/
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)

# ArXiv categories most relevant to the Integration Paradox research
TARGET_CATEGORIES = [
    "cs.AI",   # Artificial Intelligence
    "cs.LG",   # Machine Learning
    "cs.SE",   # Software Engineering
    "cs.SY",   # Systems and Control
    "cs.CR",   # Cryptography and Security (adversarial reliability)
    "cs.MA",   # Multiagent Systems
    "cs.RO",   # Robotics (system integration failures)
    "stat.ML", # Machine Learning (Statistics)
]


class ArxivTool:
    """Search and retrieve papers from ArXiv."""

    def __init__(self) -> None:
        self.client = arxiv.Client(
            page_size=50,
            delay_seconds=3.0,
            num_retries=3,
        )

    def search(
        self,
        query: str,
        max_results: int = 20,
        categories: Optional[list[str]] = None,
        days_back: int = 7,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
    ) -> list[dict[str, Any]]:
        """Search ArXiv with category filtering and date range."""
        cats = categories or TARGET_CATEGORIES

        # Build category filter
        cat_filter = " OR ".join(f"cat:{c}" for c in cats)
        full_query = f"({query}) AND ({cat_filter})"

        logger.info("arxiv_search", query=full_query, max_results=max_results)

        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=arxiv.SortOrder.Descending,
        )

        cutoff = datetime.now() - timedelta(days=days_back)
        papers = []

        try:
            for result in self.client.results(search):
                # Filter by submission date
                submitted = result.published.replace(tzinfo=None) if result.published else None
                if submitted and submitted < cutoff:
                    continue  # Skip older papers when sorted by date

                papers.append(self._normalize(result, query))

                if len(papers) >= max_results:
                    break
        except Exception as e:
            logger.error("arxiv_search_error", error=str(e), query=query)

        logger.info("arxiv_search_complete", found=len(papers))
        return papers

    def get_paper(self, arxiv_id: str) -> Optional[dict[str, Any]]:
        """Fetch a specific paper by ArXiv ID (e.g., '2401.12345')."""
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(self.client.results(search))
            if results:
                return self._normalize(results[0], "")
        except Exception as e:
            logger.error("arxiv_get_paper_error", arxiv_id=arxiv_id, error=str(e))
        return None

    def search_by_author(
        self, author_name: str, days_back: int = 7, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Find recent papers by a tracked author."""
        query = f'au:"{author_name}"'
        return self.search(query=query, max_results=max_results, days_back=days_back)

    def get_integration_paradox_papers(self, days_back: int = 7) -> list[dict[str, Any]]:
        """
        Pre-built search targeting the Integration Paradox topic.
        Uses multiple query terms and merges results.
        """
        queries = [
            "system reliability AI ML integration failure composition",
            "trustworthy AI system level assurance pipeline",
            "compositional verification machine learning systems",
            "AI component integration error propagation",
            "multi-agent system reliability failure modes",
            "interface mismatch AI pipeline distribution shift",
            "human AI handoff failure coupled systems",
            "runtime monitoring AI system trust",
        ]

        all_papers: dict[str, dict[str, Any]] = {}
        for q in queries:
            results = self.search(query=q, max_results=15, days_back=days_back)
            for p in results:
                pid = p["paper_id"]
                if pid not in all_papers:
                    all_papers[pid] = p

        logger.info("arxiv_integration_paradox_search", total_unique=len(all_papers))
        return list(all_papers.values())

    def _normalize(self, result: arxiv.Result, query: str) -> dict[str, Any]:
        """Normalize ArXiv result to our standard Paper format."""
        arxiv_id = result.get_short_id() if hasattr(result, "get_short_id") else str(result.entry_id).split("/")[-1]
        doi = result.doi or ""

        stable_id = f"arxiv:{arxiv_id}"

        # Get PDF URL
        pdf_url = next(
            (str(link.href) for link in result.links if "pdf" in str(link.href).lower()),
            f"https://arxiv.org/pdf/{arxiv_id}",
        )

        authors = [str(a) for a in (result.authors or [])]

        pub_date = result.published
        year = pub_date.year if pub_date else None

        categories = result.categories or []

        return {
            "paper_id": stable_id,
            "title": result.title or "",
            "authors": authors,
            "year": year,
            "abstract": result.summary or "",
            "doi": doi,
            "arxiv_id": arxiv_id,
            "semantic_scholar_id": "",
            "venue": "arXiv:" + ",".join(categories[:3]),
            "citation_count": 0,  # ArXiv doesn't provide citation counts
            "url": str(result.entry_id),
            "pdf_url": pdf_url,
            "fields_of_study": categories,
            "tldr": "",
            "publication_date": pub_date.strftime("%Y-%m-%d") if pub_date else "",
            "discovery_source": "arxiv",
            "search_query": query,
            "discovery_date": datetime.now().strftime("%Y-%m-%d"),
        }
