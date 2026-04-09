"""
OpenAlex API integration — free, open, comprehensive scholarly graph.
200M+ works, no API key needed for most use (polite pool with email).
Docs: https://docs.openalex.org/
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)

OPENALEX_BASE = "https://api.openalex.org"

# OpenAlex concept IDs for our research fields
RELEVANT_CONCEPT_IDS = {
    "artificial_intelligence": "C154945302",
    "machine_learning": "C119857082",
    "software_engineering": "C41008148",
    "computer_security": "C38652104",
    "natural_language_processing": "C204321447",
    "deep_learning": "C108583219",
    "neural_network": "C50644808",
    "multi_agent_system": "C71877743",
    "system_reliability": "C106159729",
}


class OpenAlexTool:
    """Search OpenAlex for papers related to the Integration Paradox."""

    def __init__(self) -> None:
        self.email = settings.openalex_email or "researcher@university.edu"
        self.headers = {"User-Agent": f"mailto:{self.email}"}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def search(
        self,
        query: str,
        days_back: int = 7,
        limit: int = 25,
        concept_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Search OpenAlex works by query string with optional concept filtering."""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params: dict[str, Any] = {
            "search": query,
            "filter": f"from_publication_date:{from_date},type:article|preprint|proceedings-article",
            "per-page": min(limit, 200),
            "select": (
                "id,doi,title,authorships,publication_year,publication_date,"
                "primary_location,best_oa_location,cited_by_count,concepts,"
                "abstract_inverted_index,related_works,referenced_works"
            ),
            "sort": "relevance_score:desc",
            "mailto": self.email,
        }

        if concept_ids:
            concept_filter = "|".join(concept_ids)
            params["filter"] += f",concepts.id:{concept_filter}"

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{OPENALEX_BASE}/works",
                    params=params,
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                logger.info("openalex_search", query=query, found=len(results))
                return [self._normalize(r, query) for r in results]
        except Exception as e:
            logger.error("openalex_search_error", query=query, error=str(e))
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def get_work(self, openalex_id: str) -> Optional[dict[str, Any]]:
        """Get a specific work by OpenAlex ID."""
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{OPENALEX_BASE}/works/{openalex_id}",
                    headers=self.headers,
                )
                resp.raise_for_status()
                return self._normalize(resp.json(), "")
        except Exception as e:
            logger.warning("openalex_get_work_failed", id=openalex_id, error=str(e))
            return None

    def get_related_works(self, openalex_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get works related to a specific paper via OpenAlex's semantic similarity."""
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{OPENALEX_BASE}/works/{openalex_id}",
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json()
                related_ids = data.get("related_works", [])[:limit]

            papers = []
            for oa_id in related_ids[:limit]:
                work = self.get_work(oa_id)
                if work:
                    work["discovery_source"] = "openalex_related"
                    papers.append(work)
                time.sleep(0.1)

            return papers
        except Exception as e:
            logger.warning("openalex_related_works_failed", id=openalex_id, error=str(e))
            return []

    def _reconstruct_abstract(self, abstract_inverted_index: Optional[dict]) -> str:
        """Reconstruct abstract from OpenAlex inverted index format."""
        if not abstract_inverted_index:
            return ""
        # Build word-position pairs and sort by position
        word_positions = []
        for word, positions in abstract_inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)

    def _normalize(self, raw: dict[str, Any], query: str) -> dict[str, Any]:
        doi = (raw.get("doi") or "").replace("https://doi.org/", "")
        openalex_id = raw.get("id", "").replace("https://openalex.org/", "")

        # Authors
        authorships = raw.get("authorships") or []
        authors = []
        for a in authorships:
            author = a.get("author") or {}
            name = author.get("display_name", "")
            if name:
                authors.append(name)

        # Location / URL
        primary_loc = raw.get("primary_location") or {}
        source = primary_loc.get("source") or {}
        venue = source.get("display_name", "")
        url = primary_loc.get("landing_page_url", "")

        best_oa = raw.get("best_oa_location") or {}
        pdf_url = best_oa.get("pdf_url", "")

        # Concepts
        concepts = [c.get("display_name", "") for c in (raw.get("concepts") or [])]

        abstract = self._reconstruct_abstract(raw.get("abstract_inverted_index"))

        year = raw.get("publication_year")
        pub_date = raw.get("publication_date", "")

        stable_id = f"doi:{doi}" if doi else f"openalex:{openalex_id}"
        if not doi and not openalex_id:
            title = raw.get("title", "")
            stable_id = "hash:" + hashlib.md5(title.encode()).hexdigest()[:12]

        return {
            "paper_id": stable_id,
            "title": raw.get("title", ""),
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": doi,
            "arxiv_id": "",
            "semantic_scholar_id": "",
            "openalex_id": openalex_id,
            "venue": venue,
            "citation_count": raw.get("cited_by_count", 0),
            "url": url,
            "pdf_url": pdf_url,
            "fields_of_study": concepts,
            "tldr": "",
            "publication_date": pub_date,
            "discovery_source": "openalex",
            "search_query": query,
            "discovery_date": datetime.now().strftime("%Y-%m-%d"),
        }
