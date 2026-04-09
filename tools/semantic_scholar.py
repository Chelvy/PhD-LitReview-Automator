"""
Semantic Scholar API integration.
Docs: https://api.semanticscholar.org/api-docs/
Rate limits: 100 req/5min (unauthenticated), 1 req/sec (authenticated).
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS_BASE = "https://api.semanticscholar.org/recommendations/v1"

PAPER_FIELDS = (
    "paperId,externalIds,title,abstract,year,authors,venue,publicationVenue,"
    "citationCount,referenceCount,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,"
    "publicationDate,journal,isOpenAccess,openAccessPdf,tldr,url"
)


class SemanticScholarTool:
    """Search and retrieve papers from Semantic Scholar."""

    def __init__(self) -> None:
        self.api_key = settings.semantic_scholar_api_key
        self.headers: dict[str, str] = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
        self._last_request_time = 0.0
        self._min_interval = 1.1 if self.api_key else 3.1  # seconds between requests

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def search_papers(
        self,
        query: str,
        limit: int = 20,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        fields_of_study: Optional[list[str]] = None,
        open_access_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Full-text paper search."""
        self._rate_limit()
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": PAPER_FIELDS,
        }
        if year_from or year_to:
            y_from = year_from or 2000
            y_to = year_to or datetime.now().year
            params["year"] = f"{y_from}-{y_to}"
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if open_access_only:
            params["openAccessPdf"] = ""

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
                    params=params,
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json()
                papers = data.get("data", [])
                logger.info("semantic_scholar_search", query=query, found=len(papers))
                return [self._normalize_paper(p, query, "semantic_scholar_search") for p in papers]
        except httpx.HTTPStatusError as e:
            logger.error("semantic_scholar_http_error", status=e.response.status_code, query=query)
            raise
        except Exception as e:
            logger.error("semantic_scholar_error", error=str(e), query=query)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def get_paper(self, paper_id: str) -> Optional[dict[str, Any]]:
        """Get full paper details by Semantic Scholar ID, DOI, or ArXiv ID."""
        self._rate_limit()
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}",
                    params={"fields": PAPER_FIELDS},
                    headers=self.headers,
                )
                resp.raise_for_status()
                return self._normalize_paper(resp.json(), "", "semantic_scholar_direct")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def get_citations(self, paper_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get papers that cite the given paper (forward citations)."""
        self._rate_limit()
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}/citations",
                    params={"fields": PAPER_FIELDS, "limit": limit},
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
                return [
                    self._normalize_paper(item["citingPaper"], "", "citation_tracking")
                    for item in data
                    if item.get("citingPaper")
                ]
        except Exception as e:
            logger.warning("citation_tracking_failed", paper_id=paper_id, error=str(e))
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def get_references(self, paper_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get papers referenced by the given paper (backward citations)."""
        self._rate_limit()
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}/references",
                    params={"fields": PAPER_FIELDS, "limit": limit},
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
                return [
                    self._normalize_paper(item["citedPaper"], "", "reference_tracking")
                    for item in data
                    if item.get("citedPaper")
                ]
        except Exception as e:
            logger.warning("reference_tracking_failed", paper_id=paper_id, error=str(e))
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def get_author_papers(
        self, author_id: str, limit: int = 10, year_from: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Get recent papers by a tracked author."""
        self._rate_limit()
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{SEMANTIC_SCHOLAR_BASE}/author/{author_id}/papers",
                    params={"fields": PAPER_FIELDS, "limit": limit},
                    headers=self.headers,
                )
                resp.raise_for_status()
                papers = resp.json().get("data", [])
                if year_from:
                    papers = [p for p in papers if (p.get("year") or 0) >= year_from]
                return [self._normalize_paper(p, "", "author_tracking") for p in papers]
        except Exception as e:
            logger.warning("author_tracking_failed", author_id=author_id, error=str(e))
            return []

    def search_recent(
        self, query: str, days_back: int = 7, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search for papers published in the last N days."""
        cutoff_year = (datetime.now() - timedelta(days=days_back)).year
        papers = self.search_papers(query=query, limit=limit, year_from=cutoff_year)
        # Filter by publication date where available
        cutoff = datetime.now() - timedelta(days=days_back)
        recent = []
        for p in papers:
            pub_date = p.get("publication_date")
            if pub_date:
                try:
                    if datetime.strptime(pub_date, "%Y-%m-%d") >= cutoff:
                        recent.append(p)
                except ValueError:
                    recent.append(p)  # keep if date parsing fails
            else:
                recent.append(p)
        return recent

    def _normalize_paper(
        self, raw: dict[str, Any], query: str, source: str
    ) -> dict[str, Any]:
        """Normalize Semantic Scholar paper to our standard Paper format."""
        external_ids = raw.get("externalIds") or {}
        authors = [a.get("name", "") for a in (raw.get("authors") or [])]
        open_access_pdf = raw.get("openAccessPdf") or {}

        paper_id = raw.get("paperId", "")
        doi = external_ids.get("DOI", "")
        arxiv_id = external_ids.get("ArXiv", "")

        # Generate stable ID
        stable_id = paper_id or (f"doi:{doi}" if doi else f"arxiv:{arxiv_id}")
        if not stable_id:
            title = raw.get("title", "")
            stable_id = "hash:" + hashlib.md5(title.encode()).hexdigest()[:12]

        return {
            "paper_id": stable_id,
            "title": raw.get("title", ""),
            "authors": authors,
            "year": raw.get("year"),
            "abstract": raw.get("abstract", ""),
            "doi": doi,
            "arxiv_id": arxiv_id,
            "semantic_scholar_id": paper_id,
            "venue": raw.get("venue", "") or (raw.get("journal") or {}).get("name", ""),
            "citation_count": raw.get("citationCount", 0),
            "url": raw.get("url", f"https://api.semanticscholar.org/paper/{paper_id}"),
            "pdf_url": open_access_pdf.get("url", ""),
            "fields_of_study": [
                f.get("category", "")
                for f in (raw.get("s2FieldsOfStudy") or [])
            ],
            "tldr": (raw.get("tldr") or {}).get("text", ""),
            "publication_date": raw.get("publicationDate", ""),
            "discovery_source": source,
            "search_query": query,
            "discovery_date": datetime.now().strftime("%Y-%m-%d"),
        }
