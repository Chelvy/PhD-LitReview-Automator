"""
CrossRef API integration via the `habanero` library.
Provides DOI metadata, publication tracking, and journal-level search.
Docs: https://www.crossref.org/documentation/retrieve-metadata/rest-api/
Rate limit: polite pool with email header (50 req/sec). Use mailto param.
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from habanero import Crossref
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)

# Key venues for Integration Paradox research
TARGET_ISSNS = {
    "IEEE Transactions on Software Engineering": "0098-5589",
    "IEEE Transactions on Reliability": "0018-9529",
    "ACM Computing Surveys": "0360-0300",
    "Empirical Software Engineering": "1382-3256",
    "Journal of Machine Learning Research": "1532-4435",
    "Nature Machine Intelligence": "2522-5839",
    "Artificial Intelligence": "0004-3702",
    "IEEE Software": "0740-7459",
}

TARGET_CONFERENCE_DOIS = [
    "10.1145/3597503",  # ICSE
    "10.1145/3611643",  # FSE
    "10.18653",          # ACL anthology
]


class CrossRefTool:
    """Search CrossRef for DOI metadata and journal/conference papers."""

    def __init__(self) -> None:
        self.cr = Crossref(
            mailto=settings.openalex_email or "researcher@university.edu"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def search(
        self,
        query: str,
        limit: int = 20,
        days_back: int = 7,
        issn: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search CrossRef by query with optional journal ISSN filter."""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        kwargs: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "filter": {"from_pub_date": from_date},
            "sort": "relevance",
        }
        if issn:
            kwargs["filter"]["issn"] = issn

        try:
            result = self.cr.works(**kwargs)
            items = result.get("message", {}).get("items", [])
            logger.info("crossref_search", query=query, found=len(items))
            return [self._normalize(item, query) for item in items if item.get("title")]
        except Exception as e:
            logger.error("crossref_search_error", error=str(e), query=query)
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def get_by_doi(self, doi: str) -> Optional[dict[str, Any]]:
        """Fetch full metadata for a paper by DOI."""
        try:
            result = self.cr.works(ids=doi)
            item = result.get("message", {})
            return self._normalize(item, "") if item else None
        except Exception as e:
            logger.warning("crossref_doi_lookup_failed", doi=doi, error=str(e))
            return None

    def search_key_venues(self, query: str, days_back: int = 7) -> list[dict[str, Any]]:
        """Search across key venues for Integration Paradox research."""
        all_papers: dict[str, dict[str, Any]] = {}
        for venue_name, issn in TARGET_ISSNS.items():
            results = self.search(query=query, limit=10, days_back=days_back, issn=issn)
            for p in results:
                pid = p["paper_id"]
                if pid not in all_papers:
                    p["venue"] = venue_name
                    all_papers[pid] = p
            time.sleep(0.5)  # Polite rate limiting

        return list(all_papers.values())

    def _normalize(self, item: dict[str, Any], query: str) -> dict[str, Any]:
        doi = item.get("DOI", "")
        titles = item.get("title", [])
        title = titles[0] if titles else ""

        authors_raw = item.get("author", [])
        authors = []
        for a in authors_raw:
            given = a.get("given", "")
            family = a.get("family", "")
            if family:
                authors.append(f"{family}, {given}".strip(", "))

        pub = item.get("published", {})
        date_parts = pub.get("date-parts", [[None]])[0]
        year = date_parts[0] if date_parts else None
        pub_date = "-".join(str(p) for p in date_parts if p) if date_parts else ""

        container = item.get("container-title", [])
        venue = container[0] if container else ""

        abstract = item.get("abstract", "")
        # CrossRef abstracts may have JATS XML markup
        if abstract:
            import re
            abstract = re.sub(r"<[^>]+>", " ", abstract).strip()

        stable_id = f"doi:{doi}" if doi else (
            "hash:" + hashlib.md5(title.encode()).hexdigest()[:12]
        )

        return {
            "paper_id": stable_id,
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": doi,
            "arxiv_id": "",
            "semantic_scholar_id": "",
            "venue": venue,
            "citation_count": item.get("is-referenced-by-count", 0),
            "url": item.get("URL", f"https://doi.org/{doi}"),
            "pdf_url": "",
            "fields_of_study": item.get("subject", []),
            "tldr": "",
            "publication_date": pub_date,
            "discovery_source": "crossref",
            "search_query": query,
            "discovery_date": datetime.now().strftime("%Y-%m-%d"),
        }
