"""
Scite API integration — smart citations with claim classification.
Scite classifies each citation as: supporting, contrasting, or mentioning.
Docs: https://scite.ai/api
Key insight: use contrasting citations to find contradictions in the literature.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)

SCITE_BASE = "https://api.scite.ai"


class SciteTool:
    """
    Scite smart citation tool.
    Primary uses:
      1. Find papers that CONTRADICT a key claim (blank-spot analysis fuel)
      2. Find papers with high "supporting" counts (high-confidence findings)
      3. Find recent papers citing our anchor papers
    """

    def __init__(self) -> None:
        self.api_key = settings.scite_api_key
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logger.warning("scite_disabled", reason="No SCITE_API_KEY configured")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def search(
        self,
        query: str,
        limit: int = 20,
        min_supporting: int = 0,
    ) -> list[dict[str, Any]]:
        """Search Scite for papers with citation quality signals."""
        if not self.enabled:
            return []

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{SCITE_BASE}/search/papers",
                    params={"term": query, "limit": limit},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                papers = data.get("papers", [])
                if min_supporting > 0:
                    papers = [
                        p for p in papers
                        if p.get("tallies", {}).get("supporting", 0) >= min_supporting
                    ]
                logger.info("scite_search", query=query, found=len(papers))
                return [self._normalize(p, query) for p in papers]
        except Exception as e:
            logger.error("scite_search_error", error=str(e), query=query)
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def get_citations_for_doi(
        self,
        doi: str,
        citation_type: str = "all",  # "supporting" | "contrasting" | "mentioning" | "all"
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get papers citing a DOI with citation type classification.
        citation_type="contrasting" is gold for contradiction detection.
        """
        if not self.enabled:
            return []

        try:
            params: dict[str, Any] = {"limit": limit}
            if citation_type != "all":
                params["type"] = citation_type

            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{SCITE_BASE}/citations/{doi}",
                    params=params,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                cites = resp.json().get("citations", [])
                logger.info(
                    "scite_citations",
                    doi=doi,
                    type=citation_type,
                    found=len(cites)
                )
                return [self._normalize_citation(c, doi) for c in cites]
        except Exception as e:
            logger.warning("scite_citations_failed", doi=doi, error=str(e))
            return []

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    def get_tally(self, doi: str) -> dict[str, int]:
        """Get citation tally for a paper: supporting, contrasting, mentioning counts."""
        if not self.enabled:
            return {}

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    f"{SCITE_BASE}/tallies/{doi}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                return resp.json().get("tally", {})
        except Exception:
            return {}

    def find_contradicted_claims(
        self, anchor_dois: list[str]
    ) -> list[dict[str, Any]]:
        """
        For a list of anchor papers, find all papers that CONTRADICT them.
        This is core to blank-spot analysis: contradictions reveal debate zones.
        """
        contrasting = []
        for doi in anchor_dois:
            papers = self.get_citations_for_doi(doi, citation_type="contrasting", limit=30)
            for p in papers:
                p["contradicts_doi"] = doi
                contrasting.append(p)
            time.sleep(0.5)
        return contrasting

    def _normalize(self, raw: dict[str, Any], query: str) -> dict[str, Any]:
        from datetime import datetime
        import hashlib

        doi = raw.get("doi", "")
        title = raw.get("title", "")
        stable_id = f"doi:{doi}" if doi else "hash:" + hashlib.md5(title.encode()).hexdigest()[:12]

        tallies = raw.get("tallies", {})

        return {
            "paper_id": stable_id,
            "title": title,
            "authors": raw.get("authors", []),
            "year": raw.get("year"),
            "abstract": raw.get("abstract", ""),
            "doi": doi,
            "arxiv_id": "",
            "semantic_scholar_id": "",
            "venue": raw.get("journal", ""),
            "citation_count": sum(tallies.values()),
            "url": f"https://scite.ai/reports/{doi}" if doi else "",
            "pdf_url": "",
            "fields_of_study": [],
            "tldr": "",
            # Scite-specific fields
            "scite_supporting": tallies.get("supporting", 0),
            "scite_contrasting": tallies.get("contrasting", 0),
            "scite_mentioning": tallies.get("mentioning", 0),
            "discovery_source": "scite",
            "search_query": query,
            "discovery_date": datetime.now().strftime("%Y-%m-%d"),
        }

    def _normalize_citation(self, raw: dict[str, Any], anchor_doi: str) -> dict[str, Any]:
        from datetime import datetime
        import hashlib

        source = raw.get("source", {}) or {}
        doi = source.get("doi", "")
        title = source.get("title", "")
        stable_id = f"doi:{doi}" if doi else "hash:" + hashlib.md5(title.encode()).hexdigest()[:12]

        return {
            "paper_id": stable_id,
            "title": title,
            "authors": source.get("authors", []),
            "year": source.get("year"),
            "abstract": "",
            "doi": doi,
            "arxiv_id": "",
            "venue": source.get("journal", ""),
            "citation_count": 0,
            "url": f"https://scite.ai/reports/{doi}" if doi else "",
            "pdf_url": "",
            "fields_of_study": [],
            "tldr": "",
            "scite_citation_type": raw.get("type", ""),
            "scite_excerpt": raw.get("excerpt", ""),
            "scite_anchor_doi": anchor_doi,
            "discovery_source": "scite_citation_tracking",
            "search_query": f"citations of {anchor_doi}",
            "discovery_date": datetime.now().strftime("%Y-%m-%d"),
        }
