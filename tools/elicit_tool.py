"""
Elicit API integration — AI research assistant with structured paper extraction.
Docs: https://elicit.com/api
Provides: paper search, structured extraction, concept synthesis.
Requires: ELICIT_API_KEY (paid tier for bulk research use)
"""
from __future__ import annotations

import time
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)

ELICIT_BASE = "https://api.elicit.com/v1"


class ElicitTool:
    """
    Elicit API for deep academic search and structured paper extraction.
    Falls back gracefully if API key is not configured.
    """

    def __init__(self) -> None:
        self.api_key = settings.elicit_api_key
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logger.warning("elicit_disabled", reason="No ELICIT_API_KEY configured")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def search(
        self,
        query: str,
        num_papers: int = 20,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search Elicit for papers relevant to the query.
        Elicit is especially strong for: finding empirical papers, comparing
        methodologies, and structured extraction from research questions.
        """
        if not self.enabled:
            return []

        payload = {
            "query": query,
            "numPapers": num_papers,
            "filters": filters or {},
            "extractColumns": [
                "research_question",
                "methodology",
                "sample_size",
                "outcome_variables",
                "key_findings",
                "limitations",
            ],
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{ELICIT_BASE}/papers/search",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                papers = data.get("papers", [])
                logger.info("elicit_search", query=query, found=len(papers))
                return [self._normalize(p, query) for p in papers]
        except httpx.HTTPStatusError as e:
            logger.error("elicit_http_error", status=e.response.status_code, query=query)
            if e.response.status_code == 429:
                time.sleep(60)
            raise
        except Exception as e:
            logger.error("elicit_error", error=str(e), query=query)
            raise

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    def extract_paper(self, doi_or_url: str) -> Optional[dict[str, Any]]:
        """
        Extract structured information from a specific paper.
        Returns: research objectives, methodology, results, conclusions.
        """
        if not self.enabled:
            return None

        try:
            with httpx.Client(timeout=90.0) as client:
                resp = client.post(
                    f"{ELICIT_BASE}/papers/extract",
                    json={
                        "identifier": doi_or_url,
                        "extractColumns": [
                            "research_question",
                            "population",
                            "intervention",
                            "outcome",
                            "methodology",
                            "sample_size",
                            "key_findings",
                            "limitations",
                            "dataset_used",
                            "evaluation_metrics",
                        ],
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning("elicit_extract_failed", identifier=doi_or_url, error=str(e))
            return None

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    def synthesize(self, question: str, paper_ids: list[str]) -> Optional[str]:
        """
        Ask Elicit to synthesize findings from a list of papers on a question.
        Useful for gap analysis and contradiction detection.
        """
        if not self.enabled:
            return None

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{ELICIT_BASE}/synthesis",
                    json={"question": question, "paperIds": paper_ids},
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                return resp.json().get("synthesis", "")
        except Exception as e:
            logger.warning("elicit_synthesis_failed", question=question, error=str(e))
            return None

    def _normalize(self, raw: dict[str, Any], query: str) -> dict[str, Any]:
        from datetime import datetime
        import hashlib

        doi = raw.get("doi", "")
        title = raw.get("title", "")
        stable_id = f"doi:{doi}" if doi else "hash:" + hashlib.md5(title.encode()).hexdigest()[:12]

        authors = raw.get("authors", [])
        if isinstance(authors, list) and authors and isinstance(authors[0], dict):
            authors = [a.get("name", "") for a in authors]

        return {
            "paper_id": stable_id,
            "title": title,
            "authors": authors,
            "year": raw.get("year"),
            "abstract": raw.get("abstract", ""),
            "doi": doi,
            "arxiv_id": raw.get("arxiv_id", ""),
            "semantic_scholar_id": raw.get("semanticScholarId", ""),
            "venue": raw.get("venue", ""),
            "citation_count": raw.get("citation_count", 0),
            "url": raw.get("url", f"https://doi.org/{doi}" if doi else ""),
            "pdf_url": raw.get("pdf_url", ""),
            "fields_of_study": raw.get("fields", []),
            "tldr": raw.get("tldr", ""),
            # Elicit-specific structured extraction (bonus fields)
            "elicit_research_question": raw.get("research_question", ""),
            "elicit_methodology": raw.get("methodology", ""),
            "elicit_key_findings": raw.get("key_findings", ""),
            "elicit_limitations": raw.get("limitations", ""),
            "discovery_source": "elicit",
            "search_query": query,
            "discovery_date": datetime.now().strftime("%Y-%m-%d"),
        }
