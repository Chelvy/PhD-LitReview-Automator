"""
Scholarcy API integration — automated research paper summarization.
Extracts: highlights, key concepts, methods, datasets, findings.
Docs: https://ref.scholarcy.com/api/
"""
from __future__ import annotations

import time
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)

SCHOLARCY_BASE = "https://ref.scholarcy.com/api"


class ScholarcyTool:
    """
    Scholarcy PDF extraction tool.
    Best for: fast structured extraction from PDFs when full-text analysis needed.
    Provides: highlights, key statements, study design, findings, limitations.
    """

    def __init__(self) -> None:
        self.api_key = settings.scholarcy_api_key
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logger.warning("scholarcy_disabled", reason="No SCHOLARCY_API_KEY configured")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def extract_from_url(self, pdf_url: str) -> Optional[dict[str, Any]]:
        """
        Extract structured summary from a PDF URL.
        Returns Scholarcy's full extraction including highlights, methods,
        datasets, key findings, and reference list.
        """
        if not self.enabled:
            return None

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{SCHOLARCY_BASE}/flashcard",
                    data={"url": pdf_url},
                    headers={"x-api-key": self.api_key},
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info("scholarcy_extracted", url=pdf_url, sections=list(data.keys()))
                return self._normalize(data, pdf_url)
        except httpx.HTTPStatusError as e:
            logger.error("scholarcy_http_error", status=e.response.status_code, url=pdf_url)
            return None
        except Exception as e:
            logger.warning("scholarcy_extract_failed", url=pdf_url, error=str(e))
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def extract_from_doi(self, doi: str) -> Optional[dict[str, Any]]:
        """Extract from a paper identified by DOI."""
        if not self.enabled:
            return None

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{SCHOLARCY_BASE}/flashcard",
                    data={"doi": doi},
                    headers={"x-api-key": self.api_key},
                )
                resp.raise_for_status()
                return self._normalize(resp.json(), f"doi:{doi}")
        except Exception as e:
            logger.warning("scholarcy_doi_failed", doi=doi, error=str(e))
            return None

    def extract_batch(
        self, papers: list[dict[str, Any]], max_concurrent: int = 3
    ) -> list[dict[str, Any]]:
        """
        Extract structured data from a batch of papers.
        Respects rate limits with sequential processing and delays.
        """
        results = []
        for paper in papers:
            pdf_url = paper.get("pdf_url", "")
            doi = paper.get("doi", "")

            extraction = None
            if pdf_url:
                extraction = self.extract_from_url(pdf_url)
            elif doi:
                extraction = self.extract_from_doi(doi)

            if extraction:
                # Merge extraction data into paper dict
                merged = {**paper, **extraction}
                results.append(merged)
            else:
                results.append(paper)  # Return original without enrichment

            time.sleep(2.0)  # Respectful rate limiting

        return results

    def _normalize(self, raw: dict[str, Any], source: str) -> dict[str, Any]:
        """
        Map Scholarcy response to our Paper extraction fields.
        Scholarcy keys: highlights, key_concepts, study_design, methods,
        datasets, findings, limitations, contributions, references
        """
        highlights = raw.get("highlights", [])
        findings = raw.get("findings", raw.get("key_findings", []))
        methods = raw.get("methods", raw.get("study_design", []))
        limitations = raw.get("limitations", [])
        datasets = raw.get("datasets", [])
        contributions = raw.get("contributions", [])

        # Scholarcy often returns lists; join for our string fields
        def join_list(val: Any) -> str:
            if isinstance(val, list):
                return " ".join(str(v) for v in val)
            return str(val) if val else ""

        return {
            "scholarcy_highlights": highlights if isinstance(highlights, list) else [highlights],
            "methodology": join_list(methods),
            "datasets_used": datasets if isinstance(datasets, list) else [],
            "results": join_list(findings),
            "key_contributions": contributions if isinstance(contributions, list) else [contributions],
            "blind_spots": limitations if isinstance(limitations, list) else [str(limitations)],
            "scholarcy_source": source,
            "extraction_complete": True,
        }
