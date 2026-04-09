"""
ArXiv API integration using direct HTTP calls (no arxiv library dependency).
The arxiv Python library requires feedparser→sgmllib3k which has build issues on Python 3.11+.
This implementation calls the ArXiv Atom API directly via httpx + lxml.
Docs: https://info.arxiv.org/help/api/
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore

from config.settings import settings

logger = structlog.get_logger(__name__)

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# ArXiv categories most relevant to the Integration Paradox research
TARGET_CATEGORIES = [
    "cs.AI",   # Artificial Intelligence
    "cs.LG",   # Machine Learning
    "cs.SE",   # Software Engineering
    "cs.SY",   # Systems and Control
    "cs.CR",   # Cryptography and Security
    "cs.MA",   # Multiagent Systems
    "cs.RO",   # Robotics
    "stat.ML", # Machine Learning (Statistics)
]


def _ns(tag: str, prefix: str = "atom") -> str:
    return f"{{{ARXIV_NS[prefix]}}}{tag}"


class ArxivTool:
    """Search and retrieve papers from ArXiv using the Atom API directly."""

    def __init__(self) -> None:
        self._last_request = 0.0
        self._min_interval = 3.0  # ArXiv recommends >= 3s between requests

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=3, max=20))
    def search(
        self,
        query: str,
        max_results: int = 20,
        categories: Optional[list[str]] = None,
        days_back: int = 7,
    ) -> list[dict[str, Any]]:
        """Search ArXiv with category filtering and recent-date preference."""
        self._rate_limit()

        cats = categories or TARGET_CATEGORIES
        cat_filter = " OR ".join(f"cat:{c}" for c in cats)
        full_query = f"({query}) AND ({cat_filter})"

        params = {
            "search_query": full_query,
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        logger.info("arxiv_search", query=query[:80], max_results=max_results)

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(ARXIV_API_BASE, params=params)
                resp.raise_for_status()
                papers = self._parse_atom_response(resp.text, query, days_back)
                logger.info("arxiv_search_complete", found=len(papers))
                return papers
        except Exception as e:
            logger.error("arxiv_search_error", error=str(e), query=query[:80])
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=3, max=20))
    def get_paper(self, arxiv_id: str) -> Optional[dict[str, Any]]:
        """Fetch a specific paper by ArXiv ID (e.g., '2401.12345')."""
        self._rate_limit()
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(ARXIV_API_BASE, params={"id_list": arxiv_id, "max_results": 1})
                resp.raise_for_status()
                papers = self._parse_atom_response(resp.text, "", days_back=99999)
                return papers[0] if papers else None
        except Exception as e:
            logger.error("arxiv_get_paper_error", arxiv_id=arxiv_id, error=str(e))
            return None

    def get_integration_paradox_papers(self, days_back: int = 7) -> list[dict[str, Any]]:
        """
        Pre-built multi-query search targeting the Integration Paradox topic.
        Merges results across queries and deduplicates.
        """
        queries = [
            "system reliability AI ML integration failure composition",
            "trustworthy AI system level assurance pipeline",
            "compositional verification machine learning systems",
            "multi-agent AI system reliability failure modes",
            "human AI handoff failure coupled systems",
            "runtime monitoring AI system trust reliability",
        ]

        all_papers: dict[str, dict[str, Any]] = {}
        for q in queries:
            try:
                results = self.search(query=q, max_results=10, days_back=days_back)
                for p in results:
                    pid = p["paper_id"]
                    if pid not in all_papers:
                        all_papers[pid] = p
                time.sleep(self._min_interval)
            except Exception as e:
                logger.warning("arxiv_multi_query_failed", query=q[:50], error=str(e))

        logger.info("arxiv_integration_paradox_search", total_unique=len(all_papers))
        return list(all_papers.values())

    def _parse_atom_response(
        self, xml_text: str, query: str, days_back: int
    ) -> list[dict[str, Any]]:
        """Parse ArXiv Atom XML response into our standard paper format."""
        cutoff = datetime.now() - timedelta(days=days_back)
        papers = []

        try:
            root = ET.fromstring(xml_text.encode("utf-8"))
        except Exception as e:
            logger.error("arxiv_xml_parse_error", error=str(e))
            return []

        for entry in root.findall(_ns("entry")):
            try:
                paper = self._parse_entry(entry, query, cutoff)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning("arxiv_entry_parse_error", error=str(e))

        return papers

    def _parse_entry(
        self, entry: Any, query: str, cutoff: datetime
    ) -> Optional[dict[str, Any]]:
        """Parse a single Atom entry into our Paper format."""
        # ID / ArXiv ID
        entry_id = (entry.findtext(_ns("id")) or "").strip()
        arxiv_id = entry_id.split("/abs/")[-1].split("v")[0] if "/abs/" in entry_id else ""

        # Publication date
        published_str = (entry.findtext(_ns("published")) or "").strip()
        pub_date = None
        if published_str:
            try:
                pub_date = datetime.strptime(published_str[:10], "%Y-%m-%d")
                if pub_date < cutoff:
                    return None  # Too old
            except ValueError:
                pass

        # Title
        title_elem = entry.find(_ns("title"))
        title = (title_elem.text or "").strip().replace("\n", " ") if title_elem is not None else ""

        # Abstract
        summary_elem = entry.find(_ns("summary"))
        abstract = (summary_elem.text or "").strip().replace("\n", " ") if summary_elem is not None else ""

        # Authors
        authors = []
        for author_elem in entry.findall(_ns("author")):
            name_elem = author_elem.find(_ns("name"))
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        # Categories
        categories = []
        for cat_elem in entry.findall(_ns("category")):
            term = cat_elem.get("term", "")
            if term:
                categories.append(term)

        # Links
        pdf_url = ""
        html_url = entry_id
        for link_elem in entry.findall(_ns("link")):
            href = link_elem.get("href", "")
            title_attr = link_elem.get("title", "")
            if "pdf" in title_attr.lower() or "pdf" in href.lower():
                pdf_url = href
            elif link_elem.get("rel") == "alternate":
                html_url = href

        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        # DOI (from arxiv namespace)
        doi = entry.findtext(_ns("doi", "arxiv")) or ""

        year = pub_date.year if pub_date else None
        stable_id = f"arxiv:{arxiv_id}" if arxiv_id else (
            "hash:" + hashlib.md5(title.encode()).hexdigest()[:12]
        )

        venue_label = "arXiv:" + ",".join(categories[:3]) if categories else "arXiv"

        return {
            "paper_id": stable_id,
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "semantic_scholar_id": "",
            "venue": venue_label,
            "citation_count": 0,
            "url": html_url,
            "pdf_url": pdf_url,
            "fields_of_study": categories,
            "tldr": "",
            "publication_date": pub_date.strftime("%Y-%m-%d") if pub_date else "",
            "discovery_source": "arxiv",
            "search_query": query,
            "discovery_date": datetime.now().strftime("%Y-%m-%d"),
        }
