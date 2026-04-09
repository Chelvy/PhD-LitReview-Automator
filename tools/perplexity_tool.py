"""
Perplexity API integration with academic mode.
Best for: deep research queries, finding non-indexed papers,
author background, research landscape synthesis.
Docs: https://docs.perplexity.ai/
"""
from __future__ import annotations

import time
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)

PERPLEXITY_BASE = "https://api.perplexity.ai"

# Academic-grade models available in Perplexity API
ACADEMIC_MODEL = "llama-3.1-sonar-large-128k-online"  # Online model with web search


class PerplexityTool:
    """
    Perplexity API for deep research queries with real-time web access.
    Use for: landscape surveys, author profiles, finding non-indexed preprints,
    regulatory document searches (EU AI Act, NIST AI RMF, ISO/IEC 42001).
    """

    def __init__(self) -> None:
        self.api_key = settings.perplexity_api_key
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logger.warning("perplexity_disabled", reason="No PERPLEXITY_API_KEY configured")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
    def research_query(
        self,
        query: str,
        focus: str = "academic",  # "academic" | "internet" | "writing"
        return_citations: bool = True,
    ) -> Optional[dict[str, Any]]:
        """
        Run a deep research query via Perplexity's online model.
        Returns: answer text + list of cited sources.
        """
        if not self.enabled:
            return None

        system_prompt = (
            "You are a research assistant for a PhD-level literature review on AI systems reliability "
            "and the Integration Paradox. Focus on peer-reviewed publications, preprints on arXiv, "
            "official standards documents (EU AI Act, ISO/IEC 42001, NIST AI RMF), and technical "
            "reports from major AI labs and research institutions. Provide specific paper titles, "
            "authors, years, and venues where possible."
        )

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{PERPLEXITY_BASE}/chat/completions",
                    json={
                        "model": ACADEMIC_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query},
                        ],
                        "return_citations": return_citations,
                        "search_recency_filter": "month",  # Focus on recent work
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                citations = data.get("citations", [])

                logger.info("perplexity_query", query=query[:80], citations_found=len(citations))

                return {
                    "query": query,
                    "response": content,
                    "citations": citations,  # List of URL strings
                    "model": data.get("model", ACADEMIC_MODEL),
                }
        except Exception as e:
            logger.error("perplexity_query_failed", error=str(e), query=query[:80])
            return None

    def landscape_survey(self, topic: str) -> Optional[str]:
        """
        Get a quick landscape survey of recent developments in a topic.
        Returns a narrative summary with key papers mentioned.
        """
        query = (
            f"What are the most significant papers, preprints, and technical reports published "
            f"in the last 6 months on: {topic}? For each, give: title, authors, venue/source, "
            f"year, and a 2-sentence summary of the key finding. Focus on papers relevant to "
            f"system-level AI reliability, not just component-level model improvements."
        )
        result = self.research_query(query)
        return result.get("response") if result else None

    def regulatory_search(self) -> Optional[dict[str, Any]]:
        """
        Search for recent developments in AI regulation relevant to our research.
        Covers: EU AI Act implementation, NIST AI RMF updates, ISO/IEC 42001.
        """
        query = (
            "What are the latest official guidance documents, technical standards, and "
            "regulatory developments (published in the last 3 months) related to: "
            "EU AI Act implementation guidance for high-risk AI systems, NIST AI Risk "
            "Management Framework updates, ISO/IEC 42001 AI management systems, "
            "system-level AI assurance requirements. Include document titles, issuing bodies, "
            "dates, and key requirements for AI system-level reliability."
        )
        return self.research_query(query)

    def find_author_recent_work(self, author_name: str) -> Optional[str]:
        """Find recent publications from a tracked author."""
        query = (
            f"Find the most recent (2024-2025) publications by {author_name} "
            f"related to AI systems reliability, trustworthy AI, system-level assurance, "
            f"or software engineering for AI. List titles, venues, and years."
        )
        result = self.research_query(query)
        return result.get("response") if result else None
