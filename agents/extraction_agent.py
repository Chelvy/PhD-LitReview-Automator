"""
Agent 2: PDF Extraction & Structured Summarization Agent

Downloads and reads PDFs (or uses abstract + metadata where PDF unavailable).
Produces structured extraction: Objectives, Methodology, Results, Conclusions,
Assumptions, Strengths, Blank Spots — ready for critical analysis.

Tools: Scholarcy (PDF extraction), SciSpace (structured summaries),
       PyPDF / PyMuPDF (direct PDF text extraction), Claude (synthesis).
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from typing import Any, Optional

import httpx
import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from config.prompts import AgentPrompts
from state.pipeline_state import PipelineState
from tools.scholarcy_tool import ScholarcyTool

logger = structlog.get_logger(__name__)

MAX_ABSTRACT_TOKENS = 3000  # Characters for abstract-only extraction


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0.1,
        max_tokens=8096,
    )


def _fetch_pdf_text(pdf_url: str, max_chars: int = 15000) -> Optional[str]:
    """Attempt to download and extract text from a PDF URL."""
    if not pdf_url:
        return None
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(pdf_url, headers={"User-Agent": "Mozilla/5.0 (research bot)"})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not pdf_url.endswith(".pdf"):
                return None

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=resp.content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
                if len(text) >= max_chars:
                    break
            doc.close()
            return text[:max_chars]
        except ImportError:
            pass

        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(resp.content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
                if len(text) >= max_chars:
                    break
            return text[:max_chars]
        except Exception:
            pass

    except Exception as e:
        logger.warning("pdf_fetch_failed", url=pdf_url, error=str(e))
    return None


def _extract_with_llm(
    llm: ChatAnthropic,
    paper: dict[str, Any],
    full_text: Optional[str] = None,
) -> dict[str, Any]:
    """
    Use Claude to extract structured information from a paper.
    Uses full text if available, otherwise abstract + metadata.
    """
    system = SystemMessage(content=AgentPrompts.EXTRACTION)

    # Build context
    authors = ", ".join(paper.get("authors", [])[:5])
    if len(paper.get("authors", [])) > 5:
        authors += " et al."

    context_parts = [
        f"TITLE: {paper.get('title', 'Unknown')}",
        f"AUTHORS: {authors}",
        f"YEAR: {paper.get('year', 'N/A')}",
        f"VENUE: {paper.get('venue', 'N/A')}",
        f"CITATION COUNT: {paper.get('citation_count', 0)}",
        f"ABSTRACT:\n{paper.get('abstract', 'Not available')[:2000]}",
    ]

    # Add Elicit pre-extraction if available (bonus data)
    if paper.get("elicit_methodology"):
        context_parts.append(f"\nELICIT METHODOLOGY HINT: {paper['elicit_methodology']}")
    if paper.get("elicit_key_findings"):
        context_parts.append(f"ELICIT FINDINGS HINT: {paper['elicit_key_findings']}")
    if paper.get("scholarcy_highlights"):
        highlights = "\n".join(paper["scholarcy_highlights"][:5])
        context_parts.append(f"\nSCHOLARCY HIGHLIGHTS:\n{highlights}")

    if full_text:
        # Include key sections of the paper text
        context_parts.append(f"\nFULL TEXT EXCERPT (first ~{len(full_text)//1000}K chars):\n{full_text[:12000]}")
        prompt_suffix = "You have access to the full paper text above. Extract all fields with high precision."
    else:
        prompt_suffix = (
            "You only have the abstract and metadata. Extract as much as possible, "
            "and mark fields as 'INFERRED FROM ABSTRACT' where you are extrapolating."
        )

    context = "\n".join(context_parts)

    prompt = f"""Extract structured information from this paper for our PhD literature review.

{context}

{prompt_suffix}

Return a JSON object with ALL of these fields (use null if truly unavailable):
{{
  "paper_id": "{paper.get('paper_id', '')}",
  "extraction_complete": true,
  "research_objectives": "What were the authors trying to find out? How does this relate to system-level AI reliability?",
  "methodology": "Specific methods used (e.g., 'Empirical study of 47 production ML pipelines using fault injection', NOT vague descriptions)",
  "datasets_used": ["dataset name 1", ...],
  "results": "Key quantitative/qualitative findings — include specific numbers where available",
  "conclusions": "Authors' main conclusions and what they mean for the Integration Paradox research",
  "key_claims": ["Specific falsifiable claim 1", ...],
  "theoretical_contribution": "What new theory, formalism, or framework does this introduce?",
  "empirical_contribution": "What empirical data or evidence does this provide?",
  "formalisms_introduced": ["Name of formalism or model introduced", ...],
  "assumptions": ["Assumption 1 the study relies on", ...],
  "strengths": ["Methodological or conceptual strength 1", ...],
  "blind_spots": ["Limitation the AUTHORS acknowledge", ...],
  "blank_spots": ["What the study COULD have investigated but DID NOT — your analytical insight", ...],
  "key_contributions": ["Contribution 1 to our research specifically", ...],
  "related_anchor_papers": ["Influential paper this builds on", ...],
  "quotes": ["Direct quote highly relevant to RQ1/2/3", ...]
}}"""

    try:
        response = llm.invoke([system, HumanMessage(content=prompt)])
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        extracted = json.loads(content)
        logger.info("extraction_complete", paper_id=paper.get("paper_id"), method="llm")
        return extracted

    except json.JSONDecodeError as e:
        logger.warning("extraction_json_parse_failed", paper_id=paper.get("paper_id"), error=str(e))
        return {
            "paper_id": paper.get("paper_id", ""),
            "extraction_complete": False,
            "research_objectives": paper.get("abstract", "")[:500],
            "methodology": "Extraction failed — manual review required",
            "results": "",
            "conclusions": "",
            "blank_spots": [],
            "blind_spots": [],
            "key_contributions": [],
        }
    except Exception as e:
        logger.error("extraction_failed", paper_id=paper.get("paper_id"), error=str(e))
        return {"paper_id": paper.get("paper_id", ""), "extraction_complete": False}


def run_extraction(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: PDF Extraction & Structured Summarization.
    Processes each new paper through Scholarcy + Claude extraction pipeline.
    """
    new_papers = state.get("new_papers", [])
    if not new_papers:
        logger.info("extraction_skip", reason="no_new_papers")
        return {"extracted_papers": [], "extraction_errors": []}

    logger.info("extraction_start", papers_to_process=len(new_papers))

    llm = _build_llm()
    scholarcy = ScholarcyTool()

    extracted_papers = []
    extraction_errors = []

    for paper in new_papers:
        paper_id = paper.get("paper_id", "unknown")
        logger.info("extracting_paper", paper_id=paper_id, title=paper.get("title", "")[:60])

        try:
            # Step 1: Try Scholarcy for PDF extraction (fast, structured)
            enriched_paper = {**paper}
            pdf_url = paper.get("pdf_url", "")

            if scholarcy.enabled and pdf_url:
                scholary_data = scholarcy.extract_from_url(pdf_url)
                if scholary_data:
                    enriched_paper.update(scholary_data)
                    logger.info("scholarcy_enriched", paper_id=paper_id)

            # Step 2: Try direct PDF text extraction
            full_text = None
            if pdf_url and not enriched_paper.get("extraction_complete"):
                full_text = _fetch_pdf_text(pdf_url)
                if full_text:
                    logger.info("pdf_text_fetched", paper_id=paper_id, chars=len(full_text))

            # Step 3: LLM extraction (uses Scholarcy/PDF data as context)
            extraction = _extract_with_llm(llm, enriched_paper, full_text)

            # Merge extraction into paper
            final_paper = {**enriched_paper, **extraction}
            extracted_papers.append(final_paper)

        except Exception as e:
            logger.error("paper_extraction_error", paper_id=paper_id, error=str(e))
            extraction_errors.append({
                "paper_id": paper_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            # Still include paper with whatever we have
            extracted_papers.append({**paper, "extraction_complete": False})

    logger.info(
        "extraction_complete",
        extracted=len(extracted_papers),
        errors=len(extraction_errors),
    )

    return {
        "extracted_papers": extracted_papers,
        "extraction_errors": extraction_errors,
    }
