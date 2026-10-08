"""
Shared test fixtures.

`isolated` points every path the pipeline writes to at a temporary directory and
disables external side effects, so tests never touch the real database, documents,
email account or Google Docs — even when a developer's .env is present.
"""
from __future__ import annotations

import json
import re
import tempfile
from typing import Any

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from config.settings import Settings, settings
    import agents.llm_utils as llm_utils
    import tools.paper_db as paper_db

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # dry-run DB copies land here
    (tmp_path / "outputs").mkdir()
    # Start from the declared defaults, so a developer's .env cannot change test outcomes
    for name, field in Settings.model_fields.items():
        monkeypatch.setattr(settings, name, field.get_default(call_default_factory=True))
    for name, value in {
        "papers_db_path": str(tmp_path / "storage" / "papers_db.json"),
        "sqlite_db_path": str(tmp_path / "storage" / "pipeline_state.db"),
        "outputs_dir": str(tmp_path / "outputs"),
        "dry_run": False,
        "anthropic_api_key": "test-key",
        "anthropic_model": "claude-opus-5-5",
        "anthropic_fallback_model": "claude-sonnet-4-6",
        "anthropic_effort": "",
        "email_to": "",
        "smtp_user": "",
        "smtp_password": "",
        "email_backend": "smtp",
        "google_service_account_file": str(tmp_path / "missing_sa.json"),
        "google_token_file": str(tmp_path / "missing_token.json"),
        "human_review_checkpoint": False,
        "semantic_scholar_api_key": "",
        "elicit_api_key": "",
        "scite_api_key": "",
        "scholarcy_api_key": "",
        "langsmith_tracing": False,
    }.items():
        monkeypatch.setattr(settings, name, value)

    monkeypatch.setattr(llm_utils, "_primary_failures", {})
    monkeypatch.setattr(paper_db, "_dry_run_copy", None)
    FakeLLM.instances.clear()
    yield tmp_path

    for db in list(paper_db._open_databases.values()):
        db.close()
    unexpected = [p for llm in FakeLLM.instances for p in llm.unexpected]
    FakeLLM.instances.clear()
    assert not unexpected, f"FakeLLM received prompts it does not recognise: {[u[:120] for u in unexpected]}"


def _reply(text: str) -> AIMessage:
    """A Claude 5-style reply: a thinking block followed by the text block."""
    return AIMessage(
        content=[{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}],
        response_metadata={"stop_reason": "end_turn"},
    )


class FakeLLM:
    """
    Stands in for ResilientLLM. Recognises each agent's prompt and returns a
    plausible reply; `fail_on` makes calls whose prompt contains a marker raise.
    Unrecognised prompts are recorded (the `isolated` fixture fails the test) rather
    than raised, because the agents catch exceptions and would hide them.
    """

    instances: list["FakeLLM"] = []

    def __init__(
        self,
        fail_on: tuple[str, ...] = (),
        include_marker: str = "Integration",
        screen_out_marker: str | None = None,
    ) -> None:
        self.fail_on = fail_on
        self.include_marker = include_marker
        self.screen_out_marker = screen_out_marker
        self.calls: list[str] = []
        self.unexpected: list[str] = []
        FakeLLM.instances.append(self)

    def invoke(self, messages, **kwargs: Any) -> AIMessage:
        prompt = messages[-1].content
        self.calls.append(prompt)
        for marker in self.fail_on:
            if marker in prompt:
                raise RuntimeError(f"simulated API failure ({marker})")

        if "highly targeted search queries" in prompt:
            return _reply('```json\n{"search_queries": ["compositional failure AI pipelines", "multi-agent reliability"]}\n```')
        if "Score the relevance" in prompt:
            titles = re.findall(r"^PAPER \d+:\nTitle: (.*)$", prompt, re.MULTILINE)
            scores = []
            for i, title in enumerate(titles):
                off_topic = bool(self.screen_out_marker and self.screen_out_marker in title)
                scores.append({"paper_number": str(i + 1), "score": "1.5" if off_topic else 7.0,
                               "reason": "off-topic" if off_topic else "relevant", "include": not off_topic})
            return _reply(json.dumps({"scores": scores}))  # dict-wrapped, string numbers
        if "Extract structured information" in prompt:
            return _reply("Here is the extraction:\n" + json.dumps({
                "paper_id": "LLM-MANGLED-ID",
                "title": "LLM rewrote the title",
                "extraction_complete": True,
                "research_objectives": "Study integration failures",
                "methodology": None,
                "results": None,
                "key_claims": ["claim"],
                "blind_spots": None,
            }))
        if "PAPER TO ANALYZE" in prompt:
            include = self.include_marker in prompt.split("Title:", 1)[1].split("\n", 1)[0]
            return _reply(json.dumps({
                "paper_id": "LLM-MANGLED-ID",
                "rq1_relevance": 8.0, "rq2_relevance": 7.0, "rq3_relevance": 6.0,
                "relevance_score": 8.0 if include else 2.0,
                "inclusion_decision": include,
                "exclusion_reason": None if include else "Component-level only",
                "gaps_addressed": ["GAP-1", "GAP-3"] if include else [],
                "wagner_analysis": {
                    "blank_spots": ["no interface contracts"],
                    "blind_spots": [],
                    "actual_contribution": "Shows system-level degradation",
                    "assumes_component_isolation": False,
                    "integration_paradox_evidence": "accuracy 94% -> 71%",
                },
                "synthesis_tags": ["compositional_failure"],
                "citation_priority": "high",
                "critical_notes": "Important <b>evidence</b>",
            }))
        if "Synthesize these" in prompt:
            return _reply(json.dumps({
                "papers_synthesized": [],
                "historical_narrative_update": {"new_developments": "New compositional evidence"},
                "key_debates_update": [],
                "gap_status_update": {"GAP-1": {"papers_addressing": 1, "trajectory": "growing", "today_update": "New evidence"}},
                "contradictions": ["Paper A and B disagree on monitoring"],
                "contribution_update": "Unified theory still missing",
                "lit_review_sections_to_update": [{"section_path": "2.2", "action": "append_paragraph"}],
                "proposal_sections_to_update": [{"section_path": "3.2", "action": "strengthen"}],
            }))
        if "Update the PhD Literature Review" in prompt:
            keys = re.findall(r'"citation_key": "([^"]+)"', prompt)
            return _reply(json.dumps({"literature_review_updates": [{
                "section_path": "2.2 System-Level AI Reliability",
                "action": "append_paragraph",
                "markdown_content": "Recent work " + " ".join(f"[@{k}]" for k in keys) + " shows compositional failure.",
                "latex_content": "Recent work \\citep{" + ",".join(keys) + "}.",
                "justification": "new evidence",
            }]}))
        if "Update the PhD Research Proposal" in prompt:
            return _reply(json.dumps({"document_updates": [{
                "section_path": "3.2 Cross-Domain Empirical Analysis",
                "action": "strengthen",
                "markdown_content": "Validation now also draws on recent incident studies.",
                "justification": "methodology",
            }]}))
        if "Research Contribution Reminder" in prompt:
            return _reply("Keep going — GAP-2 is still open.")
        self.unexpected.append(prompt)
        return _reply("{}")


def make_paper(n: int, title: str, **extra: Any) -> dict[str, Any]:
    paper = {
        "paper_id": f"ss:{n}",
        "title": title,
        "authors": [f"Author{n}, A.", "Second, B."],
        "year": 2026,
        "abstract": f"Abstract of paper {n} about AI systems.",
        "doi": f"10.1234/paper.{n}",
        "arxiv_id": "",
        "venue": "ICSE",
        "citation_count": 3,
        "url": f"https://example.org/{n}",
        "pdf_url": "",
        "publication_date": "2026-10-05",
        "discovery_source": "semantic_scholar_search",
    }
    paper.update(extra)
    return paper
