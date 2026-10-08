"""
Regression tests for defects found in the production-readiness review.

Each test names the failure it guards against. The end-to-end tests run the real
LangGraph pipeline with the LLM, search APIs and email replaced by fakes.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from tests.conftest import FakeLLM, make_paper

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── LLM access ───────────────────────────────────────────────────────────────

class TestLLMUtils:
    def test_temperature_only_sent_to_models_that_accept_it(self):
        from agents.llm_utils import supports_temperature
        # 400 "temperature is deprecated for this model" on these
        for model in ("claude-opus-5-5", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5-5",
                      "claude-haiku-5-5", "claude-fable-5-1", "claude-opus-6"):
            assert not supports_temperature(model), model
        for model in ("claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"):
            assert supports_temperature(model), model

    def test_request_payload_per_model(self, isolated):
        from langchain_core.messages import HumanMessage
        from agents.llm_utils import _make_chat_model

        opus47 = _make_chat_model("claude-opus-4-7", 0.2, 50000, "high")._get_request_payload([HumanMessage(content="x")])
        assert "temperature" not in json.dumps(opus47)
        assert opus47["max_tokens"] <= 20000  # SDK refuses larger non-streaming requests

        sonnet46 = _make_chat_model("claude-sonnet-4-6", 0.2, 1000, "high")._get_request_payload([HumanMessage(content="x")])
        assert "temperature" in json.dumps(sonnet46)

        haiku45 = _make_chat_model("claude-haiku-4-5", 0.2, 1000, "high")._get_request_payload([HumanMessage(content="x")])
        assert "effort" not in json.dumps(haiku45)  # Haiku 4.5 rejects effort

    def test_message_text_handles_thinking_blocks(self):
        from langchain_core.messages import AIMessage
        from agents.llm_utils import message_text
        msg = AIMessage(content=[{"type": "thinking", "thinking": "hmm", "signature": "s"}, {"type": "text", "text": " [1] "}])
        assert message_text(msg) == "[1]"
        assert message_text(AIMessage(content=" plain ")) == "plain"

    @pytest.mark.parametrize("reply,expected", [
        ('[1, 2]', [1, 2]),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here it is:\n{"a": {"b": [1]}}\nHope that helps.', {"a": {"b": [1]}}),
        ('```\n["x"]\n```', ["x"]),
    ])
    def test_parse_json_variants(self, reply, expected):
        from agents.llm_utils import parse_json
        assert parse_json(reply) == expected

    def test_parse_json_raises_decode_error(self):
        from agents.llm_utils import parse_json
        with pytest.raises(json.JSONDecodeError):
            parse_json("no json here")

    def test_resilient_llm_falls_back_on_error_and_refusal(self, isolated):
        from langchain_core.messages import AIMessage
        from agents.llm_utils import ResilientLLM

        class Stub:
            def __init__(self, model, reply=None, exc=None):
                self.model, self.reply, self.exc, self.calls, self.max_tokens = model, reply, exc, 0, 100
            def invoke(self, messages, **kw):
                self.calls += 1
                if self.exc:
                    raise self.exc
                return self.reply

        ok = AIMessage(content="ok", response_metadata={"stop_reason": "end_turn"})
        refused = AIMessage(content="", response_metadata={"stop_reason": "refusal"})

        fallback = Stub("fb", reply=ok)
        assert ResilientLLM(Stub("p", reply=refused), fallback).invoke([]) is ok
        assert fallback.calls == 1

        primary = Stub("bad-model", exc=RuntimeError("400 temperature is deprecated"))
        fallback = Stub("fb", reply=ok)
        llm = ResilientLLM(primary, fallback)
        for _ in range(5):
            assert llm.invoke([]) is ok
        assert primary.calls == 3  # skipped after 3 consecutive failures
        assert fallback.calls == 5

        with pytest.raises(RuntimeError):
            ResilientLLM(Stub("p2", exc=RuntimeError("boom")), None).invoke([])


# ── Configuration / CLI ──────────────────────────────────────────────────────

class TestSettingsAndCLI:
    def test_empty_env_vars_use_defaults(self, monkeypatch):
        """Unset GitHub secrets arrive as empty strings; they must not override defaults."""
        from config.settings import Settings
        monkeypatch.setenv("SMTP_PORT", "")
        monkeypatch.setenv("LITREVIEW_GOOGLE_DOC_ID", "")
        monkeypatch.setenv("DRY_RUN", "true")
        s = Settings(_env_file=None)
        assert s.smtp_port == 587
        assert s.litreview_google_doc_id == Settings.model_fields["litreview_google_doc_id"].default
        assert s.dry_run is True

    def test_dry_run_flag_reaches_settings(self, isolated, monkeypatch):
        """--dry-run used to set an env var after settings was loaded, so emails were still sent."""
        import main
        import pipeline.graph
        from config.settings import settings

        seen = {}

        def fake_run_pipeline(**kwargs):
            seen["dry_run"] = settings.dry_run
            return {"errors": []}

        monkeypatch.setattr(pipeline.graph, "run_pipeline", fake_run_pipeline)
        args = Namespace(dry_run=True, human_checkpoint=False, log_level="INFO", days_lookback=None)
        assert main.cmd_run(args) == 0
        assert seen["dry_run"] is True

    def test_missing_api_key_fails_fast(self, isolated, monkeypatch):
        import main
        from config.settings import settings
        monkeypatch.setattr(settings, "anthropic_api_key", "")
        args = Namespace(dry_run=False, human_checkpoint=False, log_level="INFO", days_lookback=None)
        assert main.cmd_run(args) == 2

    def test_discovery_prompt_has_no_format_placeholders(self):
        """The system prompt contains JSON braces; .format() on it raised KeyError('\\n  \"search_queries\"')."""
        from config.prompts import AgentPrompts
        rendered = AgentPrompts.DISCOVERY.replace("{days_lookback}", "7")
        assert "last 7 days" in rendered
        assert '"search_queries"' in rendered


# ── Persistence ──────────────────────────────────────────────────────────────

class TestPaperDatabase:
    def test_writes_reach_disk_without_close(self, isolated):
        from tools.paper_db import PaperDatabase
        from config.settings import settings
        PaperDatabase().save_paper({"paper_id": "a", "title": "A"})
        on_disk = json.loads(Path(settings.papers_db_path).read_text())
        assert on_disk["papers"]["1"]["paper_id"] == "a"

    def test_instances_share_one_handle(self, isolated):
        """Two TinyDB handles on one file used to overwrite each other's tables."""
        from tools.paper_db import PaperDatabase
        from config.settings import settings
        a, b = PaperDatabase(), PaperDatabase()
        assert a is b
        a.save_paper({"paper_id": "a"})
        b.log_run({"run_date": "2026-10-08"})
        on_disk = json.loads(Path(settings.papers_db_path).read_text())
        assert on_disk["papers"] and on_disk["runs"]

    def test_dry_run_uses_a_copy(self, isolated, monkeypatch):
        from tools.paper_db import PaperDatabase
        from config.settings import settings
        PaperDatabase().save_paper({"paper_id": "real"})
        PaperDatabase().close()
        before = Path(settings.papers_db_path).read_text()

        monkeypatch.setattr(settings, "dry_run", True)
        db = PaperDatabase()
        assert "real" in db.get_known_ids()  # sees existing data
        db.save_paper({"paper_id": "dry"})
        assert Path(settings.papers_db_path).read_text() == before

    def test_cross_source_dedup_keys(self):
        from tools.paper_db import dedup_keys
        s2 = {"paper_id": "abc", "doi": "10.48550/ARXIV.2605.14968", "title": "Compound AI System Reliability: A Failure Taxonomy"}
        arxiv = {"paper_id": "arxiv:2605.14968", "arxiv_id": "2605.14968v2", "title": "Compound AI system reliability — a failure taxonomy"}
        crossref = {"paper_id": "doi:x", "doi": "https://doi.org/10.48550/arxiv.2605.14968"}
        assert dedup_keys(s2) & dedup_keys(arxiv)
        assert dedup_keys(s2) & dedup_keys(crossref)

    def test_deduplicate_node_uses_known_keys(self):
        from pipeline.graph import deduplicate_node
        state = {
            "discovered_papers": [
                {"paper_id": "openalex:W1", "doi": "10.1234/KNOWN", "title": "Known paper under another source id"},
                {"paper_id": "ss:2", "doi": "10.1234/new", "title": "A genuinely new paper about integration"},
            ],
            "known_paper_ids": [],
            "known_paper_keys": ["doi:10.1234/known"],
        }
        result = deduplicate_node(state)
        assert [p["paper_id"] for p in result["new_papers"]] == ["ss:2"]


# ── Output formatting ────────────────────────────────────────────────────────

class TestOutputs:
    def test_bibtex_keys_unique_and_escaped(self):
        from tools.bibtex import build_entries, existing_keys
        papers = [
            {"paper_id": "1", "title": "Reliability & Composition_of AI", "authors": ["Smith, J."], "year": 2026, "venue": "ICSE"},
            {"paper_id": "2", "title": "Reliability in practice", "authors": ["John Smith"], "year": 2026, "arxiv_id": "2605.1"},
        ]
        taken = {"Smith2026Reliability"}
        entries = build_entries(papers, taken)
        keys = [k for k, _ in entries.values()]
        assert len(set(keys)) == 2 and "Smith2026Reliability" not in keys
        bib = "\n\n".join(b for _, b in entries.values())
        assert r"\&" in bib and r"\_" in bib
        assert existing_keys(bib) == set(keys)
        assert entries["2"][1].startswith("@misc{")

    def test_email_escapes_html_and_tolerates_nulls(self, isolated):
        from tools.email_tool import EmailTool
        html, text = EmailTool().render_daily_report({
            "run_date": "2026-10-08", "topic_short": "t", "n_discovered": 1, "n_included": 1, "n_excluded": 0,
            "included_papers": [{"title": "<script>alert(1)</script>", "authors": [], "venue": "", "year": "",
                                 "relevance_score": 7.0, "critical_notes": "", "actual_contribution": ""}],
            "gap_status": {}, "lit_review_updates": [], "proposal_updates": [], "doc_links": {},
            "human_action_items": [], "contribution_reminder": "x",
            "errors": [{"node": "extraction", "error": "All 3 extractions failed"}], "warnings": ["source down"],
        })
        assert "<script>" not in html and "&lt;script&gt;" in html
        assert "All 3 extractions failed" in html and "source down" in text

    def test_visualizations_tolerate_missing_authors(self, isolated):
        from agents.visualization_agent import _build_citation_network, _build_gap_matrix
        corpus = [{"paper_id": "1", "title": None, "authors": [], "gaps_addressed": ["GAP-1"], "citation_count": None}]
        assert "Unknown (?)" in _build_gap_matrix(corpus, "2026-10-08")["data"]["table_markdown"]
        assert _build_citation_network(corpus, [])["data"]["nodes"][0]["size"] == 10

    def test_markdown_updates_go_before_references(self):
        from agents.document_update_agent import _apply_updates_to_markdown
        doc = "# Review\n\n## 2. Field\ntext\n\n## 6. References\n\n*refs*\n"
        out = _apply_updates_to_markdown(doc, [{"section_path": "2", "markdown_content": "NEW TEXT"}], "2026-10-08")
        assert out.index("NEW TEXT") < out.index("## 6. References")

    def test_arxiv_query_uses_specific_terms(self):
        from tools.arxiv_tool import _query_terms, _term_clause
        terms = _query_terms("Compositional failure in multi-agent AI systems")
        assert terms == ["compositional", "failure", "multi-agent"]
        assert _term_clause(terms) == 'all:compositional AND all:failure AND all:"multi-agent"'


# ── End-to-end pipeline runs ─────────────────────────────────────────────────

def _install_fakes(monkeypatch, llm, papers, crossref_fails=True):
    """Replace the LLM, the search APIs, PDF download and email sending."""
    import agents.discovery_agent as discovery
    import agents.extraction_agent as extraction
    import agents.critical_analysis_agent as analysis
    import agents.synthesis_agent as synthesis
    import agents.document_update_agent as doc_update
    import agents.email_reporter_agent as reporter
    from tools.semantic_scholar import SemanticScholarTool
    from tools.arxiv_tool import ArxivTool
    from tools.openalex_tool import OpenAlexTool
    from tools.crossref_tool import CrossRefTool
    from tools.email_tool import EmailTool

    for module in (discovery, extraction, analysis, synthesis, doc_update, reporter):
        monkeypatch.setattr(module, "_build_llm", lambda: llm)

    monkeypatch.setattr(SemanticScholarTool, "search_recent", lambda self, query, days_back=7, limit=20: [dict(p) for p in papers])
    # arXiv returns the same first paper under its own id: must be de-duplicated by DOI/title
    arxiv_copy = {**papers[0], "paper_id": "arxiv:2610.00001", "doi": papers[0]["doi"].upper(), "discovery_source": "arxiv"}
    monkeypatch.setattr(ArxivTool, "search", lambda self, query, max_results=20, categories=None, days_back=7: [dict(arxiv_copy)])
    monkeypatch.setattr(OpenAlexTool, "search", lambda self, query, days_back=7, limit=25, concept_ids=None: [])

    def crossref_search(self, query, limit=20, days_back=7, issn=None):
        if crossref_fails:
            raise RuntimeError("crossref down")
        return []
    monkeypatch.setattr(CrossRefTool, "search", crossref_search)
    monkeypatch.setattr(extraction, "_fetch_pdf_text", lambda url, max_chars=15000: None)

    from config.settings import settings
    for name, value in {"email_to": "owner@example.org", "smtp_user": "bot@example.org", "smtp_password": "x"}.items():
        monkeypatch.setattr(settings, name, value)  # "configured"; sending itself is faked below
    sent = []
    monkeypatch.setattr(EmailTool, "send", lambda self, subject, html_body, text_body, to=None, cc=None: sent.append(subject) or True)
    return sent


PAPERS = [
    make_paper(1, "The Integration Paradox in ML Pipelines"),
    make_paper(2, "Integration failures in multi-agent systems", abstract=None),  # S2 often returns null abstracts
    make_paper(3, "Better image classification accuracy"),
]

LIT_REVIEW_SEED = "# Literature Review\n\n## 2. The State of the Field\n\nSeed text.\n\n## 6. References\n\n*refs*\n"


def _db(isolated_path):
    return json.loads((isolated_path / "storage" / "papers_db.json").read_text())


class TestEndToEnd:
    def test_full_run(self, isolated, monkeypatch):
        from pipeline.graph import run_pipeline
        (isolated / "outputs" / "literature_review.md").write_text(LIT_REVIEW_SEED)
        llm = FakeLLM()
        sent = _install_fakes(monkeypatch, llm, PAPERS)

        state = run_pipeline()

        assert [p["paper_id"] for p in state["new_papers"]] == ["ss:1", "ss:2", "ss:3"]  # arXiv copy removed
        assert sorted(p["paper_id"] for p in state["included_papers"]) == ["ss:1", "ss:2"]
        assert [p["paper_id"] for p in state["excluded_papers"]] == ["ss:3"]
        assert state.get("analysis_errors", []) == []
        assert any("crossref" in w for w in state.get("warnings", []))
        assert state.get("errors", []) == []

        included = state["included_papers"][0]
        assert included["title"] == "The Integration Paradox in ML Pipelines"  # not overwritten by the LLM
        assert included["actual_contribution"] == "Shows system-level degradation"

        db = _db(isolated)
        assert len(db["papers"]) == 3 and len(db["corpus"]) == 2 and len(db["runs"]) == 1
        assert all(p["documented"] for p in db["corpus"].values())
        assert db["runs"]["1"]["email_sent"] is True
        assert len(sent) == 1 and "2 new papers" in sent[0]

        review = (isolated / "outputs" / "literature_review.md").read_text()
        bib = (isolated / "outputs" / "references.bib").read_text()
        from tools.bibtex import existing_keys
        keys = existing_keys(bib)
        assert len(keys) == 2
        assert review.index("Recent Updates") < review.index("## 6. References")
        assert all(f"[@{k}]" in review for k in keys)
        assert (isolated / "outputs" / "visualizations" / "citation_network.json").exists()
        assert (isolated / "outputs" / "literature_review_updates.tex").exists()

    def test_second_run_skips_known_papers_without_stale_errors(self, isolated, monkeypatch):
        from pipeline.graph import run_pipeline
        _install_fakes(monkeypatch, FakeLLM(), PAPERS)
        run_pipeline()
        second = run_pipeline()
        assert second["new_papers"] == []
        assert second.get("errors", []) == []
        assert len(second.get("warnings", [])) == 1  # only this run's crossref warning
        assert len(_db(isolated)["runs"]) == 2

    def test_dry_run_changes_nothing(self, isolated, monkeypatch):
        from config.settings import settings
        from pipeline.graph import run_pipeline
        monkeypatch.setattr(settings, "dry_run", True)
        sent = _install_fakes(monkeypatch, FakeLLM(), PAPERS)

        state = run_pipeline()

        assert len(state["included_papers"]) == 2
        assert sent == []
        assert not (isolated / "storage" / "papers_db.json").exists()
        assert not (isolated / "outputs" / "references.bib").exists()
        assert [p.name for p in (isolated / "outputs" / "daily_reports").iterdir()] == ["2026-10-08_dry_run_daily_report.html"] \
            or all(p.name.endswith("_dry_run_daily_report.html") for p in (isolated / "outputs" / "daily_reports").iterdir())
        assert not (isolated / "outputs" / "visualizations").exists()

    def test_failed_analysis_is_reported_and_retried(self, isolated, monkeypatch):
        from pipeline.graph import run_pipeline
        _install_fakes(monkeypatch, FakeLLM(fail_on=("PAPER TO ANALYZE",)), PAPERS)

        first = run_pipeline()
        assert len(first["analysis_errors"]) == 3
        assert first["excluded_papers"] == []
        assert first["errors"][0]["node"] == "critical_analysis"
        assert _db(isolated).get("papers", {}) == {}  # nothing persisted, so nothing is skipped later

        _install_fakes(monkeypatch, FakeLLM(), PAPERS)
        second = run_pipeline()
        assert len(second["included_papers"]) == 2

    def test_failed_extraction_is_counted(self, isolated, monkeypatch):
        from pipeline.graph import run_pipeline
        _install_fakes(monkeypatch, FakeLLM(fail_on=("Extract structured information",)), PAPERS)
        state = run_pipeline()
        assert len(state["extraction_errors"]) == 3
        assert any(e["node"] == "extraction" for e in state["errors"])
        assert len(state["included_papers"]) == 2  # analysis still ran from the abstracts

    def test_unsent_email_fails_the_run(self, isolated, monkeypatch):
        """A configured email that fails to send used to leave the weekly job green."""
        from pipeline.graph import run_pipeline
        from tools.email_tool import EmailTool
        _install_fakes(monkeypatch, FakeLLM(), PAPERS)
        monkeypatch.setattr(EmailTool, "send", lambda self, *a, **kw: False)
        state = run_pipeline()
        assert any(e["node"] == "email_reporter" for e in state["errors"])

    def test_failed_document_update_is_retried_next_run(self, isolated, monkeypatch):
        from pipeline.graph import run_pipeline
        _install_fakes(monkeypatch, FakeLLM(fail_on=("Update the PhD Literature Review",)), PAPERS)
        first = run_pipeline()
        assert first["lit_review_updates"] == []
        assert not any(p["documented"] for p in _db(isolated)["corpus"].values())

        # Next week: no new papers at all, but the pending ones still get documented
        llm = FakeLLM()
        _install_fakes(monkeypatch, llm, PAPERS)
        second = run_pipeline()
        assert second["new_papers"] == []
        assert len(second["lit_review_updates"]) == 1
        assert all(p["documented"] for p in _db(isolated)["corpus"].values())


# ── Deployment ───────────────────────────────────────────────────────────────

class TestWorkflow:
    WORKFLOW = REPO_ROOT / ".github" / "workflows" / "weekly_litreview.yml"

    def test_weekly_schedule_and_permissions(self):
        wf = yaml.safe_load(self.WORKFLOW.read_text())
        triggers = wf[True]  # YAML parses the bare key `on` as True
        assert triggers["schedule"][0]["cron"].split()[4] == "1"  # Mondays
        assert "workflow_dispatch" in triggers
        assert wf["permissions"]["contents"] == "write"
        steps = wf["jobs"]["litreview"]["steps"]
        commit = next(s for s in steps if s.get("name", "").startswith("Commit"))
        assert "git add storage/papers_db.json" in commit["run"]
        assert "git add outputs\n" in commit["run"]  # includes the LaTeX updates file
        assert "users.noreply.github.com" in commit["run"]  # commits authored by the repo owner

    def test_actionlint(self):
        actionlint = shutil.which("actionlint") or str(Path(sys.executable).parent / "actionlint")
        if not Path(actionlint).exists():
            pytest.skip("actionlint not installed (pip install actionlint-py)")
        result = subprocess.run([actionlint, str(self.WORKFLOW)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


class TestCheckpointAndRetries:
    def test_human_checkpoint_pauses_and_resumes(self, isolated, monkeypatch):
        """The checkpoint used to stop the run for good: no resume command, no email."""
        from pipeline.graph import resume_pipeline, run_pipeline
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        sent = _install_fakes(monkeypatch, FakeLLM(), PAPERS)

        paused = run_pipeline(interrupt_before_docs=True)
        assert paused["paused_at"] == ["document_update"]
        assert sent == []

        resumed = resume_pipeline(paused["thread_id"])
        assert "paused_at" not in resumed
        assert len(resumed["lit_review_updates"]) == 1
        assert len(sent) == 1

    def test_checkpoint_ignored_in_github_actions(self, isolated, monkeypatch):
        from pipeline.graph import run_pipeline
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        sent = _install_fakes(monkeypatch, FakeLLM(), PAPERS)
        state = run_pipeline(interrupt_before_docs=True)
        assert "paused_at" not in state and len(sent) == 1

    def test_failed_synthesis_postpones_document_updates(self, isolated, monkeypatch):
        from pipeline.graph import run_pipeline
        _install_fakes(monkeypatch, FakeLLM(fail_on=("Synthesize these",)), PAPERS)
        first = run_pipeline()
        assert first["synthesis_failed"] is True
        assert any(e["node"] == "synthesis" for e in first["errors"])
        assert first.get("lit_review_updates", []) == []
        assert not any(p["documented"] for p in _db(isolated)["corpus"].values())

        _install_fakes(monkeypatch, FakeLLM(), PAPERS)
        run_pipeline()
        assert all(p["documented"] for p in _db(isolated)["corpus"].values())

    def test_seed_papers_join_the_corpus(self, isolated):
        """Seed papers used to go only into the 'papers' table, never the corpus."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import initialize
        seed = isolated / "seed.json"
        seed.write_text(json.dumps([
            {"paper_id": "seed:1", "title": "Known integration study", "inclusion_decision": True, "gaps_addressed": ["GAP-2"]},
            {"paper_id": "seed:2", "title": "Off-topic", "inclusion_decision": False},
            {"title": "missing id"},
        ]))
        initialize.load_seed_papers(str(seed))
        db = _db(isolated)
        assert [p["paper_id"] for p in db["corpus"].values()] == ["seed:1"]
        assert db["corpus"]["1"]["documented"] is True  # already in the owner's review
        assert len(db["papers"]) == 2 and len(db["excluded"]) == 1


class TestLocalScheduler:
    @pytest.mark.parametrize("expr,moment,expected", [
        ("17 6 * * 1", "2026-10-12 06:17", True),    # Monday
        ("17 6 * * 1", "2026-10-13 06:17", False),   # Tuesday
        ("17 6 * * 1", "2026-10-12 06:18", False),
        ("0 6 * * 0", "2026-10-11 06:00", True),     # Sunday as 0
        ("0 6 * * 7", "2026-10-11 06:00", True),     # Sunday as 7
        ("*/15 * * * *", "2026-10-12 10:45", True),
        ("0 9 * * 1-5", "2026-10-17 09:00", False),  # Saturday
        ("0 9 1 * 1", "2026-10-01 09:00", True),     # 1st of the month (a Thursday): day OR weekday
        ("0 9 1 * 1", "2026-10-12 09:00", True),     # a Monday
        ("0 9 1 * 1", "2026-10-13 09:00", False),
    ])
    def test_cron_matches(self, expr, moment, expected):
        from datetime import datetime
        import main
        assert main.cron_matches(expr, datetime.strptime(moment, "%Y-%m-%d %H:%M")) is expected

    @pytest.mark.parametrize("expr", ["17 x * * 1", "17 6 * *", "*/0 * * * *", "0 25 * * *", "0 6 32 * *"])
    def test_invalid_cron_rejected_up_front(self, expr):
        import main
        with pytest.raises(ValueError):
            main.parse_cron(expr)


class TestPreScreen:
    def test_screened_out_papers_are_remembered(self, isolated, monkeypatch):
        """Overlapping weekly windows would otherwise re-score rejected papers every run."""
        from pipeline.graph import run_pipeline
        papers = PAPERS + [make_paper(4, "Sourdough recipe recommendation app")]
        llm = FakeLLM(screen_out_marker="Sourdough")
        _install_fakes(monkeypatch, llm, papers)

        first = run_pipeline()
        assert "ss:4" not in [p["paper_id"] for p in first["new_papers"]]
        excluded = list(_db(isolated)["excluded"].values())
        assert any(p["paper_id"] == "ss:4" and p["exclusion_reason"].startswith("Pre-screen") for p in excluded)

        llm.calls.clear()
        run_pipeline()
        assert not any("Sourdough" in c for c in llm.calls)  # skipped before any LLM call


class TestSemanticScholar:
    def test_null_fields_and_date_filter_fallback(self, isolated, monkeypatch):
        """S2 returns explicit nulls (abstract is often null); a rejected date filter falls back to years."""
        import httpx
        import tools.semantic_scholar as s2

        calls = []

        class FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get(self, url, params=None, headers=None):
                calls.append(dict(params))
                request = httpx.Request("GET", url)
                if "publicationDateOrYear" in params:
                    return httpx.Response(400, request=request, json={"error": "bad param"})
                return httpx.Response(200, request=request, json={"data": [{
                    "paperId": "p1", "title": "T", "abstract": None, "venue": None, "authors": [{"name": None}, {"name": "A"}],
                    "citationCount": None, "publicationDate": None, "year": 2026, "externalIds": None, "openAccessPdf": None, "tldr": None,
                }]})

        monkeypatch.setattr(s2.httpx, "Client", FakeClient)
        monkeypatch.setattr(s2.time, "sleep", lambda s: None)
        papers = s2.SemanticScholarTool().search_recent("x", days_back=21, limit=5)
        assert "publicationDateOrYear" in calls[0] and "year" in calls[1]
        assert papers[0]["abstract"] == "" and papers[0]["authors"] == ["A"] and papers[0]["citation_count"] == 0


class TestFollowUps:
    """Defects found by the review of the fixes themselves."""

    def _auth_error(self):
        import anthropic
        import httpx
        response = httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        return anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)

    def test_invalid_key_does_not_fall_back(self, isolated):
        from agents.llm_utils import ResilientLLM

        class Stub:
            def __init__(self, exc=None):
                self.model, self.exc, self.calls, self.max_tokens = "m", exc, 0, 1
            def invoke(self, messages, **kw):
                self.calls += 1
                if self.exc:
                    raise self.exc
        fallback = Stub()
        with pytest.raises(Exception):
            ResilientLLM(Stub(self._auth_error()), fallback).invoke([])
        assert fallback.calls == 0

    def test_shared_outage_does_not_disable_primary(self, isolated):
        import agents.llm_utils as llm_utils
        from agents.llm_utils import ResilientLLM

        class Down:
            model, max_tokens = "primary", 1
            def invoke(self, messages, **kw):
                raise RuntimeError("network down")
        for _ in range(5):
            with pytest.raises(RuntimeError):
                ResilientLLM(Down(), Down()).invoke([])
        assert llm_utils._primary_failures.get("primary", 0) == 0

    @pytest.mark.parametrize("model,requested,expected", [
        ("claude-opus-5-5", "xhigh", "xhigh"),
        ("claude-sonnet-4-6", "xhigh", "high"),   # 4.6 has no xhigh
        ("claude-opus-4-5", "max", "high"),
        ("claude-haiku-4-5", "high", None),       # effort not supported
        ("claude-opus-5-5", "bogus", None),
    ])
    def test_effort_clamped_per_model(self, model, requested, expected):
        from agents.llm_utils import effort_for
        assert effort_for(model, requested) == expected

    def test_parse_json_prefers_the_payload_over_bracketed_prose(self):
        from agents.llm_utils import parse_json
        assert parse_json('As in [1], the result:\n{"updates": [1, 2, 3]}') == {"updates": [1, 2, 3]}

    def test_truncated_reply_is_an_error(self):
        from langchain_core.messages import AIMessage
        from agents.llm_utils import TruncatedReplyError, invoke_json

        class Cut:
            def invoke(self, messages):
                return AIMessage(content='{"a": [1, {"b": 2}', response_metadata={"stop_reason": "max_tokens"})
        with pytest.raises(TruncatedReplyError):
            invoke_json(Cut(), [])

    def test_references_insertion_uses_the_documents_own_heading(self):
        from agents.document_update_agent import _apply_updates_to_markdown
        doc = ("# Review\n\n## Recent Updates — w1\n\nWeek1 text\n\n### References\n- x\n\n---\n\n"
               "## 6. References\n\n*refs*\n")
        out = _apply_updates_to_markdown(doc, [{"section_path": "2", "markdown_content": "WEEK2"}], "w2")
        assert out.index("### References") < out.index("WEEK2") < out.index("## 6. References")

    def test_empty_document_update_keeps_papers_pending(self, isolated, monkeypatch):
        from pipeline.graph import run_pipeline

        class NoUpdates(FakeLLM):
            def invoke(self, messages, **kw):
                if "Update the PhD Literature Review" in messages[-1].content:
                    from tests.conftest import _reply
                    return _reply('{"document_updates": []}')
                return super().invoke(messages, **kw)
        _install_fakes(monkeypatch, NoUpdates(), PAPERS)
        state = run_pipeline()
        assert state["lit_review_updates"] == []
        assert not any(p["documented"] for p in _db(isolated)["corpus"].values())

    def test_failed_proposal_update_is_retried_alone(self, isolated, monkeypatch):
        from pipeline.graph import run_pipeline
        _install_fakes(monkeypatch, FakeLLM(fail_on=("Update the PhD Research Proposal",)), PAPERS)
        first = run_pipeline()
        corpus = _db(isolated)["corpus"].values()
        assert all(p["documented"] for p in corpus) and not any(p["proposal_documented"] for p in corpus)
        assert any("proposal update failed" in w for w in first["warnings"])

        llm = FakeLLM()
        _install_fakes(monkeypatch, llm, PAPERS)
        second = run_pipeline()
        assert second["lit_review_updates"] == []           # not documented twice
        assert len(second["proposal_updates"]) == 1
        assert all(p["proposal_documented"] for p in _db(isolated)["corpus"].values())
        assert not any("Update the PhD Literature Review" in c for c in llm.calls)

    def test_strip_markup_keeps_comparisons(self):
        from tools.paper_db import strip_markup
        assert strip_markup("<jats:p>Latency &lt;10 ms for inputs &gt;1 MB</jats:p>") == "Latency <10 ms for inputs >1 MB"
        assert strip_markup("&lt;p&gt;Wound <i>Localization</i>&lt;/p&gt;") == "Wound Localization"

    def test_same_title_different_authors_not_merged(self):
        from tools.paper_db import dedup_keys
        a = {"title": "Large Language Models for Software Engineering: A Survey", "authors": ["Hou, X."]}
        b = {"title": "Large Language Models for Software Engineering: A Survey", "authors": ["Fan, A."]}
        assert not (dedup_keys(a) & dedup_keys(b))

    def test_stats_and_export_tolerate_seed_rows(self, isolated):
        from tools.paper_db import PaperDatabase
        from tools.bibtex import existing_keys
        db = PaperDatabase()
        db.add_to_corpus({"paper_id": "seed:1", "title": "Reliability study", "relevance_score": None,
                          "bibtex": "@article{Smith2026Reliability,\n  title = {{X}}\n}"})
        db.add_to_corpus({"paper_id": "p2", "title": "Reliability in practice", "authors": ["Smith, J."], "year": 2026})
        assert db.get_stats()["avg_relevance"] == 0.0
        bib = db.export_corpus_bibtex()
        assert bib.count("Smith2026Reliability,") == 1 and len(existing_keys(bib)) == 2

    def test_synthesis_node_crash_postpones_documents(self, isolated, monkeypatch):
        import pipeline.graph as graph
        monkeypatch.setattr(graph, "run_synthesis", lambda state: (_ for _ in ()).throw(RuntimeError("boom")))
        result = graph.synthesis_node({"run_date": "2026-10-08"})
        assert result["synthesis_failed"] is True
        assert graph.route_after_synthesis({**result}) == "email_reporter"


class TestBatchingAndDelivery:
    def test_partial_batch_failure_keeps_only_that_batch_pending(self, isolated, monkeypatch):
        """Document updates run 6 papers per call; one failed call must not lose or block the others."""
        from pipeline.graph import run_pipeline
        papers = [make_paper(n, f"Integration study number {n} of compositional failure") for n in range(1, 9)]

        class FailSecondBatch(FakeLLM):
            def invoke(self, messages, **kw):
                prompt = messages[-1].content
                # paper 8 is only in the second literature-review batch (papers 7-8)
                if "Update the PhD Literature Review" in prompt and "Integration study number 8 " in prompt:
                    raise RuntimeError("simulated overload")
                return super().invoke(messages, **kw)

        _install_fakes(monkeypatch, FailSecondBatch(), papers)
        state = run_pipeline()
        corpus = {p["paper_id"]: p for p in _db(isolated)["corpus"].values()}
        assert len(corpus) == 8
        assert sorted(pid for pid, p in corpus.items() if not p["documented"]) == ["ss:7", "ss:8"]
        assert any("retried next run" in w for w in state["warnings"])

    def test_single_paper_auth_failure_is_reported(self, isolated, monkeypatch):
        """With a bad key, even a quiet week (1 paper) must fail the run, not stay green."""
        from pipeline.graph import run_pipeline
        llm = FakeLLM(fail_on=("Extract structured information", "PAPER TO ANALYZE"))
        def raise_auth(messages, **kw):
            raise RuntimeError("Error code: 401 - {'type': 'authentication_error'}")
        _install_fakes(monkeypatch, llm, PAPERS[:1])
        original = llm.invoke
        monkeypatch.setattr(llm, "invoke", lambda messages, **kw: raise_auth(messages) if any(
            m in messages[-1].content for m in llm.fail_on) else original(messages, **kw))
        state = run_pipeline()
        assert {e["node"] for e in state["errors"]} >= {"extraction", "critical_analysis"}

    @pytest.mark.parametrize("port,ssl_class", [(465, True), (587, False)])
    def test_smtp_transport(self, isolated, monkeypatch, port, ssl_class):
        import tools.email_tool as email_tool
        from config.settings import settings
        events = []

        class FakeSMTP:
            def __init__(self, host, port, **kw):
                events.append(("ssl" if self.SSL else "plain", host, port))
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def ehlo(self): events.append("ehlo")
            def starttls(self, context=None): events.append("starttls")
            def login(self, user, password): events.append(("login", user))
            def sendmail(self, sender, recipients, message): events.append(("sendmail", tuple(recipients)))
        plain = type("Plain", (FakeSMTP,), {"SSL": False})
        secure = type("Secure", (FakeSMTP,), {"SSL": True})
        monkeypatch.setattr(email_tool.smtplib, "SMTP", plain)
        monkeypatch.setattr(email_tool.smtplib, "SMTP_SSL", secure)
        for name, value in {"smtp_port": port, "smtp_user": "bot@x.org", "smtp_password": "pw",
                            "email_to": "a@x.org, b@x.org", "email_cc": "c@x.org"}.items():
            monkeypatch.setattr(settings, name, value)

        assert email_tool.EmailTool().send("s", "<p>h</p>", "t") is True
        assert events[0][0] == ("ssl" if ssl_class else "plain")
        assert ("starttls" in events) is (not ssl_class)
        assert ("sendmail", ("a@x.org", "b@x.org", "c@x.org")) in events

    def test_sendgrid_payload(self, isolated, monkeypatch):
        import httpx
        import tools.email_tool as email_tool
        from config.settings import settings
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured.update(url=url, json=json, headers=headers)
            return httpx.Response(202, request=httpx.Request("POST", url))
        monkeypatch.setattr(email_tool.httpx, "post", fake_post)
        for name, value in {"email_backend": "sendgrid", "sendgrid_api_key": "SG.key", "email_to": "a@x.org,b@x.org",
                            "email_from": "PhD Bot <bot@x.org>"}.items():
            monkeypatch.setattr(settings, name, value)

        assert email_tool.EmailTool().send("subject", "<p>h</p>", "text") is True
        payload = captured["json"]
        assert payload["personalizations"][0]["to"] == [{"email": "a@x.org"}, {"email": "b@x.org"}]
        assert payload["from"] == {"email": "bot@x.org", "name": "PhD Bot"}
        assert captured["headers"]["Authorization"] == "Bearer SG.key"
