"""
Unit tests for PhD LitReview Automator agents and tools.
Uses pytest with mocked API calls.
"""
import pytest
import json
from unittest.mock import MagicMock, patch
from datetime import datetime


class TestPipelineState:
    """Test state definitions."""

    def test_paper_structure(self):
        from state.pipeline_state import Paper
        paper: Paper = {
            "paper_id": "test:123",
            "title": "Test Paper",
            "authors": ["Smith, J."],
            "year": 2024,
            "abstract": "Test abstract",
            "relevance_score": 7.5,
            "inclusion_decision": True,
            "gaps_addressed": ["GAP-1", "GAP-3"],
        }
        assert paper["paper_id"] == "test:123"
        assert paper["relevance_score"] == 7.5


class TestPaperDatabase:
    """Test paper database operations."""

    def test_save_and_retrieve(self, tmp_path):
        from tools.paper_db import PaperDatabase

        db = PaperDatabase(db_path=str(tmp_path / "test_db.json"))

        paper = {
            "paper_id": "doi:10.1234/test",
            "title": "Test Paper on AI Reliability",
            "year": 2024,
            "relevance_score": 8.0,
            "gaps_addressed": ["GAP-1"],
            "inclusion_decision": True,
        }

        db.save_paper(paper)
        assert db.paper_exists("doi:10.1234/test")

        known_ids = db.get_known_ids()
        assert "doi:10.1234/test" in known_ids

        db.close()

    def test_corpus_operations(self, tmp_path):
        from tools.paper_db import PaperDatabase

        db = PaperDatabase(db_path=str(tmp_path / "test_db.json"))

        paper = {
            "paper_id": "doi:10.1234/corpus-test",
            "title": "Corpus Test Paper",
            "year": 2024,
            "relevance_score": 7.5,
            "gaps_addressed": ["GAP-2"],
        }

        db.save_paper(paper)
        db.add_to_corpus(paper)

        corpus = db.get_corpus()
        assert len(corpus) == 1
        assert corpus[0]["paper_id"] == "doi:10.1234/corpus-test"

        # Test gap filtering
        gap2_papers = db.get_papers_by_gap("GAP-2")
        assert len(gap2_papers) == 1

        db.close()

    def test_stats(self, tmp_path):
        from tools.paper_db import PaperDatabase

        db = PaperDatabase(db_path=str(tmp_path / "test_db.json"))
        stats = db.get_stats()

        assert "total_discovered" in stats
        assert "total_corpus" in stats
        assert "gaps_coverage" in stats

        db.close()


class TestSemanticScholarNormalization:
    """Test Semantic Scholar result normalization."""

    def test_normalize_paper(self):
        from tools.semantic_scholar import SemanticScholarTool
        ss = SemanticScholarTool()

        raw = {
            "paperId": "abc123",
            "title": "AI System Reliability",
            "abstract": "Test abstract",
            "year": 2024,
            "authors": [{"name": "Smith, John"}, {"name": "Jones, Alice"}],
            "venue": "ICSE",
            "citationCount": 42,
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2401.12345"},
            "openAccessPdf": {"url": "https://arxiv.org/pdf/2401.12345"},
            "s2FieldsOfStudy": [{"category": "Computer Science"}],
            "tldr": {"text": "Brief summary"},
        }

        normalized = ss._normalize_paper(raw, "test query", "test_source")

        assert normalized["paper_id"] == "abc123"
        assert normalized["title"] == "AI System Reliability"
        assert normalized["doi"] == "10.1234/test"
        assert normalized["arxiv_id"] == "2401.12345"
        assert normalized["citation_count"] == 42
        assert normalized["discovery_source"] == "test_source"


class TestSettings:
    """Test configuration loading."""

    def test_settings_load(self):
        from config.settings import settings
        assert settings.anthropic_model == "claude-sonnet-4-6"
        assert settings.max_papers_per_run > 0
        assert settings.min_relevance_score > 0

    def test_research_context(self):
        from config.settings import settings
        assert "Integration Paradox" in settings.research_topic
        assert len(settings.research_gaps) == 6
        assert len(settings.research_questions) == 3


class TestVisualizationAgent:
    """Test visualization data generation."""

    def test_gap_matrix(self):
        from agents.visualization_agent import _build_gap_matrix

        corpus = [
            {"paper_id": "1", "title": "Paper 1", "year": 2024,
             "authors": ["Smith"], "citation_count": 10,
             "gaps_addressed": ["GAP-1", "GAP-2"]},
            {"paper_id": "2", "title": "Paper 2", "year": 2023,
             "authors": ["Jones"], "citation_count": 5,
             "gaps_addressed": ["GAP-1"]},
        ]

        viz = _build_gap_matrix(corpus, "2024-01-01")
        assert viz["visualization_type"] == "gap_matrix"
        gap_counts = viz["data"]["gap_counts"]
        assert gap_counts["GAP-1"] == 2
        assert gap_counts["GAP-2"] == 1

    def test_citation_network(self):
        from agents.visualization_agent import _build_citation_network

        corpus = [
            {"paper_id": "test:1", "title": "Paper 1", "year": 2024,
             "citation_count": 20, "relevance_score": 8.0,
             "gaps_addressed": ["GAP-1"], "citation_priority": "high",
             "synthesis_tags": ["compositional_failure"]},
        ]

        viz = _build_citation_network(corpus, [])
        assert viz["visualization_type"] == "citation_network"
        assert len(viz["data"]["nodes"]) == 1


class TestDeduplicationNode:
    """Test paper deduplication logic."""

    def test_deduplication_removes_known_papers(self):
        from pipeline.graph import deduplicate_node

        state = {
            "discovered_papers": [
                {"paper_id": "doi:10.1234/known", "doi": "10.1234/known", "title": "Known Paper"},
                {"paper_id": "doi:10.1234/new", "doi": "10.1234/new", "title": "New Paper"},
            ],
            "known_paper_ids": ["doi:10.1234/known"],
        }

        result = deduplicate_node(state)
        assert result["duplicate_count"] == 1
        assert len(result["new_papers"]) == 1
        assert result["new_papers"][0]["paper_id"] == "doi:10.1234/new"

    def test_doi_deduplication(self):
        from pipeline.graph import deduplicate_node

        state = {
            "discovered_papers": [
                {"paper_id": "ss:abc123", "doi": "10.1234/same", "title": "Paper A"},
                {"paper_id": "arxiv:2401.001", "doi": "10.1234/same", "title": "Paper A (arXiv)"},
            ],
            "known_paper_ids": [],
        }

        result = deduplicate_node(state)
        # Same DOI → only one should survive
        assert len(result["new_papers"]) == 1


class TestEmailRendering:
    """Test email template rendering."""

    def test_render_daily_report(self):
        from tools.email_tool import EmailTool

        email = EmailTool()
        context = {
            "run_date": "2024-01-15",
            "topic_short": "Integration Paradox",
            "n_discovered": 12,
            "n_included": 3,
            "n_excluded": 9,
            "included_papers": [
                {
                    "title": "Test Paper on System Reliability",
                    "authors": ["Smith, J.", "Jones, A."],
                    "venue": "ICSE",
                    "year": 2024,
                    "relevance_score": 8.5,
                    "critical_notes": "Strong empirical evidence of integration paradox.",
                }
            ],
            "gap_status": {
                "GAP-1": {"papers_addressing": 5, "trajectory": "growing", "today_update": "New evidence"},
                "GAP-2": {"papers_addressing": 1, "trajectory": "nascent", "today_update": ""},
            },
            "lit_review_updates": [],
            "proposal_updates": [],
            "doc_links": {
                "lit_review": "https://docs.google.com/...",
                "proposal": "https://docs.google.com/...",
            },
            "human_action_items": [],
            "contribution_reminder": "Our study provides the missing unified theory.",
            "total_corpus": 42,
            "errors": [],
        }

        html, text = email.render_daily_report(context)
        assert "Integration Paradox" in html
        assert "3" in html  # n_included
        assert "Test Paper on System Reliability" in html
        assert len(text) > 100
