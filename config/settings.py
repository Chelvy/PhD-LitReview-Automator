"""
Central configuration for PhD LitReview Automator.
All values loaded from environment variables (.env file).
"""
from __future__ import annotations

import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Treat empty env vars (e.g. unset GitHub secrets) as missing so defaults apply.
        env_ignore_empty=True,
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(default="claude-opus-5-5")
    # Used when the primary model errors (bad model ID, no access, outage) or refuses.
    anthropic_fallback_model: str = Field(default="claude-sonnet-4-6")
    # Optional global override of each agent's reasoning effort (low|medium|high|xhigh|max).
    anthropic_effort: str = Field(default="")
    openai_api_key: str = Field(default="", description="OpenAI API key (fallback)")

    # ── LangSmith ─────────────────────────────────────────────────────────
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="phd-litreview-automator")
    langsmith_tracing: bool = Field(default=False)

    # ── Academic Search APIs ──────────────────────────────────────────────
    semantic_scholar_api_key: str = Field(default="")
    elicit_api_key: str = Field(default="")
    scite_api_key: str = Field(default="")
    scholarcy_api_key: str = Field(default="")
    perplexity_api_key: str = Field(default="")
    core_api_key: str = Field(default="")
    openalex_email: str = Field(default="")
    openalex_api_key: str = Field(default="")
    ncbi_email: str = Field(default="")
    ncbi_api_key: str = Field(default="")

    # ── Google Docs ────────────────────────────────────────────────────────
    google_service_account_file: str = Field(default="config/google_service_account.json")
    google_oauth_credentials_file: str = Field(default="config/google_credentials.json")
    google_token_file: str = Field(default="config/google_token.json")
    litreview_google_doc_id: str = Field(default="1WukjXxp115ZNI4-u-1agm-5du1KG23iA")
    proposal_google_doc_id: str = Field(default="1lplSQtusQxbcey0Wluj-vtXeTkq90ewX")
    ai_failures_google_doc_id: str = Field(default="1cP3t5cC1C0_lldC5ER480Z7FCGZwIVcV")

    # ── Email ──────────────────────────────────────────────────────────────
    email_backend: str = Field(default="smtp")
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    email_from: str = Field(default="")  # defaults to smtp_user when empty
    email_to: str = Field(default="")
    email_cc: str = Field(default="")
    sendgrid_api_key: str = Field(default="")

    # ── Pipeline Tuning ────────────────────────────────────────────────────
    max_papers_per_run: int = Field(default=50)
    min_relevance_score: float = Field(default=6.0)
    relevance_citation_boost: float = Field(default=0.5)
    # Weekly runs look back 3 weeks: sources index papers days or weeks late, and
    # de-duplication makes the overlap cheap (known papers are skipped before any LLM call).
    days_lookback: int = Field(default=21)
    human_review_checkpoint: bool = Field(default=False)
    llm_concurrency: int = Field(default=4)  # papers extracted/analyzed in parallel
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=2.0)

    # ── Storage ────────────────────────────────────────────────────────────
    sqlite_db_path: str = Field(default="storage/pipeline_state.db")
    papers_db_path: str = Field(default="storage/papers_db.json")
    outputs_dir: str = Field(default="outputs")

    # ── Scheduling ─────────────────────────────────────────────────────────
    cron_schedule: str = Field(default="17 6 * * 1")  # Mondays 06:17 (local `schedule` command)
    timezone: str = Field(default="UTC")

    # ── Debug ──────────────────────────────────────────────────────────────
    debug: bool = Field(default=False)
    dry_run: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # ── Research Context (static, embedded) ───────────────────────────────
    research_topic: str = (
        "The Integration Paradox: Why Reliable AI/ML Components Compose into "
        "Unreliable Systems? A Cross-Domain Empirical and Theoretical Analysis"
    )

    research_fields: list[str] = [
        "Artificial Intelligence",
        "Machine Learning",
        "Deep Learning",
        "Generative AI",
        "Computer Vision",
        "Natural Language Processing",
        "Intelligent Documents Processing",
        "Fraud Detection",
        "Supervised Learning",
        "Unsupervised Learning",
        "Reinforcement Learning",
        "Foundation Models",
        "Agentic AI",
    ]

    research_gaps: list[str] = [
        "Component-centric dominance: most trustworthy AI techniques optimize one model "
        "or property; guidance for multi-component, socio-technical compositions is fragmentary.",
        "No trust-composition calculus: we lack computable laws/conditions that predict when "
        "integrating acceptable components yields unacceptable system behavior under dependence, "
        "feedback, and tight coupling.",
        "Predictors and indicators are underdeveloped: compositional collapse evidence exists, "
        "but few architecture-aware, SDLC-integrated indicators forecast integration risk at interfaces.",
        "Reliability is multi-dimensional and non-monotone: improving one dimension does not "
        "reliably constrain others, undermining single-metric system certification.",
        "Assurance evidence does not compose operationally: integrating tests, simulations, "
        "formal artifacts, and monitors into coherent, auditable system-level arguments remains a bottleneck.",
        "Socio-technical integration remains weakly formalized: human oversight and workflow coupling "
        "are treated qualitatively without clear interfaces to quantitative trust.",
    ]

    research_questions: list[str] = [
        "RQ1 (Compositional formalism): How should system-level trust be represented and composed "
        "across the AI SDLC, and under what conditions does error/risk amplify superlinearly?",
        "RQ2 (Mechanisms and measurable indicators): Which integration-failure mechanisms recur "
        "across domains and SDLC stages, and what measurable indicators predict them?",
        "RQ3 (Interventions and auditable lifecycle evidence): Which end-to-end SDLC practices "
        "measurably reduce integration-paradox risk, and how can evidence be packaged into credible, "
        "update-resilient assurance?",
    ]

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v_upper

    def configure_langsmith(self) -> None:
        """Set LangSmith environment variables if tracing is enabled."""
        if self.langsmith_tracing and self.langsmith_api_key:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project

    @property
    def has_google_docs(self) -> bool:
        return os.path.exists(self.google_service_account_file) or os.path.exists(
            self.google_token_file
        )

    @property
    def has_email(self) -> bool:
        if not self.email_to:
            return False
        if self.email_backend == "sendgrid":
            return bool(self.sendgrid_api_key and (self.email_from or self.smtp_user))
        return bool(self.smtp_user and self.smtp_password)


settings = Settings()
