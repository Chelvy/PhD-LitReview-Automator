#!/usr/bin/env python3
"""
PhD LitReview Automator — Initialization Script

Run this ONCE before the first daily pipeline run to:
1. Verify all API credentials
2. Initialize the local database
3. Download and process any provided seed papers
4. Create initial document templates (if not importing from Google Docs)
5. Send a test email

Usage:
  python scripts/initialize.py
  python scripts/initialize.py --seed-papers seed_papers.json
  python scripts/initialize.py --skip-email-test
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def check_credentials() -> dict[str, bool]:
    """Verify all configured API credentials."""
    from config.settings import settings

    checks = {
        "Anthropic API (required)": bool(settings.anthropic_api_key),
        "Semantic Scholar API": bool(settings.semantic_scholar_api_key),
        "Elicit API": bool(settings.elicit_api_key),
        "Scite API": bool(settings.scite_api_key),
        "Scholarcy API": bool(settings.scholarcy_api_key),
        "Perplexity API": bool(settings.perplexity_api_key),
        "LangSmith": bool(settings.langsmith_api_key),
        "Email (SMTP)": settings.has_email,
        "Google Docs": settings.has_google_docs,
    }

    table = Table(title="API Credential Status", border_style="cyan")
    table.add_column("Service")
    table.add_column("Status", justify="center")
    table.add_column("Notes")

    notes = {
        "Anthropic API (required)": "REQUIRED — pipeline will not run without this",
        "Semantic Scholar API": "Free without key (limited); get key for 1 req/sec",
        "Elicit API": "Optional — enhances structured extraction",
        "Scite API": "Optional — enables citation quality analysis",
        "Scholarcy API": "Optional — enables PDF structured extraction",
        "Perplexity API": "Optional — enables deep research queries",
        "LangSmith": "Optional — enables pipeline observability",
        "Email (SMTP)": "Recommended — daily report delivery",
        "Google Docs": "Optional — sync to Google Docs",
    }

    all_required = True
    for service, configured in checks.items():
        status = "✅ OK" if configured else "❌ Missing"
        if "required" in service.lower() and not configured:
            all_required = False
            status = "❌ REQUIRED"
        table.add_row(service, status, notes.get(service, ""))

    console.print(table)
    return checks


def test_anthropic_api() -> bool:
    """Test Anthropic API connectivity."""
    from config.settings import settings
    if not settings.anthropic_api_key:
        console.print("[red]Anthropic API key not configured![/red]")
        return False

    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            max_tokens=50,
        )
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content="Reply only with: READY")])
        console.print(f"[green]✓ Anthropic API working: {response.content.strip()}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Anthropic API failed: {e}[/red]")
        return False


def test_semantic_scholar() -> bool:
    """Test Semantic Scholar API with a known query."""
    try:
        from tools.semantic_scholar import SemanticScholarTool
        ss = SemanticScholarTool()
        results = ss.search_papers("AI system reliability integration failure", limit=3)
        console.print(f"[green]✓ Semantic Scholar: found {len(results)} test papers[/green]")
        return True
    except Exception as e:
        console.print(f"[yellow]⚠ Semantic Scholar: {e} (optional, pipeline continues)[/yellow]")
        return False


def initialize_database() -> None:
    """Initialize the papers database."""
    from tools.paper_db import PaperDatabase
    Path("storage").mkdir(exist_ok=True)
    db = PaperDatabase()
    stats = db.get_stats()
    console.print(
        f"[green]✓ Database initialized: {stats['total_corpus']} corpus papers, "
        f"{stats['total_discovered']} total discovered[/green]"
    )


def create_document_templates() -> None:
    """Create initial document templates if they don't exist."""
    lit_review_path = Path("outputs/literature_review.md")
    proposal_path = Path("outputs/research_proposal.md")

    Path("outputs").mkdir(exist_ok=True)

    if not lit_review_path.exists():
        lit_review_template = f"""# Literature Review: The Integration Paradox

**Research Topic:** The Integration Paradox: Why Reliable AI/ML Components
Compose into Unreliable Systems? A Cross-Domain Empirical and Theoretical Analysis

**Last Updated:** {datetime.now().strftime("%Y-%m-%d")}
**Maintained by:** PhD LitReview Automator (LangGraph + Claude claude-sonnet-4-6)

---

## 1. Introduction and Scope

[To be developed — auto-populated by daily pipeline runs]

## 2. The State of the Field

### 2.1 Component-Level AI Reliability: The Dominant Paradigm

[To be developed]

### 2.2 System-Level AI Reliability: Emerging Concerns

[To be developed]

### 2.3 Compositional Approaches

[To be developed]

### 2.4 Human-AI Handoff and Socio-Technical Coupling

[To be developed]

### 2.5 Regulatory Landscape: EU AI Act and System-Level Requirements

[To be developed]

## 3. Key Debates

[To be developed]

## 4. Research Gaps

### GAP-1: Component-Centric Dominance

[To be developed — papers: 0]

### GAP-2: No Trust-Composition Calculus

[To be developed — papers: 0]

### GAP-3: Underdeveloped Predictors and Indicators

[To be developed — papers: 0]

### GAP-4: Non-Monotone and Multi-Dimensional Reliability

[To be developed — papers: 0]

### GAP-5: Assurance Evidence Composition

[To be developed — papers: 0]

### GAP-6: Socio-Technical Integration Weakly Formalized

[To be developed — papers: 0]

## 5. Justification and Contribution of the Present Study

[To be developed]

## 6. References

[Auto-populated from outputs/references.bib]

---
*Auto-maintained by PhD LitReview Automator. Manual edits welcome.*
"""
        lit_review_path.write_text(lit_review_template, encoding="utf-8")
        console.print("[green]✓ Literature review template created[/green]")
    else:
        console.print("[dim]Literature review exists — skipping template creation[/dim]")

    if not proposal_path.exists():
        proposal_template = f"""# Research Proposal: The Integration Paradox

**Title:** The Integration Paradox: Why Reliable AI/ML Components Compose into
Unreliable Systems? A Cross-Domain Empirical and Theoretical Analysis

**Last Updated:** {datetime.now().strftime("%Y-%m-%d")}

---

## 1. Introduction and Motivation

[To be developed]

## 2. Research Questions

**RQ1 (Compositional Formalism):** How should system-level trust be represented
and composed across the AI SDLC, and under what conditions does error/risk amplify superlinearly?

**RQ2 (Mechanisms and Measurable Indicators):** Which integration-failure mechanisms
recur across domains and SDLC stages, and what measurable indicators predict them?

**RQ3 (Interventions and Auditable Lifecycle Evidence):** Which end-to-end SDLC
practices measurably reduce integration-paradox risk, and how can evidence be packaged
into credible, update-resilient assurance?

## 3. Methodology

[To be developed]

## 4. Expected Contributions

[To be developed]

## 5. Timeline

[To be developed]

## 6. References

[Auto-populated]

---
*Auto-maintained by PhD LitReview Automator.*
"""
        proposal_path.write_text(proposal_template, encoding="utf-8")
        console.print("[green]✓ Research proposal template created[/green]")
    else:
        console.print("[dim]Research proposal exists — skipping template creation[/dim]")


def load_seed_papers(seed_file: str) -> None:
    """Load seed papers from a JSON file into the database."""
    try:
        with open(seed_file) as f:
            seed_papers = json.load(f)

        from tools.paper_db import PaperDatabase
        db = PaperDatabase()
        added = db.save_papers_batch(seed_papers)
        console.print(f"[green]✓ Loaded {len(seed_papers)} seed papers ({added} new)[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to load seed papers: {e}[/red]")


def test_email_configuration() -> None:
    """Send a test email to verify configuration."""
    from tools.email_tool import EmailTool
    from config.settings import settings

    email = EmailTool()
    if not email.enabled:
        console.print("[yellow]⚠ Email not configured — skipping test[/yellow]")
        return

    sent = email.send(
        subject=f"PhD LitReview Automator — Initialization Test ({datetime.now().strftime('%Y-%m-%d')})",
        html_body=(
            "<h1>PhD LitReview Automator Initialized!</h1>"
            "<p>The daily pipeline is configured and ready to run.</p>"
            "<p>You will receive daily reports at this address.</p>"
        ),
        text_body=(
            "PhD LitReview Automator Initialized!\n\n"
            "The daily pipeline is configured and ready to run.\n"
            "You will receive daily reports at this address."
        ),
    )
    if sent:
        console.print(f"[green]✓ Test email sent to {settings.email_to}[/green]")
    else:
        console.print("[red]✗ Failed to send test email[/red]")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Initialize PhD LitReview Automator")
    parser.add_argument("--seed-papers", help="JSON file with seed papers to load")
    parser.add_argument("--skip-email-test", action="store_true")
    parser.add_argument("--skip-api-test", action="store_true")
    args = parser.parse_args()

    console.print(Panel(
        "[bold blue]PhD LitReview Automator — Initialization[/bold blue]\n"
        "Setting up the daily literature review pipeline...",
        border_style="blue"
    ))

    # 1. Check credentials
    console.print("\n[bold]Step 1: Checking API credentials...[/bold]")
    checks = check_credentials()

    # 2. Test critical APIs
    if not args.skip_api_test:
        console.print("\n[bold]Step 2: Testing API connectivity...[/bold]")
        if not test_anthropic_api():
            console.print("[red bold]Anthropic API test failed! Pipeline cannot run without it.[/red bold]")
            sys.exit(1)
        test_semantic_scholar()

    # 3. Initialize database
    console.print("\n[bold]Step 3: Initializing database...[/bold]")
    initialize_database()

    # 4. Create document templates
    console.print("\n[bold]Step 4: Creating document templates...[/bold]")
    create_document_templates()

    # 5. Load seed papers
    if args.seed_papers:
        console.print("\n[bold]Step 5: Loading seed papers...[/bold]")
        load_seed_papers(args.seed_papers)

    # 6. Test email
    if not args.skip_email_test:
        console.print("\n[bold]Step 6: Testing email configuration...[/bold]")
        test_email_configuration()

    # Done
    console.print(Panel(
        "[bold green]✓ Initialization complete![/bold green]\n\n"
        "Next steps:\n"
        "1. Run: [bold]python main.py run --dry-run[/bold] (test without writes)\n"
        "2. Run: [bold]python main.py run[/bold] (full run)\n"
        "3. Deploy: Push to GitHub — Actions will run daily at 6 AM UTC\n\n"
        "Check [bold]python main.py status[/bold] anytime for pipeline statistics.",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
