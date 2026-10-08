# PhD LitReview Automator

A production-grade, scheduled multi-agent pipeline that automates and maintains
a PhD-level literature review process for research on **"The Integration Paradox:
Why Reliable AI/ML Components Compose into Unreliable Systems?"**

**Framework:** LangGraph (selected after rigorous 2026-era benchmarking — see [BENCHMARKING.md](BENCHMARKING.md))
**LLM:** Anthropic Claude — `claude-opus-5-5` by default, configurable, with automatic fallback to a second model
**Runs:** Weekly (Mondays 06:17 UTC) via GitHub Actions, or on demand

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PhD LitReview Automator                          │
│                   (LangGraph StateGraph)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  START                                                               │
│    │                                                                  │
│    ▼                                                                  │
│  [generate_run_metadata]  Load run_date, known IDs from SQLite DB    │
│    │                                                                  │
│    ▼                                                                  │
│  [discovery]  ─────────── Semantic Scholar + ArXiv + CrossRef        │
│    │          ─────────── + OpenAlex + Elicit + Perplexity           │
│    ▼                                                                  │
│  [deduplicate]  Remove known papers, DOI-level dedup                 │
│    │                                                                  │
│    ├── (no new papers) ──────────────────────────────┐               │
│    ▼                                                  │               │
│  [extraction]  Scholarcy + PyPDF + Claude            │               │
│    │                                                  │               │
│    ▼                                                  │               │
│  [critical_analysis]  Wagner framework + Scite       │               │
│    │                                                  │               │
│    ├── (none included) ─────────────────────────────┤               │
│    ▼                                                  │               │
│  [synthesis]  Cross-paper synthesis + gap update     │               │
│    │                                                  │               │
│    ▼                                                  │               │
│  [document_update]  ◄──── OPTIONAL HUMAN CHECKPOINT │               │
│    │  Lit Review + Proposal (Google Docs + Markdown) │               │
│    ▼                                                  │               │
│  [visualization]  Citation network + gap matrix      │               │
│    │                                                  │               │
│    ▼                ◄────────────────────────────────┘               │
│  [email_reporter]  HTML summary of each run + archive                │
│    │                                                                  │
│  END                                                                  │
│                                                                      │
│  Persistence: SQLite checkpointer (survives crashes, enables        │
│               resume from last checkpoint)                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The 7 Specialized Agents

| Agent | Role | Primary Tools |
|-------|------|---------------|
| **1. Discovery** | Search across 5+ academic databases each run | Semantic Scholar, ArXiv, CrossRef, OpenAlex, Elicit, Perplexity |
| **2. Extraction** | PDF extraction + structured summarization | Scholarcy, PyPDF, PyMuPDF, Claude |
| **3. Critical Analysis** | Wagner blank-spot framework + inclusion/exclusion | Scite, Claude |
| **4. Synthesis** | Cross-paper synthesis + gap identification | Claude, PaperDB |
| **5. Document Update** | Incremental lit review + proposal updates | Google Docs API, Markdown, LaTeX |
| **6. Visualization** | Citation networks + gap matrix + timeline | D3.js-ready JSON, Mermaid, Gephi |
| **7. Email Reporter** | Executive summary of each run + archive | Gmail SMTP, SendGrid, Jinja2 HTML |

---

## Critical Analysis Framework (Wagner)

Every paper is analyzed through Jon Wagner's **blank-spot** method:

- **Blind spots**: Limitations the authors themselves acknowledge
- **Blank spots**: What the study *could have* investigated but *did not* (our analysis)

**Inclusion criteria:**
- Directly addresses system-level (not just component-level) AI/ML reliability
- Studies compositional failures, interface misalignment, or error propagation
- Proposes trust composition methods, formalisms, or metrics
- Addresses EU AI Act/governance from a system-level perspective
- Provides empirical evidence of the integration paradox

**Exclusion criteria:**
- Purely component-level model accuracy improvements
- Tangentially related without clear relevance to any RQ
- Methodologically weak or duplicates existing corpus entry

---

## Research Context

**Topic:** The Integration Paradox: Why Reliable AI/ML Components Compose into Unreliable Systems?

**Research Questions:**
1. **RQ1 (Compositional Formalism):** How should system-level trust be represented and composed across the AI SDLC, and under what conditions does error/risk amplify superlinearly?
2. **RQ2 (Mechanisms & Indicators):** Which integration-failure mechanisms recur across domains and SDLC stages, and what measurable indicators predict them?
3. **RQ3 (Interventions & Assurance):** Which end-to-end SDLC practices measurably reduce integration-paradox risk, and how can evidence be packaged into credible, update-resilient assurance?

**Six Research Gaps tracked:** GAP-1 through GAP-6 (component-centric dominance, no trust-composition calculus, underdeveloped predictors, non-monotone reliability, assurance evidence composition, socio-technical formalization).

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Git
- Anthropic API key (required)

### 2. Installation

```bash
git clone https://github.com/chelvy/phd-litreview-automator.git
cd phd-litreview-automator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or your preferred editor
```

**Minimum required:**
```
ANTHROPIC_API_KEY=sk-ant-...
EMAIL_TO=your.email@university.edu
SMTP_USER=your.email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
OPENALEX_API_KEY=...      # free; keyless OpenAlex requests fail on shared IPs
```
Leave every other optional key blank rather than keeping a placeholder.

### 4. Initialize

```bash
# Initialize database, verify APIs, create templates, send test email
python scripts/initialize.py

# Check status
python main.py status
```

### 5. First Run

```bash
# Dry run (no email, no document or database writes — test everything works)
python main.py run --dry-run

# Full run
python main.py run

# Full run with human review checkpoint (pauses before writing to documents)
python main.py run --human-checkpoint
# ...review, then continue with the thread ID it prints:
python main.py resume --thread-id run-YYYYMMDD-HHMMSS-xxxxxxxx
```

---

## Deployment (GitHub Actions — Recommended)

`.github/workflows/weekly_litreview.yml` runs the pipeline **every Monday at 06:17 UTC**
on the default branch. Each run commits the updated documents, bibliography,
visualizations and `storage/papers_db.json` back to the repository (authored as the
repository owner), so the next run skips papers it has already analyzed. Every run
looks back 21 days; overlapping windows catch papers that sources index late, and
known papers are skipped before any LLM call.

### 1. Add the secrets (one time)

With the [GitHub CLI](https://cli.github.com) signed in (`gh auth login`), copy your
local `.env` into the repository's Actions secrets and variables:

```bash
scripts/setup_github_secrets.sh            # or: scripts/setup_github_secrets.sh owner/repo
```

It also uploads `config/google_service_account.json` (if present) as
`GOOGLE_SERVICE_ACCOUNT_JSON`. To do it by hand instead: **Settings → Secrets and
variables → Actions**.

| Secret | Required | Description |
|--------|----------|-------------|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic Claude API key |
| `EMAIL_TO` | ✅ | Recipient email address |
| `SMTP_USER` / `SMTP_PASSWORD` | ✅ (SMTP) | Gmail address and App Password |
| `OPENALEX_API_KEY` | Recommended | Free key; keyless requests fail on GitHub's shared IPs |
| `SEMANTIC_SCHOLAR_API_KEY` | Recommended | Higher rate limits |
| `EMAIL_FROM`, `EMAIL_CC` | Optional | Sender display name/address, CC list |
| `SENDGRID_API_KEY` | Optional | When the `EMAIL_BACKEND` variable is `sendgrid` |
| `LANGSMITH_API_KEY` | Optional | Tracing (enabled automatically when set) |
| `ELICIT_API_KEY`, `SCITE_API_KEY`, `SCHOLARCY_API_KEY`, `PERPLEXITY_API_KEY` | Optional | Extra signals |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Optional | Google Docs sync (contents of the key file) |
| `LITREVIEW_GOOGLE_DOC_ID`, `PROPOSAL_GOOGLE_DOC_ID` | Optional | Override the document IDs |

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Primary model |
| `ANTHROPIC_FALLBACK_MODEL` | `claude-sonnet-4-6` | Used on errors/refusals of the primary |
| `ANTHROPIC_EFFORT` | per agent | Override reasoning effort for all agents |
| `DAYS_LOOKBACK` | `21` | Search window in days |
| `MAX_PAPERS_PER_RUN`, `MIN_RELEVANCE_SCORE` | `50`, `6.0` | Volume and inclusion threshold |
| `LLM_CONCURRENCY` | `4` | Papers extracted/analyzed in parallel |
| `EMAIL_BACKEND`, `SMTP_HOST`, `SMTP_PORT` | `smtp`, Gmail, `587` | Mail transport |

### 2. Test it

**Actions → Weekly Literature Review → Run workflow**, tick *Dry run* (or
`gh workflow run weekly_litreview.yml -f dry_run=true`). A dry run sends no email and
commits nothing; its log and HTML report are attached to the run as an artifact.

If a scheduled run fails or times out — including when the report email cannot be
sent — the workflow opens (or comments on) an issue labelled `pipeline-failure`.
Results of a partly failed run are still committed, and papers whose analysis or
document update failed are retried automatically the next week.

A run takes roughly 15–90 minutes depending on how many new papers it finds
(papers are processed `LLM_CONCURRENCY`=4 at a time; the job limit is 5.5 hours).

### Running locally as well

GitHub Actions is the single place that should write the corpus. Locally, use
`python main.py run --dry-run` — it never changes `storage/papers_db.json` or the
documents. If you do a full local run (or load seed papers), commit and push
`storage/papers_db.json` and `outputs/` afterwards, and `git pull` before the next
local run so you have the papers the weekly job added.

> **Privacy:** in a public repository the committed literature review, proposal and
> paper database are public, as are the Actions logs and artifacts. Make the
> repository private (Settings → General → Danger Zone) if that is not intended — a
> private repository has 2,000 free Actions minutes per month, enough for weekly runs.

### Running Locally on a Schedule

```bash
# Uses CRON_SCHEDULE and TIMEZONE from .env (default: Mondays 06:17 UTC)
python main.py schedule
```

---

## Output Files

```
outputs/
├── literature_review.md          # Auto-updated literature review
├── research_proposal.md          # Auto-updated research proposal
├── references.bib                # Auto-maintained BibTeX bibliography
├── visualizations/
│   ├── citation_network.json     # For Gephi/D3.js
│   ├── concept_map.mmd           # Mermaid concept map
│   ├── gap_matrix.md             # Gap coverage table
│   └── timeline.mmd              # Research timeline
├── literature_review_updates.tex  # LaTeX versions of each update (for Overleaf)
└── daily_reports/
    └── YYYY-MM-DD_daily_report.html  # Archived email reports (Actions artifact, not committed)

storage/
├── papers_db.json                # Analyzed papers, corpus, run log (tracked in git; updated by the workflow)
└── pipeline_state.db             # LangGraph checkpoints (SQLite, local only)
```

---

## Google Docs Integration

The pipeline can sync updates directly to your existing Google Docs:

**Existing documents:**
- Literature Review: `https://docs.google.com/document/d/1WukjXxp115ZNI4-u-1agm-5du1KG23iA/`
- Research Proposal: `https://docs.google.com/document/d/1lplSQtusQxbcey0Wluj-vtXeTkq90ewX/`
- AI Integration Failures: `https://docs.google.com/document/d/1cP3t5cC1C0_lldC5ER480Z7FCGZwIVcV/`

**Setup options:**

Option A — Service Account (for GitHub Actions):
1. Create a Google Cloud project → Enable Docs and Drive APIs
2. Create a Service Account → Download JSON key → save as `config/google_service_account.json`
3. Share your Google Docs with the service account email (Editor)
4. Run `scripts/setup_github_secrets.sh` (uploads it as `GOOGLE_SERVICE_ACCOUNT_JSON`)

Option B — OAuth2 (for local runs):
1. Create OAuth2 credentials in Google Cloud Console
2. Download `credentials.json` → save to `config/google_credentials.json`
3. Run `python main.py google-auth` → browser sign-in, token saved to `config/google_token.json`

The documents must be native Google Docs: an uploaded `.docx` has to be converted
first (File → Save as Google Docs), and its new ID used. `python scripts/initialize.py`
checks that each document can be opened. Updates are appended with an
`[Auto-updated DATE — SECTION]` marker and never appended twice.

---

## Cost Estimation

Rough estimates per weekly run with up to 50 analyzed papers (Claude API list prices,
Oct 2026). Most discovered papers are filtered out by a cheap pre-screen first.

| Component | Cost per run (est.) | Notes |
|-----------|--------------------|-------|
| `claude-opus-5-5` ($4 / $20 per MTok) | ~$4–12 | extraction + critical analysis per paper, one synthesis, batched document updates |
| `claude-sonnet-5-5` ($2 / $10 per MTok) | ~$2–6 | set the `ANTHROPIC_MODEL` variable |
| Semantic Scholar, arXiv, OpenAlex, CrossRef | Free | API keys are free |
| GitHub Actions | Free (public) / within the free private quota | ~15–90 min per run |

### Cost Optimization Tips

1. Lower `MAX_PAPERS_PER_RUN` or raise `MIN_RELEVANCE_SCORE`
2. Use a cheaper model via the `ANTHROPIC_MODEL` variable
3. Lower `ANTHROPIC_EFFORT` (e.g. `medium`) — each agent otherwise uses its own level
   (low for screening, medium for extraction, high for analysis and writing)

---

## Observability

### LangSmith (Recommended)

Every pipeline run generates a LangSmith trace showing:
- Each agent invocation with full prompt/response
- Token usage per node
- Latency breakdowns
- Error traces with full context

Setup: Set `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` in `.env`

### Local Logs

```bash
# View today's log
cat logs/pipeline_$(date +%Y%m%d).log

# Tail in real-time
tail -f logs/pipeline_$(date +%Y%m%d).log
```

### Pipeline Status

```bash
python main.py status
```

---

## Hallucination Mitigation

1. **Structured JSON outputs** — all LLM outputs validated as JSON before use
2. **Source citation requirements** — every claim in document updates must cite a paper in corpus
3. **Cross-agent verification** — discovery scores are re-evaluated by critical analysis
4. **Scite citation quality** — supporting/contrasting citation counts validate paper credibility
5. **Redundancy check** — synthesis agent explicitly detects and flags contradictions
6. **Dry-run testing** — `--dry-run` allows inspecting LLM outputs before any document changes
7. **Human checkpoint** — `--human-checkpoint` pauses before document writes for review
8. **LangSmith tracing** — every decision is auditable post-run

---

## CLI Reference

```bash
python main.py run                        # Full pipeline run
python main.py run --dry-run              # Dry run (test mode)
python main.py run --human-checkpoint     # Pause before doc updates
python main.py resume --thread-id ID      # Continue a paused run
python main.py run --days-lookback 30     # Override the search window
python main.py run --log-level DEBUG      # Verbose output
python main.py schedule                   # Local scheduler (CRON_SCHEDULE)
python main.py status                     # Show pipeline stats
python main.py export-bibtex              # Export corpus BibTeX
python main.py export-bibtex --output my.bib  # Custom output path
python main.py test-email                 # Verify email configuration
python main.py google-auth                # One-time Google OAuth sign-in
python -m pytest                          # Run the test suite
```

---

## Adding Tracked Authors

Edit `agents/discovery_agent.py`:

```python
TRACKED_AUTHORS = [
    # (Semantic Scholar author ID, display name)
    ("143977260", "Sculley et al."),      # ML technical debt
    ("1741101", "Rahimi, Ali"),           # Random features
]
```

---

## Seeding with Existing Papers

Create a `seed_papers.json` file with papers you already know about and run the
command below. Papers with `"inclusion_decision": true` join the corpus (marked as
already documented, so they are not written into your documents again); the others
are recorded as excluded. The pipeline never re-analyzes seeded papers.

```bash
python scripts/initialize.py --seed-papers seed_papers.json
```

Format:
```json
[
  {
    "paper_id": "doi:10.1145/3377930.3389821",
    "title": "Compositional Reliability in Machine Learning Systems",
    "authors": ["Smith, J.", "Jones, A."],
    "year": 2024,
    "abstract": "...",
    "doi": "10.1145/3377930.3389821",
    "venue": "ICSE 2024",
    "citation_count": 45,
    "relevance_score": 8.5,
    "gaps_addressed": ["GAP-1", "GAP-3"],
    "inclusion_decision": true
  }
]
```

---

## Project Structure

```
PhD-LitReview-Automator/
├── agents/                     # 7 specialized agents
│   ├── discovery_agent.py      # Multi-source academic search
│   ├── extraction_agent.py     # PDF + structured extraction
│   ├── critical_analysis_agent.py  # Wagner framework analysis
│   ├── synthesis_agent.py      # Cross-paper synthesis
│   ├── document_update_agent.py    # Living document maintenance
│   ├── visualization_agent.py  # Citation networks + maps
│   ├── email_reporter_agent.py # Run report generation
│   └── llm_utils.py            # Shared Claude client: model fallback, response parsing
├── config/
│   ├── settings.py             # Pydantic settings from .env
│   └── prompts.py              # All agent system prompts
├── pipeline/
│   └── graph.py                # LangGraph orchestration
├── state/
│   └── pipeline_state.py       # Shared TypedDict state
├── tools/                      # API integrations
│   ├── semantic_scholar.py     # Semantic Scholar API
│   ├── arxiv_tool.py           # ArXiv API
│   ├── crossref_tool.py        # CrossRef API
│   ├── openalex_tool.py        # OpenAlex API
│   ├── elicit_tool.py          # Elicit API
│   ├── scite_tool.py           # Scite smart citations
│   ├── scholarcy_tool.py       # PDF extraction
│   ├── perplexity_tool.py      # Deep research queries
│   ├── google_docs.py          # Google Docs sync
│   ├── email_tool.py           # Email dispatch (SMTP / SendGrid)
│   ├── bibtex.py               # Deterministic BibTeX from paper metadata
│   └── paper_db.py             # Local paper database + cross-source de-duplication
├── scripts/
│   ├── initialize.py           # Setup & verification
│   └── setup_github_secrets.sh # Copy .env into GitHub Actions secrets
├── .github/workflows/
│   └── weekly_litreview.yml    # GitHub Actions (Mondays 06:17 UTC)
├── tests/                      # pytest suite (unit + end-to-end with a fake LLM)
├── outputs/                    # Generated documents
├── storage/                    # papers_db.json (tracked; updated by the workflow)
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example
└── BENCHMARKING.md             # Framework selection rationale
```

---

## License

MIT License. Research use permitted with attribution.

---

*Built with LangGraph and the Anthropic Claude API.*
