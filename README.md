# PhD LitReview Automator

A production-grade, daily-running multi-agent pipeline that automates and maintains
a PhD-level literature review process for research on **"The Integration Paradox:
Why Reliable AI/ML Components Compose into Unreliable Systems?"**

**Framework:** LangGraph (selected after rigorous 2026-era benchmarking — see [BENCHMARKING.md](BENCHMARKING.md))
**LLM:** Claude claude-sonnet-4-6 (Anthropic)
**Runs:** Daily at 6:00 AM UTC via GitHub Actions

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
│  [email_reporter]  Daily HTML summary + archive                      │
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
| **1. Discovery** | Daily search across 5+ academic databases | Semantic Scholar, ArXiv, CrossRef, OpenAlex, Elicit, Perplexity |
| **2. Extraction** | PDF extraction + structured summarization | Scholarcy, PyPDF, PyMuPDF, Claude claude-sonnet-4-6 |
| **3. Critical Analysis** | Wagner blank-spot framework + inclusion/exclusion | Scite, Claude claude-sonnet-4-6 |
| **4. Synthesis** | Cross-paper synthesis + gap identification | Claude claude-sonnet-4-6, PaperDB |
| **5. Document Update** | Incremental lit review + proposal updates | Google Docs API, Markdown, LaTeX |
| **6. Visualization** | Citation networks + gap matrix + timeline | D3.js-ready JSON, Mermaid, Gephi |
| **7. Email Reporter** | Daily executive summary + archive | Gmail SMTP, SendGrid, Jinja2 HTML |

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

- Python 3.11+
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
```

### 4. Initialize

```bash
# Initialize database, verify APIs, create templates, send test email
python scripts/initialize.py

# Check status
python main.py status
```

### 5. First Run

```bash
# Dry run (no email, no document writes — test everything works)
python main.py run --dry-run

# Full run
python main.py run

# Full run with human review checkpoint (pauses before writing to documents)
python main.py run --human-checkpoint
```

---

## Deployment (GitHub Actions — Recommended)

The pipeline runs automatically via GitHub Actions every day at 6:00 AM UTC.

### Setup GitHub Secrets

In your repository: **Settings → Secrets and variables → Actions → New repository secret**

| Secret | Required | Description |
|--------|----------|-------------|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic Claude API key |
| `EMAIL_TO` | ✅ | Recipient email address |
| `SMTP_USER` | ✅ | Gmail address |
| `SMTP_PASSWORD` | ✅ | Gmail App Password |
| `SEMANTIC_SCHOLAR_API_KEY` | Recommended | Higher rate limits |
| `LANGSMITH_API_KEY` | Recommended | Pipeline observability |
| `ELICIT_API_KEY` | Optional | Enhanced extraction |
| `SCITE_API_KEY` | Optional | Citation quality signals |
| `SCHOLARCY_API_KEY` | Optional | PDF extraction |
| `PERPLEXITY_API_KEY` | Optional | Deep research queries |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Optional | Google Docs sync |
| `LITREVIEW_GOOGLE_DOC_ID` | Optional | Your lit review doc ID |
| `PROPOSAL_GOOGLE_DOC_ID` | Optional | Your proposal doc ID |

### Manual Trigger

Go to **Actions → PhD LitReview Daily Pipeline → Run workflow** to trigger manually.

### Running Locally on a Schedule

```bash
# Start local scheduler (runs daily at 6 AM UTC)
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
└── daily_reports/
    └── YYYY-MM-DD_daily_report.html  # Archived email reports

storage/
├── papers_db.json                # All discovered papers (TinyDB)
└── pipeline_state.db             # LangGraph checkpoints (SQLite)
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
2. Create a Service Account → Download JSON key
3. Share your Google Docs with the service account email
4. Set `GOOGLE_SERVICE_ACCOUNT_JSON` as a GitHub secret

Option B — OAuth2 (for local runs):
1. Create OAuth2 credentials in Google Cloud Console
2. Download `credentials.json` → save to `config/google_credentials.json`
3. Run `python scripts/initialize.py` → browser auth flow on first use

---

## Cost Estimation

| Component | Cost/Day (est.) | Notes |
|-----------|----------------|-------|
| Claude claude-sonnet-4-6 | ~$0.40–1.20 | 7 agent invocations × 50 papers max; estimate based on current pricing |
| Semantic Scholar | Free | Free tier (100 req/5min) |
| ArXiv | Free | Free |
| OpenAlex | Free | Free |
| CrossRef | Free | Free (polite pool) |
| Elicit | $0–15/mo | Depends on tier |
| Scite | $0–20/mo | Depends on tier |
| GitHub Actions | Free | 2000 min/month free |
| LangSmith | Free | Developer tier (5000 traces/mo) |

**Total estimated daily cost: $0.40–$1.20 (LLM only)**
**Monthly: ~$12–36**

### Cost Optimization Tips

1. **Reduce `MAX_PAPERS_PER_RUN`** from 50 to 20 on quieter days (fewer LLM calls)
2. **Cache Semantic Scholar results** — don't re-fetch same papers
3. **Use Claude Haiku** for extraction (cheaper), Sonnet only for analysis/synthesis
4. **Batch LLM calls** — the extraction agent already batches 10 papers per call
5. **Skip extraction for low-relevance papers** — only extract papers with preliminary score ≥ 6.0

```env
# In .env for cost-optimized mode:
MAX_PAPERS_PER_RUN=20
MIN_RELEVANCE_SCORE=7.0
DAYS_LOOKBACK=3
```

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
python main.py run --log-level DEBUG      # Verbose output
python main.py schedule                   # Start daily scheduler
python main.py status                     # Show pipeline stats
python main.py export-bibtex              # Export corpus BibTeX
python main.py export-bibtex --output my.bib  # Custom output path
python main.py test-email                 # Verify email configuration
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

Create a `seed_papers.json` file with paper metadata and run:

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
│   └── email_reporter_agent.py # Daily report generation
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
│   ├── email_tool.py           # Email dispatch
│   └── paper_db.py             # Local paper database
├── scripts/
│   └── initialize.py           # Setup & verification
├── .github/workflows/
│   └── daily_run.yml           # GitHub Actions (daily 6AM UTC)
├── outputs/                    # Generated documents
├── storage/                    # Database files (gitignored)
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example
└── BENCHMARKING.md             # Framework selection rationale
```

---

## License

MIT License. Research use permitted with attribution.

---

*Built with LangGraph + Claude claude-sonnet-4-6 by the PhD LitReview Automator.*
