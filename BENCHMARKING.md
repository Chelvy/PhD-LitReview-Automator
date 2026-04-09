# Framework Benchmarking Report — 2026 Era
## PhD LitReview Automator: Agentic Framework Selection

**Benchmarked:** April 2026 | **Purpose:** Select optimal multi-agent framework for a daily-running,
iterative, production-grade academic literature review pipeline.

---

## Evaluation Criteria & Scoring (1–10)

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **C1** Role-based agents + memory | 15% | Specialized agents with long-term memory and tool use |
| **C2** Persistent state | 20% | Native support for shared state across daily runs (critical) |
| **C3** Tool integration | 15% | API calls, MCP protocol, browser tools, function calling |
| **C4** Scheduling/deployment | 10% | GitHub Actions, cron, cloud hosting, native scheduler |
| **C5** Document generation | 10% | Markdown → Google Docs/PDF, version control integration |
| **C6** Email automation | 5% | Email notification support |
| **C7** Scalability & reliability | 15% | Hundreds of papers, hallucination mitigation |
| **C8** Production readiness | 10% | Cost, observability, human-in-the-loop |

---

## Framework Scorecards

### 1. LangGraph (LangChain Ecosystem)

**Version:** 0.2.50+ (2026) | **Language:** Python

| Criterion | Score | Evidence |
|-----------|-------|---------|
| C1: Role-based agents + memory | **9/10** | `create_react_agent`, `ToolNode`; any LLM with any tool set; long-term memory via `MemorySaver`/checkpointers; `SqliteSaver`/`PostgresSaver` for cross-run persistence |
| C2: Persistent state | **10/10** | Native `SqliteSaver`, `PostgresSaver`, `RedisSaver` checkpointers; state survives crashes, enables incremental daily updates; `TypedDict` state shared across ALL nodes with `Annotated[list, operator.add]` for safe merging |
| C3: Tool integration | **9/10** | First-class LangChain tool ecosystem (1000+ tools); MCP server integration via `langchain-mcp`; full async/streaming support; `@tool` decorator + Pydantic validation |
| C4: Scheduling/deployment | **9/10** | Deploy on GitHub Actions (zero infra), GCP/AWS/Azure, LangGraph Platform (managed, auto-scaling); any cron/APScheduler; LangGraph Cloud handles orchestration natively |
| C5: Document generation | **8/10** | No native doc generation but integrates with Google Docs API, LaTeX, Markdown; state nodes can write files; excellent for maintaining running documents |
| C6: Email automation | **8/10** | Via LangChain tools (SMTP, SendGrid, Gmail API); email node in graph |
| C7: Scalability & reliability | **9/10** | Async-first; sub-graph parallel execution; retry policies in edges; verification nodes natively supported in graph; structured outputs with Pydantic reduce hallucinations; LangSmith for error tracing |
| C8: Production readiness | **9/10** | LangSmith observability (traces, evals, monitoring); human-in-the-loop via `interrupt_before`; LangGraph Platform with auto-scaling; MIT license; massive community (60k+ GitHub stars) |

**Weighted Score: 9.15/10** ⭐⭐⭐⭐⭐

**Unique Strengths for This Use Case:**
- Persistent state across daily runs is **native and trivial** (SQLite checkpointer)
- Graph structure perfectly models the pipeline's sequential + conditional flow
- `interrupt_before` provides frictionless human-in-the-loop checkpoints
- LangSmith gives full observability into every LLM call (critical for debugging)
- State `Annotated[list, operator.add]` safely handles parallel agent outputs

---

### 2. CrewAI

**Version:** 0.80+ (2026) | **Language:** Python

| Criterion | Score | Evidence |
|-----------|-------|---------|
| C1: Role-based agents + memory | **9/10** | Excellent role definitions, goal/backstory system; `Agent(memory=True)` with vector store; strong for collaborative multi-agent crews |
| C2: Persistent state | **5/10** | CrewAI's memory is primarily short-term (within a crew run); persistent cross-run state requires manual database integration; no native checkpointer equivalent to LangGraph |
| C3: Tool integration | **8/10** | Good tool ecosystem; LangChain tool compatibility; custom tool support |
| C4: Scheduling/deployment | **6/10** | No native scheduler; must integrate with APScheduler or cron; CrewAI+ cloud (2026) adds some hosting but not mature |
| C5: Document generation | **6/10** | Agents can write files; no native document management; requires custom tooling |
| C6: Email automation | **6/10** | Via custom tools; not native |
| C7: Scalability & reliability | **7/10** | Good for bounded tasks; less proven at high-volume daily incremental workloads |
| C8: Production readiness | **7/10** | Growing community; CrewAI+ cloud platform; observability via LangSmith integration; human review requires custom implementation |

**Weighted Score: 6.85/10** ⭐⭐⭐

**Key Gap:** Persistent state across daily runs is not native — would require significant custom plumbing for our incremental update workflow. No built-in checkpointing means pipeline restarts lose all in-progress state.

---

### 3. Microsoft AutoGen / AG2

**Version:** 0.4+ (AutoGen Studio 2026) | **Language:** Python

| Criterion | Score | Evidence |
|-----------|-------|---------|
| C1: Role-based agents + memory | **8/10** | Strong conversational agent patterns; `AssistantAgent`, `UserProxyAgent`; memory via vector stores |
| C2: Persistent state | **6/10** | AG2 (community fork 2025+) adds better state management; but not as native as LangGraph's typed state + checkpointers |
| C3: Tool integration | **8/10** | Good function calling support; code execution capabilities |
| C4: Scheduling/deployment | **6/10** | AutoGen Studio has some deployment options; no native cron; GitHub Actions deployment possible |
| C5: Document generation | **6/10** | Can generate documents via code execution agent; awkward for structured doc maintenance |
| C6: Email automation | **5/10** | Possible but requires custom tool or code execution |
| C7: Scalability & reliability | **7/10** | Good for multi-agent conversation; less proven for high-throughput daily document pipelines |
| C8: Production readiness | **6/10** | Microsoft backing but complex; Microsoft/community split (AutoGen vs AG2) creates uncertainty; limited LangSmith-equivalent observability |

**Weighted Score: 6.65/10** ⭐⭐⭐

**Key Gap:** The conversation-centric design is less suited to our state-machine-like pipeline. The Microsoft/AG2 community fork fragmentation creates maintenance risk.

---

### 4. OpenAI Swarm

**Version:** 0.0.1 (experimental) | **Language:** Python

| Criterion | Score | Evidence |
|-----------|-------|---------|
| C1: Role-based agents + memory | **5/10** | Simple agent handoff concept; no built-in memory |
| C2: Persistent state | **2/10** | Context window only; no persistence mechanism |
| C3: Tool integration | **6/10** | OpenAI function calling; limited tool ecosystem |
| C4: Scheduling/deployment | **3/10** | Educational framework; not production-ready |
| C5: Document generation | **2/10** | No native support |
| C6: Email automation | **2/10** | Not supported |
| C7: Scalability & reliability | **3/10** | Experimental; not designed for production |
| C8: Production readiness | **1/10** | Explicitly marked "experimental"; OpenAI themselves say not for production |

**Weighted Score: 3.25/10** ⭐

**Verdict:** Eliminated. OpenAI Swarm is explicitly educational. OpenAI now recommends their Agents SDK (2025) over Swarm, but the Agents SDK still lacks the stateful, iterative pipeline capabilities we need.

---

### 5. n8n (with AI Agents)

**Version:** 1.60+ (2026) | **Language:** Node.js/Visual |

| Criterion | Score | Evidence |
|-----------|-------|---------|
| C1: Role-based agents + memory | **6/10** | AI Agent node supports tool use and basic memory; less flexible than code-based frameworks for specialized PhD-level prompting |
| C2: Persistent state | **7/10** | Native workflow state; database nodes; but state schema is not typed/structured |
| C3: Tool integration | **8/10** | 400+ native integrations; HTTP nodes for any API; Google Docs, Gmail native |
| C4: Scheduling/deployment | **9/10** | Built-in cron scheduling; self-hosted or cloud; excellent for daily runs |
| C5: Document generation | **7/10** | Native Google Docs node; Markdown rendering; file operations |
| C6: Email automation | **9/10** | Native Gmail, SMTP, SendGrid nodes; templated emails |
| C7: Scalability & reliability | **6/10** | Visual workflows limit complex LLM orchestration and structured output handling |
| C8: Production readiness | **7/10** | Self-hosted free tier; n8n Cloud paid; good community; limited Python ML libraries |

**Weighted Score: 7.20/10** ⭐⭐⭐

**Key Gap:** n8n excels at integrations and scheduling but is weak on complex LLM orchestration, structured output validation, and Python-native ML tooling. The visual paradigm makes it difficult to implement nuanced agent logic (Wagner framework analysis, multi-source synthesis). No native support for PDF parsing, ArXiv API, or academic search SDKs.

---

### 6. Dify

**Version:** 0.14+ (2026) | **Language:** Python/YAML |

| Criterion | Score | Evidence |
|-----------|-------|---------|
| C1: Role-based agents + memory | **7/10** | Agent workflows with tool calling; conversation memory; but limited to Dify's abstractions |
| C2: Persistent state | **6/10** | Conversation memory and variable state; cross-run persistence requires external DB |
| C3: Tool integration | **7/10** | Plugin marketplace; HTTP tools; limited Python extensibility |
| C4: Scheduling/deployment | **5/10** | No native scheduler; requires external trigger; cloud-hosted |
| C5: Document generation | **7/10** | Knowledge base integration; document generation via LLM |
| C6: Email automation | **4/10** | Limited email support; requires external webhook |
| C7: Scalability & reliability | **6/10** | Good for single-workflow but limited for complex multi-stage pipelines |
| C8: Production readiness | **7/10** | LLMOps focus; good observability; SaaS pricing model |

**Weighted Score: 6.30/10** ⭐⭐⭐

**Key Gap:** Dify is excellent for LLM app deployment but not designed for complex, multi-step research automation pipelines. Limited Python extensibility prevents integrating academic-specific tools (arxiv, semanticscholar, PyPDF).

---

### 7. MetaGPT

**Version:** 0.8+ (2026) | **Language:** Python |

| Criterion | Score | Evidence |
|-----------|-------|---------|
| C1: Role-based agents + memory | **8/10** | Excellent role-based design; Product Manager, Engineer, QA roles natively |
| C2: Persistent state | **5/10** | Memory system exists but optimized for software development context |
| C3: Tool integration | **6/10** | Good but limited to built-in roles; less flexible for academic tools |
| C4: Scheduling/deployment | **4/10** | No native scheduler; designed for one-shot complex tasks |
| C5: Document generation | **8/10** | Excellent software documentation generation; adaptable to academic |
| C6: Email automation | **4/10** | Not native |
| C7: Scalability & reliability | **6/10** | Good for software project contexts; not proven for daily iterative academic pipelines |
| C8: Production readiness | **6/10** | Active development; community driven; less production tooling than LangGraph |

**Weighted Score: 5.95/10** ⭐⭐⭐

**Key Gap:** MetaGPT is optimized for software development workflows, not academic research pipelines. The role definitions (PM, Engineer, QA) don't map naturally to Discovery/Extraction/Synthesis/Document Update agents.

---

### 8. LlamaIndex Workflows

**Version:** 0.12+ (2026) | **Language:** Python |

| Criterion | Score | Evidence |
|-----------|-------|---------|
| C1: Role-based agents + memory | **7/10** | Good agent and RAG capabilities; excellent for retrieval-augmented generation |
| C2: Persistent state | **6/10** | Workflow state management; IndexStore for persistence; less native than LangGraph |
| C3: Tool integration | **8/10** | Excellent for document ingestion, vector stores, and retrieval; strong API tool support |
| C4: Scheduling/deployment | **5/10** | No native scheduler; LlamaCloud for hosting |
| C5: Document generation | **8/10** | Strong document processing; excellent for RAG over existing lit review |
| C6: Email automation | **4/10** | Not native |
| C7: Scalability & reliability | **7/10** | Very strong for retrieval-heavy workloads; hallucination reduction via RAG |
| C8: Production readiness | **7/10** | LlamaCloud managed hosting; good community; strong enterprise adoption |

**Weighted Score: 6.75/10** ⭐⭐⭐

**Key Gap:** LlamaIndex is excellent for RAG and document retrieval but its workflow system is less mature than LangGraph for complex multi-agent orchestration. Best as a complementary library (e.g., for vector search over the corpus) rather than the primary orchestration framework.

---

## Final Comparison Table

| Framework | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | **Weighted Score** |
|-----------|----|----|----|----|----|----|----|----|-------------------|
| **LangGraph** | 9 | **10** | 9 | 9 | 8 | 8 | 9 | 9 | **9.15 ✅ WINNER** |
| n8n | 6 | 7 | 8 | 9 | 7 | 9 | 6 | 7 | 7.20 |
| CrewAI | 9 | 5 | 8 | 6 | 6 | 6 | 7 | 7 | 6.85 |
| LlamaIndex | 7 | 6 | 8 | 5 | 8 | 4 | 7 | 7 | 6.75 |
| AutoGen/AG2 | 8 | 6 | 8 | 6 | 6 | 5 | 7 | 6 | 6.65 |
| Dify | 7 | 6 | 7 | 5 | 7 | 4 | 6 | 7 | 6.30 |
| MetaGPT | 8 | 5 | 6 | 4 | 8 | 4 | 6 | 6 | 5.95 |
| OpenAI Swarm | 5 | 2 | 6 | 3 | 2 | 2 | 3 | 1 | 3.25 |

---

## Selected Framework: LangGraph ✅

### Justification (Why LangGraph Wins for This Specific Use Case)

**The decisive factor is Criterion C2 (Persistent State — 20% weight), where LangGraph scores
10/10 and all competitors score 5–7/10.**

This is not an arbitrary criterion — it is the architectural requirement that makes or breaks
a *daily, iterative, ongoing* literature review pipeline:

1. **The pipeline must remember what it processed yesterday.** Without native persistence,
   every daily run starts from scratch or requires complex custom state management. LangGraph's
   `SqliteSaver` provides turnkey persistence across runs with zero extra code.

2. **Daily incremental updates require atomic, crash-safe state.** If the pipeline crashes
   mid-run (e.g., API timeout at step 4 of 7), LangGraph resumes from the last checkpoint.
   No other framework provides this out-of-the-box.

3. **The graph metaphor perfectly captures our pipeline's conditional flow.** The
   `deduplicate → [conditional: new papers?] → extract` and `critical_analysis →
   [conditional: included?] → synthesize` routing maps directly to LangGraph's
   `add_conditional_edges` with named routing functions.

4. **LangSmith provides the observability we need.** For a research pipeline where we must
   audit every LLM decision (inclusion/exclusion, gap analysis), LangSmith's trace inspection
   is essential — not a luxury.

5. **Human-in-the-loop is a first-class citizen.** The `interrupt_before=["document_update"]`
   feature allows the researcher to review AI-proposed document changes before they're applied,
   preserving academic integrity without requiring pipeline redesign.

6. **The ecosystem depth is unmatched.** 60k+ GitHub stars, LangGraph Platform for managed
   deployment, active Anthropic/LangChain collaboration, and hundreds of compatible tools
   mean we're building on a stable, well-supported foundation.

**Why not CrewAI (runner-up)?** CrewAI's role-based agents are excellent, but the lack of
native cross-run persistence (C2: 5/10) is a fatal flaw for our use case. Every daily run
would require custom database plumbing to avoid reprocessing 500+ papers.

**Why not n8n (2nd highest)?** n8n is superb for simple integrations and scheduling but
is fundamentally a visual workflow tool that cannot implement the nuanced, PhD-level agent
logic our pipeline requires (Wagner framework analysis, structured JSON validation, academic
search SDK integration, PDF parsing).
