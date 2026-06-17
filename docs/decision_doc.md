# Decision Document — Slack AI Proposal Generator

## 1. Overview

This document records the key architectural, technical, and design decisions made while building the Slack-based AI proposal generator. Each section explains **what** was decided, **why**, and **what alternatives were considered**.

---

## 2. Architecture Decisions

### 2.1 Why 4 Agents Instead of 3?

**Decision:** Use 4 agents — Intake, Research, Writer, Revision — instead of the minimum 3.

**Reasoning:**
The assessment requires both initial proposal generation AND follow-up capabilities (asking questions about the transcript, editing specific sections). These are fundamentally different cognitive tasks:

| Task | Requires | Agent |
|------|----------|-------|
| Extract client info from transcript | Structured extraction, validation | **Intake** |
| Find relevant past proposals | Semantic search, fallback logic | **Research** |
| Generate a new proposal | Creative writing, tone adaptation | **Writer** |
| Answer questions + edit sections | Transcript Q&A, surgical document editing | **Revision** |

Merging Revision into the Writer would violate single-responsibility: the Writer optimizes for creative generation (temperature=0.7), while the Revision agent optimizes for precision (temperature=0). They use different prompts, different tools, and different Qdrant collections.

**Alternative considered:** 3 agents with the Writer handling revisions. Rejected because:
- The prompt complexity would double
- Temperature requirements conflict (creative vs. precise)
- Testing and debugging become harder when one agent does two very different jobs

---

### 2.2 Agent Communication: Shared State vs. Message Passing

**Decision:** Use a shared `ProposalState` (TypedDict) that flows through the LangGraph pipeline.

**Reasoning:**
LangGraph is built around shared state. Each agent reads from and writes to the same state dictionary. This is simpler than message passing for a linear pipeline and gives us:
- Full visibility into what each agent produced
- Easy state persistence (serialize to SQLite)
- Clean interrupt/resume for human-in-loop

**Alternative considered:** LangChain agent message passing (each agent sends a message to the next). Rejected because:
- Harder to persist mid-pipeline state
- Harder to implement human-in-loop interrupts
- No clear benefit for a 4-step linear pipeline

---

### 2.3 Why LangGraph Over Vanilla LangChain Chains?

**Decision:** Use LangGraph's `StateGraph` for pipeline orchestration.

**Reasoning:**
1. **Human-in-loop:** LangGraph has built-in `interrupt()` support — pause the graph, wait for user input, resume from where we left off. LangChain chains don't have this.
2. **Conditional routing:** The Research agent has a 3-layer fallback (RAG → web search → zero-shot) that requires conditional edges based on state.
3. **Checkpointing:** LangGraph's `MemorySaver` persists state across interrupts automatically.
4. **Assessment alignment:** The assessment explicitly asks for "inter-agent communication" and "state management" — LangGraph provides both out of the box.

**Alternative considered:** Simple async function calls (no framework). Rejected because we'd have to manually implement state persistence, interrupts, and conditional routing.

---

## 3. RAG Decisions

### 3.1 Two Qdrant Collections

**Decision:** Separate `past_proposals` and `transcripts` into two distinct Qdrant collections.

**Reasoning:**
- **Different use cases:** Proposals are searched by the Research agent for context. Transcripts are searched by the Revision agent for Q&A.
- **Different metadata:** Proposals have `industry`, `source` (seed/generated). Transcripts have `thread_ts` (session ID).
- **No cross-contamination:** A search for "similar proposals" should never return transcript chunks.

**Alternative considered:** Single collection with a `type` metadata filter. Rejected because:
- Higher risk of cross-contamination in search results
- Harder to manage different chunking strategies
- Separate collections are cleaner and easier to reason about

---

### 3.2 Smart Save Gate (Deduplication)

**Decision:** Before saving a generated proposal back to Qdrant, run a similarity check against existing proposals in the same industry. If the similarity score exceeds 0.85, skip the save.

**Reasoning:**
Without dedup, the database bloats rapidly:
- User generates 10 healthcare proposals → 10 nearly identical entries
- RAG results become repetitive and unhelpful
- Storage and search performance degrade

The Smart Save Gate solves this by:
1. Embedding the first 500 chars of the new proposal
2. Searching `past_proposals` filtered by the same industry
3. If the top match scores > 0.85 → skip (too similar)
4. If ≤ 0.85 → save (novel enough to be useful)

**Why 0.85 as the threshold?**
- Below 0.8: Too aggressive — filters out genuinely similar but useful proposals
- Above 0.9: Too lenient — allows near-duplicates through
- 0.85 is a balanced default (configurable via `DEDUP_THRESHOLD` in `.env`)

**Alternative considered:** Save everything, deduplicate later with a batch job. Rejected because it adds operational complexity and the DB stays dirty between batch runs.

---

### 3.3 Revision Updates Instead of New Inserts

**Decision:** When a proposal is revised, UPDATE the existing Qdrant entry (delete old chunks, re-insert new chunks under the same `thread_ts`) instead of creating a new entry.

**Reasoning:**
If we insert a new entry on every revision:
- Original draft + revision 1 + revision 2 = 3 entries for the same proposal
- RAG results would return outdated drafts
- The "latest" version is ambiguous

By using `thread_ts` as the logical ID and updating in-place, the database always contains only the latest version of each proposal.

---

### 3.4 Embedding Model: all-MiniLM-L6-v2

**Decision:** Use `sentence-transformers/all-MiniLM-L6-v2` for text embeddings.

**Reasoning:**
| Model | Dimensions | Speed | Quality | Size |
|-------|-----------|-------|---------|------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | 80MB |
| all-mpnet-base-v2 | 768 | Slower | Better | 420MB |
| OpenAI text-embedding-3-small | 1536 | API call | Best | N/A |

We chose MiniLM because:
- **Local execution:** No API calls, no cost, no latency, no rate limits
- **Good enough:** For proposal similarity matching, MiniLM's quality is sufficient
- **Fast:** Embeddings generate in milliseconds, important for interactive Slack bot
- **Small footprint:** 80MB vs 420MB for mpnet

**Alternative considered:** OpenAI embeddings. Rejected because:
- Adds another API dependency and cost
- Assessment doesn't require it
- Latency for every embedding operation

---

## 4. LLM Decisions

### 4.1 Groq + Llama 3.3 70B Versatile

**Decision:** Use Groq as the LLM provider with `llama-3.3-70b-versatile` as the model.

**Reasoning:**
- **Speed:** Groq's custom LPU hardware delivers ~500 tokens/sec — critical for a Slack bot where users expect fast responses
- **Quality:** Llama 3.3 70B is one of the best open-source models for instruction-following and structured extraction
- **Cost:** Groq's free tier is generous enough for assessment/demo purposes
- **No vendor lock-in:** Using LangChain's `ChatGroq` wrapper means switching to another provider (OpenAI, Anthropic) is a single config change

**Alternative considered:** OpenAI GPT-4o. Rejected because:
- Slower response times (important for Slack UX)
- Higher cost
- Assessment didn't specify OpenAI

---

### 4.2 Temperature Strategy

**Decision:** Use different temperatures for different agents:

| Agent | Temperature | Reason |
|-------|------------|--------|
| Intake | 0.0 | Deterministic extraction — same transcript should always produce the same ClientInfo |
| Research | N/A | No LLM call — pure vector search |
| Writer | 0.7 | Creative writing needs variation — proposals shouldn't sound identical |
| Revision | 0.0 | Precision edits — rewrite exactly what the user asked, nothing more |

---

## 5. Slack Integration Decisions

### 5.1 FastAPI + Slack Bolt (HTTP Mode)

**Decision:** Use FastAPI as the web server with `slack-bolt` mounted at `/slack/events`.

**Reasoning:**
- FastAPI gives us a health check endpoint (`/health`) for monitoring
- Slack Bolt handles all the Slack API complexity (event verification, OAuth)
- HTTP mode (vs Socket Mode) works behind load balancers and in containers
- FastAPI's async support aligns with our async LangGraph pipeline

**Alternative considered:** Socket Mode (WebSocket connection to Slack). Works well for development but doesn't scale in production behind load balancers.

---

### 5.2 Four Human-in-Loop Touchpoints

**Decision:** Pause the pipeline at 4 points and wait for user input in Slack:

| # | When | Why |
|---|------|-----|
| 1 | After Intake | Let user verify/correct extracted info before we search |
| 2 | After Research (no RAG match) | Don't spend Tavily API credits without user approval |
| 3 | Before Writer | Tone is subjective — user should choose |
| 4 | After Writer (revision loop) | User reviews output, asks questions, requests changes |

**Reasoning:**
The assessment explicitly requires "human oversight at key decision points." These 4 points are where human judgment adds the most value — the agent shouldn't guess about extracted info, spend money on web searches, or choose a tone without asking.

---

## 6. Persistence Decisions

### 6.1 SQLite for Session State

**Decision:** Use SQLite (via SQLAlchemy ORM) for persisting pipeline state.

**Reasoning:**
- **Zero infrastructure:** No database server to manage — SQLite is a single file
- **Assessment-appropriate:** For a demo with low concurrency, SQLite is perfect
- **Cross-session recall:** The assessment requires the system to remember transcripts and drafts "if the user returns the next day"
- **JSON state storage:** `state_json` column stores the full `ProposalState` as JSON — easy to serialize/deserialize

**Alternative considered:** PostgreSQL. Better for production but overkill for an assessment demo and adds Docker complexity.

---

## 7. Document Generation Decisions

### 7.1 Markdown → DOCX Conversion

**Decision:** Generate proposals in Markdown (from the LLM), then convert to DOCX using `python-docx`.

**Reasoning:**
- **LLMs are good at Markdown:** Asking the LLM to output Markdown is natural and reliable
- **Programmatic styling:** `python-docx` gives full control over fonts, colors, headers, cover pages
- **No external tools:** No need for Pandoc, LibreOffice, or other system dependencies
- **Bytes output:** DOCX is generated as in-memory bytes — no temporary files on disk

**Alternative considered:** Direct DOCX generation (skip Markdown). Rejected because:
- LLMs produce much better output in Markdown format
- Markdown is easier to parse, store, and display in Slack
- Section-level editing in the Revision agent works naturally on Markdown

---

## 8. Trade-offs and Known Limitations

### What We Traded Off

| Trade-off | Chose | Over | Reason |
|-----------|-------|------|--------|
| Speed vs. Quality | Groq (fast) | OpenAI (higher quality) | Slack UX requires fast responses |
| Local vs. Cloud embeddings | Local (MiniLM) | Cloud (OpenAI) | No API cost, no latency |
| SQLite vs. PostgreSQL | SQLite | PostgreSQL | Zero infrastructure for assessment |
| HTTP mode vs. Socket mode | HTTP | Socket | Production-ready, works behind load balancers |

### Known Limitations

1. **Single-worker:** The FastAPI app runs with `--workers 1` because SQLite doesn't handle concurrent writes well. For production, switch to PostgreSQL.
2. **In-memory checkpointer:** LangGraph's `MemorySaver` doesn't survive process restarts. For production, use `SqliteSaver` or `PostgresSaver`.
3. **No authentication:** The Slack bot responds to any user in the channel. For production, add user allowlisting.
4. **Embedding model is English-only:** `all-MiniLM-L6-v2` performs best on English text. Multilingual support would require `paraphrase-multilingual-MiniLM-L12-v2`.

---

## 9. Assessment Requirement Mapping

| Requirement | Where It's Implemented |
|-------------|----------------------|
| Minimum 3 agents | 4 agents: Intake, Research, Writer, Revision |
| Each agent has one clear job | See Section 2.1 |
| Inter-agent communication via shared state | `ProposalState` TypedDict via LangGraph |
| RAG with metadata filtering | Qdrant `search_with_metadata_filter()` — filters by industry, source |
| Human-in-loop at decision points | 4 `interrupt()` points in LangGraph |
| Follow-up questions | Revision agent searches `transcripts` collection |
| Section-level edits | Revision agent's `_edit_section()` with regex parsing |
| DOCX generation | `docx_generator.py` with cover page, styled sections |
| State persistence | SQLite `SessionModel` with `state_json` |
| Self-learning RAG | Smart Save Gate — saves novel proposals back to Qdrant |
| Internet search fallback | Tavily API via `web_search.py` |
