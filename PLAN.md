# Sales Intelligence Platform – Architecture & Implementation Plan

> **Stack**: DeepSeek (open-source LLM) · SQLite · Python · Docker (WSL2)
> **Date**: 2026-05-13

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [RAG vs TAG – Decision Guide](#2-rag-vs-tag--decision-guide)
3. [Recommended Architecture: TAG-first, RAG-augmented](#3-recommended-architecture-tag-first-rag-augmented)
4. [LLM Selection & LoRA Fine-Tuning](#4-llm-selection--lora-fine-tuning)
5. [Data Layer](#5-data-layer)
6. [Application Layer](#6-application-layer)
7. [Executive Presentation: Chatbot vs Dashboard](#7-executive-presentation-chatbot-vs-dashboard)
8. [Docker / WSL2 Packaging](#8-docker--wsl2-packaging)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [File Structure](#10-file-structure)

---

## 1. Problem Statement

Executives need fast, reliable answers from a relational sales database containing:

| Table          | Contents |
|----------------|----------|
| `leads`        | Prospects, source, status, assigned rep |
| `opportunities`| Pipeline stage, estimated value, probability, close dates |
| `orders`       | Closed revenue, quantities, discount, fulfilment status |
| `activities`   | Interaction log (calls, emails, demos, notes) |

Questions range from precise numerical queries ("What was Q1 revenue by region?") to open-ended summaries ("Why is the APAC pipeline stalling?").

---

## 2. RAG vs TAG – Decision Guide

### Definitions

| Acronym | Full Name | Mechanism |
|---------|-----------|-----------|
| **RAG** | Retrieval-Augmented Generation | Embed text chunks → vector search → inject top-k chunks into LLM prompt |
| **TAG** | Text-to-Aggregation (SQL) Generation | LLM converts natural language → SQL → execute against DB → LLM narrates result |

### When to use RAG

- Source data is **unstructured** (PDF reports, email threads, call transcripts).
- Questions need **semantic similarity** ("find deals similar to the Apex Logistics win").
- You want to surface **free-text notes** or activity summaries.

### When to use TAG (Text-to-SQL)

- Source data is **relational and structured** — this is your situation.
- Questions involve **aggregation, filtering, joins** ("top 5 reps by closed revenue this quarter").
- Answers must be **precise and auditable** — executives verify numbers.
- Avoids hallucination of facts that exist exactly in the DB.

### Verdict for this project

> **Use TAG as the primary engine.**
>
> Your database is entirely relational. SQL can answer >90 % of executive questions exactly.
> Add a RAG layer only for the `notes`, `summary`, and `activities.summary` free-text columns
> so the LLM can synthesise qualitative context alongside the numbers.

**Hybrid pipeline:**
```
User Question
     │
     ▼
[TAG Layer]  LLM → SQL → SQLite → structured result set
     │
     ▼
[RAG Layer]  Embed activity/notes text → vector search → top-k snippets
     │
     ▼
[Synthesis]  LLM combines SQL result + RAG snippets → executive answer
```

---

## 3. Recommended Architecture: TAG-first, RAG-augmented

```
┌─────────────────────────────────────────────────────┐
│                    Frontend                         │
│   Streamlit Chatbot  +  Plotly/Grafana Dashboard    │
└────────────────────┬────────────────────────────────┘
                     │ REST / WebSocket
┌────────────────────▼────────────────────────────────┐
│               FastAPI Backend                       │
│  ┌──────────────┐  ┌───────────────┐                │
│  │  TAG Engine  │  │  RAG Engine   │                │
│  │  (sql_agent) │  │ (vector store)│                │
│  └──────┬───────┘  └───────┬───────┘                │
│         │                  │                        │
│  ┌──────▼──────────────────▼───────┐                │
│  │        LLM Adapter Layer        │                │
│  │  DeepSeek-V3 / LoRA fine-tuned  │                │
│  └─────────────────────────────────┘                │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                        │
┌───────▼───────┐        ┌───────▼───────┐
│  SQLite DB    │        │  ChromaDB /   │
│  sales.db     │        │  FAISS index  │
│ (relational)  │        │ (text embeds) │
└───────────────┘        └───────────────┘
```

---

## 4. LLM Selection & LoRA Fine-Tuning

### Model choices (open-source, self-hostable)

| Model | Size | SQL ability | Notes |
|-------|------|-------------|-------|
| **DeepSeek-V3** | 671B MoE (37B active) | Excellent | Best quality; needs GPU cluster or quantised |
| **DeepSeek-Coder-V2-Lite** | 16B | Excellent | Best SQL/code, fits on a single 24 GB GPU |
| **DeepSeek-R1-Distill-Qwen-7B** | 7B | Good | Reasoning model, runs on 8 GB VRAM |
| **Qwen2.5-Coder-7B** | 7B | Very good | Lightweight alternative |

> **Recommended starting point**: `deepseek-coder-v2-lite` (GGUF Q4_K_M quantised) via **Ollama**
> — runs entirely inside Docker on a machine with 16 GB RAM (CPU-only, slower) or an NVIDIA GPU.

### LoRA Fine-Tuning Strategy

Fine-tuning is valuable when:
- You want the model to know your **exact schema** without schema injection in every prompt.
- You have **domain-specific terminology** (product names, sales stages, internal jargon).
- You need to enforce a strict **SQL output format** or JSON structure.

#### What to fine-tune on

```
Training pairs:
  Input:  "What is total revenue for North America this quarter?"
  Output: SELECT r.name, SUM(o.total_amount) AS revenue
          FROM orders o JOIN regions r ON o.region_id=r.region_id
          WHERE r.name='North America'
            AND strftime('%Y-%m', o.order_date) >= strftime('%Y-%m', 'now', 'start of quarter')
          GROUP BY r.name;
```

Generate ~500-2000 such pairs covering:
- Aggregation queries (SUM, COUNT, AVG)
- Pipeline stage filters
- Rep-level performance
- Time-range filters (QTD, YTD, last N days)
- Ranking (TOP N)

#### LoRA implementation with `peft` + `transformers`

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,               # rank – higher = more capacity, more memory
    lora_alpha=32,      # scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],   # attention layers
    bias="none",
)

model = get_peft_model(base_model, lora_config)
# Train with SFTTrainer on your SQL Q&A dataset
# Merge adapter: model.merge_and_unload()  → save as GGUF for Ollama
```

#### LoRA hyperparameter guide

| Hyperparameter | Recommended | Notes |
|----------------|-------------|-------|
| `r` | 8–16 | Start low; increase if underfitting |
| `lora_alpha` | 2×r | Common heuristic |
| `lora_dropout` | 0.05–0.10 | Regularisation |
| Batch size | 4–8 | With gradient accumulation |
| Learning rate | 2e-4 | With cosine schedule |
| Epochs | 3–5 | Monitor eval loss |

---

## 5. Data Layer

### SQLite Schema Summary

```
regions        → lookup: region names
products       → catalog with unit prices
sales_reps     → rep info, region assignment
leads          → top-of-funnel: company, contact, source, status
opportunities  → pipeline: stage, value, probability, close dates
orders         → closed revenue: qty, price, discount, fulfilment
activities     → interaction log (TEXT – used for RAG)
```

### Vector Index (RAG layer)

Index the following free-text columns into **ChromaDB** (default) or **FAISS**:

```python
# Columns to embed
RAG_SOURCES = [
    ("leads",         "lead_id",        "notes"),
    ("opportunities", "opportunity_id", "notes"),
    ("activities",    "activity_id",    "summary"),
]
```

Use `sentence-transformers/all-MiniLM-L6-v2` (MIT license, 80 MB, CPU-friendly) for embeddings.

---

## 6. Application Layer

### TAG Engine – SQL Agent

```python
# app/agents/sql_agent.py
SYSTEM_PROMPT = """
You are a SQL expert. Given a question and this SQLite schema, 
return ONLY a valid SQLite SELECT query. No explanation.

Schema:
{schema}
"""

def question_to_sql(question: str, llm_client) -> str:
    ...

def execute_and_narrate(question: str, sql: str, results, llm_client) -> str:
    # Second LLM call: turn result rows into plain-English executive summary
    ...
```

### RAG Engine – Context Retrieval

```python
# app/agents/rag_agent.py
def retrieve_context(question: str, top_k: int = 5) -> list[str]:
    # Embed question → cosine search ChromaDB → return top-k text snippets
    ...
```

### Hybrid Orchestrator

```python
# app/orchestrator.py
def answer(question: str) -> dict:
    sql        = question_to_sql(question)
    sql_result = run_sql(sql)
    context    = retrieve_context(question)
    narrative  = synthesise(question, sql_result, context)
    return {"sql": sql, "data": sql_result, "answer": narrative}
```

---

## 7. Executive Presentation: Chatbot vs Dashboard

### Recommendation: **Both, layered**

| Layer | Tool | Audience use case |
|-------|------|-------------------|
| **Chatbot** (primary) | Streamlit `st.chat_message` | Ad-hoc NL questions: "Why did LATAM miss target?" |
| **Dashboard** (secondary) | Streamlit tabs + Plotly | Pre-built KPI views: pipeline, revenue, rep scorecard |

**Why not just a dashboard?**

Dashboards answer *known* questions. Executives frequently ask novel questions that no pre-built chart covers. A chatbot with TAG covers the long tail of questions. The dashboard provides at-a-glance trust ("the numbers I see in chat match what I see here").

**Why not just a chatbot?**

Executives in board meetings need stable, consistently formatted KPI charts — not a chat window.

### Key executive KPIs to surface

1. **Pipeline health**: Total estimated value by stage (funnel chart)
2. **Win rate**: Closed Won / (Closed Won + Closed Lost)
3. **Revenue by region/rep/product**: Bar + heatmap
4. **Lead conversion rate**: Leads → Qualified → Opportunity → Won
5. **Average deal size & sales cycle length**
6. **Forecast accuracy**: Expected close vs actual close

---

## 8. Docker / WSL2 Packaging

### Architecture

```
docker-compose.yml
  ├── app          (FastAPI + Streamlit)
  ├── ollama       (LLM server – DeepSeek model)
  └── chromadb     (vector store)
```

### Key considerations for WSL2

- Map GPU via `--gpus all` (requires NVIDIA Container Toolkit installed in WSL2).
- Use **named volumes** for `sales.db` and ChromaDB index so data persists across restarts.
- Ollama pulls the model on first run (~4-9 GB download); pre-pull in Dockerfile for CI.
- Expose only port `8501` (Streamlit) to the Windows host; internal services communicate on the Docker network.

---

## 9. Implementation Roadmap

### Phase 1 – Foundation (Week 1-2)

- [x] Database schema (`db/schema.sql`)
- [x] Seed script (`db/seed.py`)
- [ ] FastAPI skeleton + health endpoint
- [ ] Ollama integration (pull & query DeepSeek model)
- [ ] Basic TAG: schema-aware prompt → SQL → execute → narrate

### Phase 2 – RAG Layer (Week 3)

- [ ] Embed `activities.summary` + `*.notes` into ChromaDB
- [ ] Hybrid orchestrator (SQL result + RAG context → synthesis)
- [ ] Unit tests for SQL generation accuracy

### Phase 3 – Frontend (Week 4)

- [ ] Streamlit chatbot interface
- [ ] Streamlit dashboard tabs (pipeline funnel, revenue charts, rep scorecard)
- [ ] Chat history with SQL transparency ("Show me the query")

### Phase 4 – Fine-Tuning (Week 5-6)

- [ ] Generate 500+ SQL Q&A training pairs from the seeded DB
- [ ] LoRA fine-tune `deepseek-coder-v2-lite` with `peft` + `trl`
- [ ] Convert to GGUF → load into Ollama as custom model
- [ ] A/B test fine-tuned vs base model on SQL accuracy

### Phase 5 – Production Hardening (Week 7-8)

- [ ] Docker Compose with GPU support, volume mounts, health checks
- [ ] `.env`-based configuration (no hardcoded secrets)
- [ ] SQL injection prevention (parameterised queries, read-only DB user)
- [ ] Rate limiting + auth (basic API key for exec access)
- [ ] Logging + query audit trail

---

## 10. File Structure

```
sales_rag/
├── db/
│   ├── schema.sql          # DDL
│   ├── seed.py             # Sample data generator
│   └── sales.db            # Generated database (gitignored)
├── app/
│   ├── main.py             # FastAPI entrypoint
│   ├── config.py           # Settings (env vars)
│   ├── agents/
│   │   ├── sql_agent.py    # TAG: NL → SQL → narrate
│   │   └── rag_agent.py    # RAG: embed + retrieve context
│   ├── orchestrator.py     # Hybrid TAG+RAG pipeline
│   ├── db.py               # SQLite connection helper
│   └── vector_store.py     # ChromaDB helpers
├── frontend/
│   └── streamlit_app.py    # Chatbot + dashboard UI
├── training/
│   ├── generate_pairs.py   # Build SQL Q&A dataset from DB
│   ├── finetune_lora.py    # LoRA fine-tune script
│   └── data/
│       └── sql_pairs.jsonl # Training data (generated)
├── docker/
│   ├── Dockerfile          # App image
│   └── Dockerfile.ollama   # Optional: pre-pull model
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── PLAN.md                 # This document
```
