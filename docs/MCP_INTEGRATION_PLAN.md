# MCP Integration Plan – Sales Intelligence Platform

> **Date**: 2026-05-19  
> **Stack context**: FastAPI · Ollama (DeepSeek) · SQLite · ChromaDB · Streamlit  
> **MCP SDK**: [`mcp`](https://github.com/modelcontextprotocol/python-sdk) (Python, by Anthropic)

---

## Table of Contents

1. [What is MCP?](#1-what-is-mcp)
2. [Why Integrate MCP Here?](#2-why-integrate-mcp-here)
3. [Integration Architecture](#3-integration-architecture)
4. [Tools to Expose](#4-tools-to-expose)
5. [File Changes](#5-file-changes)
6. [Implementation Steps](#6-implementation-steps)
7. [Docker & Config Changes](#7-docker--config-changes)
8. [Connecting MCP Clients](#8-connecting-mcp-clients)
9. [Optional: MCP Client Inside the Orchestrator](#9-optional-mcp-client-inside-the-orchestrator)
10. [Testing the MCP Server](#10-testing-the-mcp-server)
11. [Roadmap Summary](#11-roadmap-summary)

---

## 1. What is MCP?

**Model Context Protocol (MCP)** is an open standard by Anthropic that defines a uniform
interface between LLM applications (clients) and external data/tool providers (servers).

```
MCP Client (Claude Desktop, Cursor, custom agent)
        │  JSON-RPC 2.0  (stdio  or  SSE/HTTP)
        ▼
MCP Server  ←  your code: exposes Tools, Resources, Prompts
        │
        ▼
   Your Data  (SQLite, ChromaDB, files, APIs …)
```

An MCP **Tool** is a callable function the LLM can invoke, like a REST endpoint but
described to the LLM via a JSON schema so it can decide when and how to call it.

---

## 2. Why Integrate MCP Here?

| Current limitation | MCP solution |
|--------------------|--------------|
| Sales intelligence locked inside FastAPI / Streamlit | Any MCP-aware LLM client (Claude Desktop, Cursor, custom agent) can query the DB directly |
| Orchestrator hard-codes which agents to call | LLM decides which tools to invoke based on the question |
| No standard way for external tools (web search, CRM APIs) to plug in | MCP client in the orchestrator can call external MCP servers transparently |
| Agentic loop requires custom code | MCP tool-call loop is handled by the client runtime |

**Primary goal**: expose the Sales RAG platform as an MCP server so any MCP-aware
LLM runtime can answer sales questions using the same TAG + RAG pipeline already built.

**Secondary goal** (optional, Phase 2): use an MCP client *inside* the orchestrator to
call external MCP servers (e.g. Brave Search, file system, CRM API) and incorporate
their results into the synthesis step.

---

## 3. Integration Architecture

### Current (direct calls)

```
Streamlit / FastAPI
      │
      ▼
orchestrator.answer()
   ├── sql_agent.question_to_sql()  → Ollama
   ├── db.run_query()               → SQLite
   └── rag_agent.retrieve()         → ChromaDB
```

### After MCP Integration

```
┌─────────────────────────────────────────────────────────────────┐
│  MCP Clients                                                    │
│  Claude Desktop · Cursor · custom agentic scripts              │
└────────────────────────┬────────────────────────────────────────┘
          stdio / SSE    │
┌─────────────────────────▼───────────────────────────────────────┐
│  app/mcp_server.py  (new)                                       │
│  Tools:                                                         │
│    get_schema · run_sql · retrieve_context · ask_sales          │
└──────┬───────────────────────┬──────────────────────────────────┘
       │                       │
       ▼                       ▼
  app/db.py               app/orchestrator.py
  SQLite                  ChromaDB · Ollama
```

FastAPI (`main.py`) and Streamlit stay unchanged — the MCP server is a separate
entry-point that reuses the same `app/` Python package.

---

## 4. Tools to Expose

### Tool 1 – `get_schema`
Returns the full SQLite DDL so the LLM can understand the data model before writing SQL.

```
Input:  (none)
Output: string  – multi-table DDL
```

### Tool 2 – `run_sql`
Executes a read-only SELECT against `sales.db` and returns rows as JSON.

```
Input:  sql (string)  – a valid SQLite SELECT
Output: list[dict]    – result rows
```

Security: re-uses the existing `db.run_query()` guard (SELECT-only, read-only connection).

### Tool 3 – `retrieve_context`
Vector-searches ChromaDB for free-text snippets relevant to a question.

```
Input:  question (string), top_k (int, default 5)
Output: list[string]  – ranked snippets with source metadata
```

### Tool 4 – `ask_sales`
The full TAG + RAG + synthesis pipeline in one tool call.  
Use this when you just want an answer without controlling the sub-steps.

```
Input:  question (string), use_rag (bool, default true)
Output: { sql, rows, context, answer }
```

---

## 5. File Changes

```
sales_rag/
├── app/
│   ├── mcp_server.py        ← NEW: MCP server entry-point
│   ├── main.py              (unchanged)
│   ├── orchestrator.py      (unchanged)
│   ├── agents/              (unchanged)
│   ├── db.py                (unchanged)
│   └── config.py            (minor: add MCP_TRANSPORT env var)
├── requirements.txt         ← add: mcp[cli]>=1.0
├── docker-compose.yml       ← add: mcp service (SSE mode)
├── docker/
│   └── entrypoint.sh        ← optional: start MCP server alongside app
└── .env.example             ← add: MCP_TRANSPORT, MCP_PORT
```

---

## 6. Implementation Steps

### Step 1 – Install the MCP Python SDK

```
mcp[cli]>=1.0.0
```

Add to `requirements.txt`.

---

### Step 2 – Create `app/mcp_server.py`

```python
"""
MCP server – exposes the Sales RAG platform as MCP tools.

Run (stdio, for Claude Desktop / Cursor):
    python -m app.mcp_server

Run (SSE/HTTP, for network clients):
    python -m app.mcp_server --transport sse --port 8010
"""

from mcp.server.fastmcp import FastMCP
from app.db import get_schema, run_query
from app.agents.rag_agent import retrieve, build_index
from app.orchestrator import answer

mcp = FastMCP(
    name="sales-rag",
    instructions=(
        "You have access to a sales intelligence platform with four tools. "
        "Use get_schema first to understand the data model. "
        "Use run_sql for precise aggregations. "
        "Use retrieve_context for qualitative notes and activity logs. "
        "Use ask_sales for a full answer combining SQL + RAG."
    ),
)

# ── Build vector index at import time (no-op if already built) ────────────
build_index()


@mcp.tool()
def get_schema() -> str:
    """Return the full SQLite DDL for the sales database."""
    return get_schema()


@mcp.tool()
def run_sql(sql: str) -> list[dict]:
    """
    Execute a read-only SELECT against the sales SQLite database.
    Only SELECT statements are permitted.
    Returns a list of row dicts.
    """
    return run_query(sql)


@mcp.tool()
def retrieve_context(question: str, top_k: int = 5) -> list[str]:
    """
    Semantic search over activity summaries and deal/lead notes.
    Returns the top_k most relevant text snippets with source metadata.
    """
    return retrieve(question, top_k=top_k)


@mcp.tool()
def ask_sales(question: str, use_rag: bool = True) -> dict:
    """
    Full TAG + RAG pipeline: converts the question to SQL, executes it,
    retrieves relevant context, and synthesises an executive answer.

    Returns: { sql, rows, context, answer }
    """
    return answer(question, use_rag=use_rag)


if __name__ == "__main__":
    import sys
    transport = "stdio"
    port = 8010
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--transport" and i + 1 < len(sys.argv) - 1:
            transport = sys.argv[i + 2]
        if arg == "--port" and i + 1 < len(sys.argv) - 1:
            port = int(sys.argv[i + 2])

    if transport == "sse":
        mcp.run(transport="sse", port=port)
    else:
        mcp.run(transport="stdio")
```

> `FastMCP` (from `mcp.server.fastmcp`) is the high-level decorator-based API —
> same ergonomics as FastAPI route decorators.

---

### Step 3 – Update `app/config.py`

Add two optional settings so transport and port are configurable via `.env`:

```python
class Settings(BaseSettings):
    # … existing fields …
    mcp_transport: str = "stdio"   # "stdio" or "sse"
    mcp_port: int = 8010
```

---

### Step 4 – Update `.env.example`

```dotenv
# MCP Server
MCP_TRANSPORT=sse     # use "stdio" for Claude Desktop / Cursor
MCP_PORT=8010
```

---

## 7. Docker & Config Changes

### Option A – Run MCP in the same `app` container

Update `docker/entrypoint.sh` to launch the MCP server alongside FastAPI and Streamlit:

```sh
#!/bin/bash
set -e

# Seed DB if not present
python db/seed.py

# Start FastAPI (background)
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit (background)
streamlit run frontend/streamlit_app.py --server.port 8501 --server.headless true &

# Start MCP server (SSE transport, foreground)
python -m app.mcp_server --transport sse --port 8010
```

Expose port `8010` in `docker-compose.yml` for the `app` service:

```yaml
ports:
  - "8000:8000"   # FastAPI
  - "8501:8501"   # Streamlit
  - "8010:8010"   # MCP (SSE)
```

### Option B – Dedicated MCP service in Docker Compose

Add a separate service that shares the same image but different command:

```yaml
mcp:
  build:
    context: .
    dockerfile: docker/Dockerfile
  container_name: sales_mcp
  restart: unless-stopped
  depends_on:
    chromadb:
      condition: service_healthy
  volumes:
    - app_data:/data
  ports:
    - "8010:8010"
  env_file: .env
  environment:
    - OLLAMA_BASE_URL=http://host.docker.internal:11434
    - CHROMA_PATH=/data/chroma
    - DB_PATH=/data/sales.db
    - MCP_TRANSPORT=sse
    - MCP_PORT=8010
  extra_hosts:
    - "host.docker.internal:host-gateway"
  command: ["python", "-m", "app.mcp_server", "--transport", "sse", "--port", "8010"]
```

Option B is recommended for cleaner separation and independent scaling.

---

## 8. Connecting MCP Clients

### Claude Desktop (stdio transport)

Edit `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`,
Windows: `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "sales-rag": {
      "command": "python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "D:/sales_rag",
      "env": {
        "DB_PATH": "D:/sales_rag/db/sales.db",
        "CHROMA_PATH": "D:/sales_rag/data/chroma",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

Claude Desktop will spawn the MCP server as a subprocess and communicate over stdio.
The four tools (`get_schema`, `run_sql`, `retrieve_context`, `ask_sales`) will appear
in the Claude tool palette automatically.

### Cursor / VS Code (SSE transport)

With the Docker service running on port `8010`:

```json
{
  "mcp": {
    "servers": {
      "sales-rag": {
        "url": "http://localhost:8010/sse"
      }
    }
  }
}
```

### Custom Python Agent

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def query_sales(question: str):
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "app.mcp_server"],
        cwd="/path/to/sales_rag",
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "ask_sales",
                arguments={"question": question}
            )
            return result.content[0].text
```

---

## 9. Optional: MCP Client Inside the Orchestrator

Instead of (or in addition to) exposing tools, the orchestrator itself can become an
MCP **client** that calls external MCP servers to enrich answers.

Example: add a Brave Search MCP server to pull live market news into the synthesis prompt.

```python
# app/orchestrator.py  (extended)
from mcp import ClientSession
from mcp.client.sse import sse_client

async def fetch_market_context(query: str) -> str:
    """Call the Brave Search MCP server for live context."""
    async with sse_client("http://brave-mcp:3000/sse") as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            res = await session.call_tool("brave_web_search", {"query": query})
            return res.content[0].text
```

This pattern means the orchestrator is itself an **agent** that can call:
- Internal MCP tools: `run_sql`, `retrieve_context` (same server)
- External MCP servers: Brave Search, Slack, Salesforce CRM, etc.

---

## 10. Testing the MCP Server

### Manual test with the MCP CLI

```bash
# Install mcp CLI
pip install "mcp[cli]"

# Inspect tools
mcp dev app/mcp_server.py

# Interactive test in the MCP Inspector UI (opens browser)
# This launches a web UI where you can call tools manually
```

### Automated test (pytest)

```python
# tests/test_mcp.py
import pytest
from mcp.server.fastmcp import FastMCP
from app.mcp_server import mcp  # import the FastMCP instance

def test_get_schema():
    # FastMCP exposes .tools for direct introspection
    tools = {t.name: t for t in mcp.list_tools()}
    assert "get_schema" in tools
    assert "run_sql" in tools
    assert "retrieve_context" in tools
    assert "ask_sales" in tools

@pytest.mark.asyncio
async def test_run_sql():
    result = await mcp.call_tool("run_sql", {"sql": "SELECT 1 AS val"})
    assert result == [{"val": 1}]
```

---

## 11. Roadmap Summary

### Phase 1 – Server scaffold (1–2 days)
- [ ] Add `mcp[cli]>=1.0.0` to `requirements.txt`
- [ ] Create `app/mcp_server.py` with four tools (Step 2 above)
- [ ] Add `MCP_TRANSPORT` / `MCP_PORT` to `config.py` and `.env.example`
- [ ] Smoke-test with `mcp dev app/mcp_server.py`

### Phase 2 – Docker integration (1 day)
- [ ] Choose Option A (shared container) or Option B (dedicated service)
- [ ] Update `docker-compose.yml` and `entrypoint.sh` accordingly
- [ ] Verify SSE endpoint at `http://localhost:8010/sse`

### Phase 3 – Client connections (1 day)
- [ ] Connect Claude Desktop via stdio config
- [ ] Connect Cursor via SSE config
- [ ] Demonstrate: ask Claude Desktop "What is total revenue by region?" with no Streamlit

### Phase 4 – External MCP client (optional, 2–3 days)
- [ ] Add Brave Search or another external MCP server to `docker-compose.yml`
- [ ] Extend `orchestrator.answer()` to optionally call external tools via MCP client
- [ ] Surface external context in the synthesis prompt

### Phase 5 – Hardening (1 day)
- [ ] Auth: add `BEARER_TOKEN` check in SSE transport (MCP supports OAuth 2.0)
- [ ] Rate-limit tool calls (re-use FastAPI middleware pattern)
- [ ] Add structured logging for every tool invocation (tool name, latency, token usage)
- [ ] Add `tests/test_mcp.py` to CI

---

## Key Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Transport for local dev | `stdio` | Zero network config; Claude Desktop / Cursor natively support it |
| Transport for Docker / remote | `sse` | Works over HTTP; compatible with any network-accessible client |
| Tool granularity | 4 tools (coarse + fine) | `ask_sales` for simple use, individual tools for agentic loops that need control |
| Server location | Separate Docker service | Cleaner ops; can scale or restart independently of FastAPI |
| Auth | Bearer token on SSE | stdio is inherently local; SSE needs a token to prevent open access |
