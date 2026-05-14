# Sales Intelligence Platform

A hybrid **TAG + RAG** sales analytics platform that lets executives ask natural language questions about sales data and receive precise, AI-generated answers combining SQL results with qualitative context.

## Architecture

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

**Services:**
- **FastAPI** – REST API backend (`/ask`, `/schema`, `/health`)
- **Streamlit** – Chat + dashboard frontend
- **Ollama** – Local LLM server (DeepSeek Coder v2 by default)
- **ChromaDB** – Vector store for semantic search over free-text notes

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose v2
- At least **16 GB RAM** and **20 GB free disk** (model download ~9 GB)
- *(Optional)* NVIDIA GPU with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for faster inference

## Quick Start (Docker)

### 1. Clone and configure

```bash
git clone <repo-url>
cd sales_rag
cp .env.example .env
```

Edit `.env` if you want to change the model or paths (defaults work out of the box):

```env
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=deepseek-coder-v2:16b-lite-instruct-q4_K_M
DB_PATH=/data/sales.db
CHROMA_PATH=/data/chroma
APP_SECRET_KEY=change-me-in-production
LOG_LEVEL=INFO
```

### 2. Start the stack

**CPU only:**
```bash
docker compose -f docker/docker-compose.yml up -d
```

**With NVIDIA GPU:**
```bash
docker compose -f docker/docker-compose.yml --profile gpu up -d
```

> **First run:** The `model-pull` service will automatically download the LLM (~9 GB). Monitor progress with:
> ```bash
> docker compose -f docker/docker-compose.yml logs -f model-pull
> ```

### 3. Verify services are up

```bash
docker compose -f docker/docker-compose.yml ps
curl http://localhost:8000/health
```

### 4. Open the UI

| Service       | URL                          |
|---------------|------------------------------|
| Streamlit App | http://localhost:8501        |
| FastAPI Docs  | http://localhost:8000/docs   |
| Ollama API    | http://localhost:11434       |
| ChromaDB API  | http://localhost:8002        |

## Stopping the Stack

```bash
docker compose -f docker/docker-compose.yml down
```

To remove all data volumes (database, model, vector index):
```bash
docker compose -f docker/docker-compose.yml down -v
```

## Running Locally (Without Docker)

### 1. Install dependencies

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

### 2. Start Ollama separately

Install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull deepseek-coder-v2:16b-lite-instruct-q4_K_M
ollama serve
```

### 3. Seed the database

```bash
DB_PATH=./db/sales.db python db/seed.py
```

### 4. Configure environment

Create a `.env` file in the project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-coder-v2:16b-lite-instruct-q4_K_M
DB_PATH=./db/sales.db
CHROMA_PATH=./data/chroma
APP_SECRET_KEY=change-me
LOG_LEVEL=INFO
```

### 5. Start the FastAPI backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the Streamlit frontend

In a separate terminal:

```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

## API Usage

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was total revenue by region last quarter?", "use_rag": true}'
```

**Response:**
```json
{
  "question": "...",
  "sql": "SELECT ...",
  "rows": [...],
  "context": ["[entity:...] activity note..."],
  "answer": "Executive-friendly summary..."
}
```

Set `"use_rag": false` to skip the vector search step and return a SQL-only answer.

### View database schema

```bash
curl http://localhost:8000/schema
```

## Project Structure

```
sales_rag/
├── app/
│   ├── agents/
│   │   ├── rag_agent.py       # Vector search over free-text columns
│   │   └── sql_agent.py       # Natural language → SQL → narration
│   ├── config.py              # Settings (pydantic-settings)
│   ├── db.py                  # Read-only SQLite helpers
│   ├── main.py                # FastAPI entrypoint
│   └── orchestrator.py        # Hybrid TAG + RAG pipeline
├── db/
│   ├── schema.sql             # Table definitions
│   ├── seed.py                # Database seeder
│   └── sales.db               # SQLite database file
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── frontend/
│   └── streamlit_app.py       # Streamlit chat + dashboard
├── training/
│   ├── generate_pairs.py      # Generate fine-tuning data pairs
│   └── finetune_lora.py       # Optional LoRA fine-tuning script
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Optional: LoRA Fine-Tuning

The `training/` directory contains scripts to fine-tune the model on your domain-specific SQL pairs. This requires a GPU host with additional dependencies:

```bash
pip install torch transformers peft trl datasets bitsandbytes accelerate
```

1. Generate training pairs:
   ```bash
   python training/generate_pairs.py
   ```
2. Run fine-tuning:
   ```bash
   python training/finetune_lora.py
   ```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Ollama container is unhealthy | Wait 1-2 minutes for model download to complete; check `docker logs sales_ollama` |
| `connection refused` on port 8000 | App container waits for Ollama health check; retry after Ollama is ready |
| Out of memory errors | Switch to a smaller model by setting `OLLAMA_MODEL` in `.env` |
| ChromaDB index empty | The index is built automatically on first API startup; check `docker logs sales_app` |
