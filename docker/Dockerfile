# ── Stage 1: dependency install ────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/pkg -r requirements.txt

# ── Stage 2: runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="sales-rag"

# System deps for sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages
COPY --from=deps /pkg /usr/local

# Copy application source
COPY app/       ./app/
COPY frontend/  ./frontend/
COPY db/        ./db/
COPY training/  ./training/

# Persistent data volumes (DB + vector index)
VOLUME ["/data"]

EXPOSE 8000 8501

# Entrypoint: seed DB if not present, then start API + Streamlit
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
