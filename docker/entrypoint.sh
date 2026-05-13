#!/bin/bash
set -e

DB_PATH="${DB_PATH:-/data/sales.db}"

# Seed database if it doesn't exist yet
if [ ! -f "$DB_PATH" ]; then
    echo "[entrypoint] Seeding database at $DB_PATH …"
    DB_PATH="$DB_PATH" python db/seed.py
fi

# Start FastAPI in background
echo "[entrypoint] Starting FastAPI on :8000 …"
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit
echo "[entrypoint] Starting Streamlit on :8501 …"
streamlit run frontend/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

wait
