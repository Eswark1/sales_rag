#!/bin/bash
set -e

DB_PATH="${DB_PATH:-/data/sales.db}"

# Ensure the data directory exists
mkdir -p "$(dirname "$DB_PATH")"

# Seed database if it doesn't exist yet
if [ ! -f "$DB_PATH" ]; then
    echo "[entrypoint] Seeding database at $DB_PATH …"
    DB_PATH="$DB_PATH" python db/seed.py
fi

# Start FastAPI in background
echo "[entrypoint] Starting FastAPI on :8000 …"
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to be ready before starting Streamlit
echo "[entrypoint] Waiting for FastAPI to be ready …"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "[entrypoint] FastAPI is ready."
        break
    fi
    echo "[entrypoint] Waiting… ($i/30)"
    sleep 2
done

# Start Streamlit
echo "[entrypoint] Starting Streamlit on :8501 …"
streamlit run frontend/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

wait
