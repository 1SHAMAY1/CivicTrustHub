#!/usr/bin/env bash
# ==============================================================
# CivicTrust Hub – One-click launcher
# Run: bash run.sh
# ==============================================================
set -e

PYTHON="/usr/bin/python3"
UVICORN="$PYTHON -m uvicorn"
PORT=8000

echo "🛡️  CivicTrust Hub – Starting backend..."
echo "➡  API  → http://localhost:$PORT/api/health"
echo "➡  App  → http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Install deps if not present
$PYTHON -m pip install --quiet --user fastapi uvicorn pydantic 2>/dev/null || true

# Start server (auto-opens browser after 2s)
(sleep 2 && open "http://localhost:$PORT") &
$UVICORN main:app --host 0.0.0.0 --port $PORT --reload

