#!/usr/bin/env bash
# ==============================================================
# CivicTrust Hub – One-click cross-platform launcher
# Run: bash run.sh
# ==============================================================
set -e

# Detect available Python interpreter
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "❌ Error: Python not found in PATH. Please install Python 3.10+."
    exit 1
fi

PORT=${PORT:-8000}

echo "🛡️  CivicTrust Hub – Starting backend..."
echo "➡  API  → http://localhost:$PORT/api/health"
echo "➡  App  → http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Install dependencies if needed
$PYTHON -m pip install --quiet -r requirements.txt 2>/dev/null || true

# Cross-platform browser opener
open_browser() {
    sleep 2
    local url="http://localhost:$PORT"
    if command -v xdg-open &>/dev/null; then
        xdg-open "$url" &>/dev/null
    elif command -v open &>/dev/null; then
        open "$url" &>/dev/null
    elif command -v start &>/dev/null; then
        start "$url" &>/dev/null
    fi
}
open_browser &

$PYTHON -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
