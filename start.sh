#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/2] Starting Docker services (Virtuoso + Qdrant)..."
docker compose up -d

echo "[2/2] Launching Streamlit frontend..."
if command -v uv >/dev/null 2>&1; then
    exec uv run ama-kbqa-frontend "$@"
else
    exec ama-kbqa-frontend "$@"
fi
