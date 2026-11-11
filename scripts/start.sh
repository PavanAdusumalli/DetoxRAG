#!/usr/bin/env bash
# Start DetoxRAG locally: builds the React frontend if needed, then serves
# API + UI from FastAPI at http://localhost:8000
#
#   ./scripts/start.sh            # models load on the first assistant request
#   PRELOAD=1 ./scripts/start.sh  # load models at startup (~5 s)
set -euo pipefail
cd "$(dirname "$0")/.."

[ -x .venv/bin/python ] || { echo "Missing .venv — see README setup."; exit 1; }
[ -f data/synthetic_patients/detoxrag.db ] || .venv/bin/python scripts/generate_synthetic_data.py --csv
[ -f data/processed/chunks.jsonl ] || .venv/bin/python scripts/build_index.py

if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src -newer frontend/dist/index.html -print -quit)" ]; then
  command -v npm >/dev/null || export PATH="/opt/homebrew/bin:$PATH"
  (cd frontend && { [ -d node_modules ] || npm install; } && npm run build)
fi

export DETOX_PRELOAD_MODELS="${PRELOAD:-0}"
echo "DetoxRAG → http://localhost:8000   (API docs: http://localhost:8000/docs)"
exec .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port "${PORT:-8000}"
