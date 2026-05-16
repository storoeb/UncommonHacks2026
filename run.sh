#!/usr/bin/env bash
# ProphetHacks 2026 — agent run script.
# Starts the FastAPI agent on :8000. Installs deps first.
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

# Use any active venv first; otherwise create one.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null || true
fi

pip install -q -r requirements.txt

exec uvicorn prophet_agent.server:app --host 0.0.0.0 --port 8000
