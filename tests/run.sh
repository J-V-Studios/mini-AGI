#!/usr/bin/env bash
# Run efficiency tests with a torch-capable interpreter.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${MINAGI_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for cand in \
      "$ROOT/.venv/bin/python" \
      "$HOME/mini-AGI/.venv/bin/python" \
      "$(command -v python3)"; do
    if [[ -x "$cand" ]] && "$cand" -c "import torch" 2>/dev/null; then
      PY="$cand"
      break
    fi
  done
fi
if [[ -z "${PY:-}" ]]; then
  echo "no python with torch found; set MINAGI_PYTHON" >&2
  exit 1
fi
cd "$ROOT"
exec "$PY" -m unittest discover -s tests -v "$@"
