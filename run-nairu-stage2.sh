#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run NAIRU + Output Gap analysis (Stages 2 & 3 - loads saved results)
cd "$ROOT"

echo "=== Stage 2: Analysis ==="
uv run python -m src.models.nairu.stage2 "$@"

echo ""
"$ROOT/run-nairu-stage3.sh"
