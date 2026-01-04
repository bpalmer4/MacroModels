#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run NAIRU + Output Gap scenario analysis (Stage 3 only - loads saved results)
cd "$ROOT"

echo "=== Stage 3a: Deterministic Scenarios ==="
uv run python -m src.models.nairu.stage3

echo ""
echo "=== Stage 3b: Monte Carlo Forward Sampling ==="
uv run python -m src.models.nairu.stage3_forward_sampling
