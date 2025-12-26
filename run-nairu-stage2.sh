#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run NAIRU + Output Gap analysis (Stage 2 only - loads saved results)
cd "$ROOT"
uv run python -m src.models.nairu_output_gap_stage2 \
    --output-dir model_outputs \
    --chart-dir charts/nairu_output_gap \
    "$@"
