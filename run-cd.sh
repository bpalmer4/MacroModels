#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run Cobb-Douglas MFP Decomposition model
cd "$ROOT"
uv run python -m src.models.cobb_douglas.model "$@"
