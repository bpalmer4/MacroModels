#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run GDP nowcast DFM model
cd "$ROOT"
uv run python -m src.models.gdp_nowcast_dfm.model "$@"
