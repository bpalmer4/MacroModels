#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run the components (expenditure-identity) GDP nowcast
cd "$ROOT"
uv run python -m src.models.gdp_nowcast_components.model "$@"
