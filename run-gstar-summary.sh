#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Compare potential growth (g*) across every model that estimates one
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.gstar_summary.run "$@"

# Typical usage:
#   ./run-gstar-summary.sh                 # read saved runs as they stand
#   ./run-gstar-summary.sh --refresh       # re-run stale models first
#   ./run-gstar-summary.sh --start 2000Q1
