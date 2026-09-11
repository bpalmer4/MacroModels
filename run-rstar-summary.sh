#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Every r* model on one nominal scale
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.rstar_summary.run "$@"

# Any model whose saved trace was not written TODAY is re-run first, which
# takes minutes and overwrites that model's own outputs and charts.
#
# Typical usage:
#   ./run-rstar-summary.sh                  # refresh anything stale, then chart
#   ./run-rstar-summary.sh --no-refresh     # chart the saved runs as they stand
#   ./run-rstar-summary.sh --start 2000Q1   # shorter window
