#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Six specifications of the u* model on one chart: the state law (decay, spline
# with one knot, spline with two) crossed with whether the Okun equation is in.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.ustar_summary.run "$@"

# Each specification writes to its own prefix, so a refresh never touches
# ustar's own default outputs or charts. About 20 seconds per stale run.
#
# Typical usage:
#   ./run-ustar-summary.sh                # refresh anything stale, then chart
#   ./run-ustar-summary.sh --no-refresh   # chart the saved runs as they stand
