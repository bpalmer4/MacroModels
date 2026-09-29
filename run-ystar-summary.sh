#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# The output gap from every live model that estimates y*, on one chart
#
# Reads the saved runs of ystar, ystar_ustar and rstar_qpm. Any whose saved
# trace was not written TODAY is re-run first, which redraws that model's own
# charts; the ystar_ustar run takes a few minutes.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.ystar_summary.run "$@"

# Typical usage:
#   ./run-ystar-summary.sh                 # refresh anything stale, then chart
#   ./run-ystar-summary.sh --no-refresh    # chart the saved runs as they stand
#   ./run-ystar-summary.sh --start 2000Q1
