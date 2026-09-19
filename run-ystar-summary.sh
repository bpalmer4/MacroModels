#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Five specifications of the y* model on one set of charts, all from 1984Q1
# with a phased inflation anchor: measured expectations before 1993Q1, gliding
# to the target across 1993Q1-1998Q4.
#
# The five do not observe the same data, so the fit column scores GDP alone,
# over the quarters every specification fitted. Read it with the sampling gate
# and the descriptive columns, not on its own.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.ystar_summary.run "$@"

# Typical usage:
#   ./run-ystar-summary.sh              # refresh anything stale, then chart
#   ./run-ystar-summary.sh --no-refresh # chart the saved runs as they stand
