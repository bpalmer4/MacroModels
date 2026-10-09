#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Compare potential growth (g*) across every model that estimates one
#
# Reads the saved runs of ystar (both specs), ystar_ustar and rstar_qpm; any
# missing is skipped, but at least one must exist. Stale ones are re-run
# first unless --no-refresh is given.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.gstar_summary.run "$@"

# Typical usage:
#   ./run-gstar-summary.sh                 # re-run stale models first, then chart
#   ./run-gstar-summary.sh --no-refresh    # read saved runs as they stand
#   ./run-gstar-summary.sh --start 2000Q1
