#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# r* by conditional inversion of an IMPOSED IS curve
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.rstar_invert.run "$@"

# Reads saved output from the expectations model and a completed ystar_ustar
# run, which supplies the output gap, as data.
# Run order: ./run-expectations.sh -> ./run-ystar-ustar.sh -> ./run-rstar-invert.sh
#
# Typical usage:
#   ./run-rstar-invert.sh                       # the deliverable: one conditional r* path
#   ./run-rstar-invert.sh --a-r -0.10           # condition on a weaker rate channel
#   ./run-rstar-invert.sh --ensemble            # the diagnostic grid (24 cells, ~10 min)
#   ./run-rstar-invert.sh --rate-lags 4,5,6     # where transmission should live
#   ./run-rstar-invert.sh --free-sigma-r        # estimate the smoothness (expect the prior back)
#   ./run-rstar-invert.sh --exclude-qe          # also drop 2008Q4-2021Q3
#   ./run-rstar-invert.sh --skip-estimate       # re-chart a saved run
