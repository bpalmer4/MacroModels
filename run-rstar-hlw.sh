#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run HLW Bayesian r-star estimation model
#
# Reads saved output from the expectations model.
# Run order: ./run-expectations.sh -> ./run-rstar-hlw.sh
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.rstar_hlw.run "$@"

# Typical usage:
#   ./run-rstar-hlw.sh                          # Resolution A (canonical HLW: r* = g + z, default)
#   ./run-rstar-hlw.sh -v
#   ./run-rstar-hlw.sh --resolution C           # blend with fixed Beta(1,1) on alpha
#   ./run-rstar-hlw.sh --resolution G           # blend + hierarchical Beta on alpha
#   ./run-rstar-hlw.sh --estimate-only
#   ./run-rstar-hlw.sh --skip-estimate
