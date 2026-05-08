#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run HLW Bayesian r-star estimation model
cd "$ROOT"
uv run python -m src.models.rstar_hlw.run "$@"

# Typical usage:
#   ./run-rstar-hlw.sh                          # Resolution C (blend, default)
#   ./run-rstar-hlw.sh -v
#   ./run-rstar-hlw.sh --resolution A           # canonical HLW: r* = g + z
#   ./run-rstar-hlw.sh --estimate-only
#   ./run-rstar-hlw.sh --skip-estimate
