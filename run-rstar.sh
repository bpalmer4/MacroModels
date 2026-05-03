#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run HLW Bayesian r-star estimation model
cd "$ROOT"
uv run python -m src.models.rstar.run "$@"

# Typical usage:
#   ./run-rstar.sh                          # Resolution C (blend, default)
#   ./run-rstar.sh -v
#   ./run-rstar.sh --resolution A           # canonical HLW: r* = g + z
#   ./run-rstar.sh --estimate-only
#   ./run-rstar.sh --skip-estimate
