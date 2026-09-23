#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Trend r* and short-run neutral from a semi-structural open-economy model:
#   IS curve, real exchange rate, Phillips curve, policy rule and the 5y5y forward,
#   states integrated out by Kalman filter, NUTS on the parameters.
#   --recovery re-estimates simulated economies at known parameters.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.rstar_qpm.run "$@"
