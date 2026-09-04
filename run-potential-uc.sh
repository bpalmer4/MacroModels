#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run the unobserved-components potential output model
# (default spec: core — potential output from GDP and inflation alone)
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.potential_uc.run "$@"

# Typical usage:
#   ./run-potential-uc.sh                       # default: 1993Q1+, anchor 2.5, sigma_c 0.6
#   ./run-potential-uc.sh --verbose
#   ./run-potential-uc.sh --analyse-only        # recharts from the saved trace
#   ./run-potential-uc.sh --no-analyse          # estimate without charting
#   ./run-potential-uc.sh --ratio-g 0.05        # looser trend growth
#   ./run-potential-uc.sh --ratio-ystar 0       # HP's own trend process: no level shock
#   ./run-potential-uc.sh --supply-control import_prices
#   ./run-potential-uc.sh --spec labour --ratio-g-lp 0.05
#   ./run-potential-uc.sh --sigma-c 0.8         # wider cycle, hence smoother trends
#
# The variance settings are imposed, not estimated. To see how much the answer
# depends on them, and on the inflation anchor:
#   uv run python -m src.models.potential_uc.sigma_sweep --param ratio_g
#   uv run python -m src.models.potential_uc.sigma_sweep --param ratio_ystar
#   uv run python -m src.models.potential_uc.sigma_sweep --param anchor
#
# Endpoint fragility — the estimate a real-time user would have had:
#   uv run python -m src.models.potential_uc.realtime
