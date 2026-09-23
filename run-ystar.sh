#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run the unobserved-components potential output model
# (default spec: inflation — potential output from GDP and inflation alone)
#
# Reads no other model's output by default. The --anchor-phase option requires
# a completed ./run-expectations.sh.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.ystar.run "$@"

# Typical usage:
#   ./run-ystar.sh                       # default: 1993Q1+, anchor 2.5, sigma_c 0.6
#   ./run-ystar.sh --verbose
#   ./run-ystar.sh --analyse-only        # recharts from the saved trace
#   ./run-ystar.sh --no-analyse          # estimate without charting
#   ./run-ystar.sh --ratio-g 0.05        # looser trend growth
#   ./run-ystar.sh --ratio-ystar 0       # HP's own trend process: no level shock
#   ./run-ystar.sh --supply-control import_prices
#   ./run-ystar.sh --spec labour --ratio-g-lp 0.05
#   ./run-ystar.sh --sigma-c 0.8         # wider cycle, hence smoother trends
#
# The variance settings are imposed, not estimated. To see how much the answer
# depends on them, and on the inflation anchor:
#   uv run python -m src.models.ystar.sigma_sweep --param ratio_g
#   uv run python -m src.models.ystar.sigma_sweep --param ratio_ystar
#   uv run python -m src.models.ystar.sigma_sweep --param anchor
#
# Endpoint fragility — the estimate a real-time user would have had:
#   uv run python -m src.models.ystar.realtime
