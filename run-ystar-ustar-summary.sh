#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Eight specifications of the joint y*/u* model on one set of charts, crossing
# the structure imposed on u* with the definition of the output gap:
#   decay u*, or a spline with 1, 2 or 3 knots
#   x  inflation-defined gap, or gap = y - y*
#
# Ranked on leave-one-out accuracy over the two equations all eight observe,
# with the sampling diagnostics as a gate and the Phillips-implied residual as
# the check that no structure is pulling u* off where inflation wants it.
#
# Re-estimates any specification whose saved trace is not from today. Each
# writes to its own yus_sum_* prefix, so nothing the model owns is touched.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.ystar_ustar_summary.run "$@"

# Typical usage:
#   ./run-ystar-ustar-summary.sh              # refresh anything stale, then chart
#   ./run-ystar-ustar-summary.sh --no-refresh # chart the saved runs as they stand
