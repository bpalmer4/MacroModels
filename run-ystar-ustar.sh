#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Estimate y* and u* jointly, with the output gap partly free:
#     gap_t = c x (pi - 2.5) + v_t
#
# The point of joining is sigma_v. Inside ystar alone, v and the GDP residual
# are one additive term and cannot be separated; adding the Okun equation puts
# v in a second place, so the covariance between the two residuals identifies
# it. That number says how much of what ystar books as GDP noise is cycle that
# unemployment can see.
#
# Reads saved output from the expectations model. Does NOT read ystar or ustar:
# it re-estimates both.
# Run order: ./run-expectations.sh -> ./run-ystar-ustar.sh
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.ystar_ustar.run "$@"

# Typical usage:
#   ./run-ystar-ustar.sh --verbose
#   ./run-ystar-ustar.sh --analyse-only        # recharts from the saved trace
#   ./run-ystar-ustar.sh --no-okun             # control: sigma_v should return its prior
#   ./run-ystar-ustar.sh --no-phillips         # sigma_v with inflation out of the LHS
#   ./run-ystar-ustar.sh --no-free-gap         # ystar's identity, jointly estimated
#   ./run-ystar-ustar.sh --sigma-v-prior 0.5   # is sigma_v prior-driven?
#   ./run-ystar-ustar.sh --gap-pi-basis quarterly   # the higher-contamination basis
