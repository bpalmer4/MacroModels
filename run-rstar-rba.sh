#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Neutral from the RBA's own reaction to inflation:
#   r_t - b_t = lambda x (pi_t - 2.5) + u_t,  b_t = neutral
# Two gaps assumed proportional. No IS curve, no world anchor, no bond market.
#
# Reads saved output from the expectations model. The Taylor chart also reads a
# completed rstar_bonds run, and is skipped without one. The --employment
# option requires a completed ystar_ustar run.
# Run order: ./run-expectations.sh -> ./run-rstar-rba.sh
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.rstar_rba.run "$@"
