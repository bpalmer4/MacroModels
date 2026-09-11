#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Neutral from the RBA's own reaction to inflation:
#   r_t - b_t = lambda x (pi_t - 2.5) + u_t,  b_t = neutral
# Two gaps assumed proportional. No IS curve, no world anchor, no bond market.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.rstar_rba.run "$@"
