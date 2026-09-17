#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# r* from a time-varying parameter VAR with stochastic volatility, after
# Lubik and Matthes at the Richmond Fed. Three variables (inflation, GDP growth,
# the real cash rate), coefficients that drift as random walks, and volatilities
# that drift too. r* is the model's H-quarter-ahead projection of the real
# policy rate.
#
# This is the only r* model in the repo that needs NEITHER an IS curve NOR a
# term premium, which is what makes it an independent reading rather than a
# fourth variation on the same conditioning.
#
# Needs a completed ./run-expectations.sh for the deflator.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.rstar_tvpvar.run "$@"

# STRIPPED BACK TO THE CANONICAL SPEC ON 2026-09-17. The default run is now the
# published Lubik-Matthes model: three variables, quarterly annualised changes,
# r* = the 20-quarter-ahead projection, and NO inflation conditioning. The
# annual basis, the steady-state and 5y5y-window definitions, and the anchored
# projection are all still here, but they are now things you ask for.
#
# The first thing to read in the output is "Does the coefficient drift do
# anything?", which puts r* beside a constant-coefficient VAR run through the
# same projection. If that line sits inside the credible band, the time
# variation is not doing visible work and the answer is not a TVP-VAR's.
#
# Typical usage:
#   ./run-rstar-tvpvar.sh --smoke             # does it build and sample at all
#   ./run-rstar-tvpvar.sh --verbose
#   ./run-rstar-tvpvar.sh --analyse-only      # recharts from the saved trace
#   ./run-rstar-tvpvar.sh --sigma-q 0.005     # THE sweep: how much drift is allowed
#   ./run-rstar-tvpvar.sh --sigma-q 0.05      #   the answer is decided here
#   ./run-rstar-tvpvar.sh --horizon 40        # 10 years rather than 5
#   ./run-rstar-tvpvar.sh --lags 1            # fewer coefficients to drift
#   ./run-rstar-tvpvar.sh --basis annual      # quieter, at the cost of an MA(3)
#   ./run-rstar-tvpvar.sh --training-sample 0 # centre theta_0 on the FULL-sample OLS
#   ./run-rstar-tvpvar.sh --exclude-covid     # does SV actually absorb the spike?
#
# To get the pre-strip-back model back, all four switches together:
#   ./run-rstar-tvpvar.sh --basis annual --rstar-definition forward \
#       --anchor-projection --training-sample 0
