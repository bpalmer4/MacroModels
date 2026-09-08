#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Read u* off the stretches where inflation was flat, back to 1959.
# No likelihood, no priors: a rule, and the sensitivity of the answer to it.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.long_run_ustar.run "$@"

# Typical usage:
#   ./run-long-run-ustar.sh
#   ./run-long-run-ustar.sh --require-flat-u        # both series settled, not just inflation
#   ./run-long-run-ustar.sh --window 12             # a longer stretch must be flat
#   ./run-long-run-ustar.sh --tolerance 1.5         # a looser reading of "flat"
#   ./run-long-run-ustar.sh --smooth 1              # no smoothing: finds nothing before 2002
#   ./run-long-run-ustar.sh --require-target        # post-1993, flat AND near 2.5
