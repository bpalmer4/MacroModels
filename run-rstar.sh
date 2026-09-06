#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run the r* model: the natural rate as the permanent component of the real
# bond yield, anchored on published world r*. No IS curve.
#
# The Taylor rule and the output-gap term read a completed ystar run; r*
# itself does not, so the model still runs without one.
# Run order: ./run-expectations.sh -> ./run-ystar.sh -> ./run-rstar.sh
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.rstar.run "$@"

# Typical usage:
#   ./run-rstar.sh --verbose
#   ./run-rstar.sh --analyse-only        # recharts from the saved trace
#   ./run-rstar.sh --sigma-r 0.05        # the setting the answer hinges on
#   ./run-rstar.sh --no-world            # does the global anchor do the work?
#   ./run-rstar.sh --world-source US     # the marginal pricer, not the average
#   ./run-rstar.sh --taylor-ugap         # rule on the unemployment gap instead
