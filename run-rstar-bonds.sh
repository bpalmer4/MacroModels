#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run the r* model: the natural rate as the permanent component of the real
# yield curve, anchored on published world r*. Three windows on one r* — the
# indexed 10-year, the real cash rate, and a medium maturity in between. No
# IS curve.
#
# The Taylor rule and the output-gap term read a completed ystar_ustar run; r*
# itself does not, so the model still runs without one.
# Run order: ./run-expectations.sh -> ./run-ystar-ustar.sh -> ./run-rstar-bonds.sh
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.rstar_bonds.run "$@"

# Typical usage:
#   ./run-rstar-bonds.sh --verbose
#   ./run-rstar-bonds.sh --analyse-only        # recharts from the saved trace
#   ./run-rstar-bonds.sh --sigma-walk 0.03     # the one imposed setting, swept
#   ./run-rstar-bonds.sh --no-curve            # two windows: drop the belly
#   ./run-rstar-bonds.sh --no-short            # one window: the long yield alone
#   ./run-rstar-bonds.sh --deflator trimmed    # backward-looking real cash rate
#   ./run-rstar-bonds.sh --impose-world-loading  # b_world = 1 rather than estimated
#   ./run-rstar-bonds.sh --no-world            # does the global anchor do the work?
#   ./run-rstar-bonds.sh --world-source US     # the marginal pricer, not the average
#   ./run-rstar-bonds.sh --taylor-ugap         # rule on the unemployment gap instead
