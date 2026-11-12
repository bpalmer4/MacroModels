#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run the u* model: ONE expectations-augmented Phillips curve sets u*, which is
# a random walk. The Okun equation is OFF by default (config.include_okun), because
# the default gap is a fixed multiple of inflation's deviation from target and the
# two equations would then read one signal twice; --okun records what adding it
# does to the level and the band.
#
# Reads saved output from the expectations and ystar models. The ystar gap is
# loaded under the defaults even though no default equation uses it, so a
# completed ystar run is required either way.
# Run order: ./run-expectations.sh -> ./run-ystar.sh -> ./run-ustar.sh
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.ustar.run "$@"

# Typical usage:
#   ./run-ustar.sh --verbose
#   ./run-ustar.sh --analyse-only        # recharts from the saved trace
#   ./run-ustar.sh --okun --no-phillips  # Okun only, no expectations dependency
#                                        #   (--no-phillips alone leaves no equation
#                                        #    at all, so u* returns its random walk)
#   ./run-ustar.sh --okun                # add Okun back beside the Phillips curve
#   ./run-ustar.sh --no-inflation-lag    # drop last quarter's inflation from the Phillips curve
#   ./run-ustar.sh --gap-source actual   # log_gdp - y* instead of the defined gap
#   ./run-ustar.sh --sigma-ustar 0.30    # let u* move more freely
