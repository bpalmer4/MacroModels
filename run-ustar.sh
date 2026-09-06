#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run the u* model: Okun sets the level from a given output gap,
# the Phillips curve supplies the nominal content.
#
# Reads saved output from the expectations and ystar models.
# Run order: ./run-expectations.sh -> ./run-ystar.sh -> ./run-ustar.sh
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.ustar.run "$@"

# Typical usage:
#   ./run-ustar.sh --verbose
#   ./run-ustar.sh --analyse-only        # recharts from the saved trace
#   ./run-ustar.sh --no-phillips         # Okun only, no expectations dependency
#   ./run-ustar.sh --gap-source actual   # log_gdp - y* instead of the defined gap
#   ./run-ustar.sh --sigma-ustar 0.30    # let u* move more freely
