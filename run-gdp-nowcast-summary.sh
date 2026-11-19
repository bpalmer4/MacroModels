#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Chart every GDP nowcast's distribution for the target quarter, Q/Q and annual.
# Runs all four nowcasts afresh; the components nowcast is left off while any
# of its components is unpublished.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.gdp_nowcast_summary.run "$@"

# Typical usage:
#   ./run-gdp-nowcast-summary.sh       # each model prints its own summary too
#   ./run-gdp-nowcast-summary.sh -q    # just the summary table
