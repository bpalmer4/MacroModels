#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Six u* specifications from two models on one chart
#
# Three from ustar and three from ystar_ustar, each a specification from that
# model's --compare. Any whose saved trace was not written TODAY is re-run
# first, which redraws that specification's own charts; the ystar_ustar runs
# take a few minutes each.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.ustar_summary.run "$@"

# Typical usage:
#   ./run-ustar-summary.sh                 # refresh anything stale, then chart
#   ./run-ustar-summary.sh --no-refresh    # chart the saved runs as they stand
#   ./run-ustar-summary.sh --start 2000Q1
