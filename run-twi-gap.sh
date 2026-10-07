#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# The real TWI gap from commodity prices, shaded behind the cash rate: is the
# dollar adding to or offsetting the policy rate? Exploratory, not a model.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.twi_gap.run "$@"
