#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Bank funding and lending rates against the cash rate. Descriptive, no model:
# it exists because rstar measures the policy stance on the cash rate, and after
# the GFC the cash rate stopped summarising the price of credit.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.bank_costs.run "$@"
