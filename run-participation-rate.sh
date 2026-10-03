#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Does the participation rate rise after RBA rate rises? Event study and local
# projections; prints the tests and writes charts. Exploratory, not a model.
cd "$ROOT"
source "$ROOT/ssl-env.sh"
uv run python -m src.models.participation_rate.run "$@"
