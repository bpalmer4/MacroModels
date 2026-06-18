#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run NAIRU + Output Gap joint estimation model
cd "$ROOT"
uv run python -m src.models.nairu.run "$@"

# Default: --variant simple_excess_rstar_blend --anchor target
#   (simple_excess with the fixed 35/65 growth/yield r* blend)
# Typical usage:
#   ./run-nairu.sh --variant complex
#   ./run-nairu.sh --variant simple complex
#   ./run-nairu.sh --skip-estimate --variant complex
#   ./run-nairu.sh --estimate-only --variant simple_excess
