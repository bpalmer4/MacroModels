#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run NAIRU + Output Gap joint estimation model
cd "$ROOT"
uv run python -m src.models.nairu.run "$@"

# Default: --variant simple_excess_regime --anchor target
# Typical usage:
#   ./run-nairu.sh --variant complex
#   ./run-nairu.sh --variant simple complex
#   ./run-nairu.sh --skip-estimate --variant complex
#   ./run-nairu.sh --estimate-only --variant simple_excess
