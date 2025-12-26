#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Run NAIRU + Output Gap joint estimation model
cd "$ROOT"
uv run python -m src.models.nairu_output_gap "$@"
