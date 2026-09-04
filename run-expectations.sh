#!/bin/bash
# Run the inflation expectations signal extraction model

ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/ssl-env.sh"

uv run python -m src.models.expectations.model "$@"
