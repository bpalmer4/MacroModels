#!/bin/bash
# Run the inflation expectations signal extraction model

uv run python -m src.models.expectations.model "$@"
