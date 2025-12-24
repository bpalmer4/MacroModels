#!/bin/bash
# Run NAIRU + Output Gap analysis (Stage 2 only - loads saved results)
uv run python -m src.models.nairu_output_gap_stage2 \
    --output-dir model_outputs \
    --chart-dir charts/nairu_output_gap \
    "$@"
