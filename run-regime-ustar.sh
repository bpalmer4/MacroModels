#!/bin/zsh
# u* as a step function over imposed regimes, back to 1959.
# PYTENSOR_FLAGS="cxx=" because PyTensor's -ld64 flag breaks C compilation
# under current Xcode; sampling is numpyro, so nothing is lost.
# Reads saved output from the expectations model.
# Run order: ./run-expectations.sh -> ./run-regime-ustar.sh
PYTENSOR_FLAGS="cxx=" uv run python -m src.models.regime_ustar.run "$@"
