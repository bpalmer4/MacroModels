#!/bin/zsh
# u* as a step function over imposed regimes, back to 1959.
# PYTENSOR_FLAGS="cxx=" because PyTensor's -ld64 flag breaks C compilation
# under current Xcode; sampling is numpyro, so nothing is lost.
PYTENSOR_FLAGS="cxx=" uv run python -m src.models.regime_ustar.run "$@"
