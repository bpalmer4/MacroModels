#!/bin/zsh
# u* read off the inverted Phillips curve, then smoothed. Nothing estimated.
# Needs a saved regime_ustar trace for beta, unless --beta is given.
PYTENSOR_FLAGS="cxx=" uv run python -m src.models.regime_ustar.inversion_run "$@"
