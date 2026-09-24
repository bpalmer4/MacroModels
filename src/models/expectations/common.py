"""Common constants and utilities for expectations models."""

from src.paths import CHARTS, OUTPUT

# --- Paths ---

OUTPUT_DIR = OUTPUT / "expectations"
CHART_DIR = CHARTS / "expectations"

# --- Sampler Settings ---

DEFAULT_DRAWS = 10000
DEFAULT_TUNE = 4000
DEFAULT_CHAINS = 4

# --- Model Names ---

MODEL_NAMES = {
    "unanchored": "EXPECTATIONS (All Surveys)",
    "short": "SHORT RUN (1 Year)",
    "market": "LONG RUN (10-Year Bond)",
}

MODEL_TYPES = ["unanchored", "short", "market"]
