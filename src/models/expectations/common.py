"""Common constants and utilities for expectations models."""

from src.paths import CHARTS, OUTPUT

# --- Paths ---

OUTPUT_DIR = OUTPUT / "expectations"
CHART_DIR = CHARTS / "expectations"

# --- Sampler Settings ---

DEFAULT_DRAWS = 10000
DEFAULT_TUNE = 4000
DEFAULT_CHAINS = 4

# --- Target Anchoring ---

ANCHOR_TARGET = 2.5  # Inflation target (%)
ANCHOR_SIGMA = 0.35  # Observation noise for target anchoring

# --- Model Names ---

MODEL_NAMES = {
    "target": "TARGET ANCHORED",
    "unanchored": "EXPECTATIONS (All Surveys)",
    "short": "SHORT RUN (1 Year)",
    "market": "LONG RUN (10-Year Bond)",
}

MODEL_TYPES = ["target", "unanchored", "short", "market"]
