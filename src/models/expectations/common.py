"""Common constants and utilities for expectations models."""

from pathlib import Path

# --- Paths ---

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output" / "expectations"
CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "expectations"

# --- Sampler Settings ---

DEFAULT_DRAWS = 10000
DEFAULT_TUNE = 4000
DEFAULT_CHAINS = 4

# --- Target Anchoring ---

ANCHOR_TARGET = 2.5  # Inflation target (%)
ANCHOR_SIGMA = 0.3   # Observation noise for target anchoring

# --- Model Names ---

MODEL_NAMES = {
    "target": "TARGET ANCHORED",
    "short": "SHORT RUN (1 Year)",
    "market": "LONG RUN (10-Year Bond)",
}

MODEL_TYPES = ["target", "short", "market"]
