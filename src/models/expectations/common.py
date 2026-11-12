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

# The window over which the survey-only model's mean is set to mean inflation.
# Here rather than in its config so the labels below and the config read one value.
RESET_START = "1996Q1"
RESET_END = "2019Q4"
_RESET_YEARS = f"{RESET_START[:4]}-{RESET_END[:4]}"

MODEL_NAMES = {
    "unanchored": "TREND EXPECTATIONS (Surveys, inflation and wages; no target imposed)",
    "short": "SHORT RUN (1 Year)",
    "market": "LONG RUN (10-Year Bond)",
    "cusbert": f"SURVEY-ONLY EXPECTATIONS (Level set to {_RESET_YEARS} inflation; Cusbert-style)",
}

MODEL_TYPES = ["unanchored", "short", "market", "cusbert"]

# What a reader sees in chart legends. The code names stay as identifiers;
# these are display only, and say what separates the models: their inputs, and
# where their level comes from.
LEGEND_LABELS = {
    "unanchored": "Trend: surveys, inflation, wages",
    "short": "Short run (1 year)",
    "market": "Long run (10-year bond)",
    "cusbert": f"Surveys only, level set to {_RESET_YEARS} inflation",
}
