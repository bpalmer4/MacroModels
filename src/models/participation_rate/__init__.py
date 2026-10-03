"""Test whether the participation rate rises after the RBA raises the cash rate.

Exploratory regressions, not a structural model. The theory of the case is
that a rate rise squeezes household budgets and some households respond by
taking on work. The tests here establish the pattern; they cannot separate
that channel from the RBA hiking into strong labour markets (see MODEL_NOTES).
"""

import pandas as pd

from src.paths import CHARTS

CHART_DIR = CHARTS / "participation-after-hikes"

# The cash rate moves in discrete target steps from here; before, it is a
# market average and every month would register as a "decision".
START = pd.Period("1990-08", "M")

HMAX = 18  # months after a decision traced by the projections and charts
PRE = 12  # months before a decision shown on the event-study charts

# A first move of a cycle has no move in the same direction in the previous
# CYCLE_GAP months.
CYCLE_GAP = 12

# Outcome windows that touch the lockdowns are dropped: participation fell and
# recovered for reasons no rate decision had anything to do with.
COVID = (pd.Period("2020-03", "M"), pd.Period("2021-12", "M"))

# Age comparison of the latest cycle (Original data, so changes are compared
# with the same calendar window in normal years). Lower bound of each band.
AGE_BANDS = {"15-24": 15, "25-54": 25, "55-64": 55, "65+": 65}
BASELINE_START = 2010  # first normal year
# Years whose window is distorted by the lockdowns and the reopening rebound.
BASELINE_EXCLUDE = range(2020, 2023)
