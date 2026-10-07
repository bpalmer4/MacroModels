"""The real TWI gap from commodity prices, as a backplane to the cash rate.

Exploratory, not a structural model. The real TWI is regressed on the RBA's
US$ commodity price index, in logs; the residual is how far the dollar sits
above (dearer, tightening) or below (cheaper, easing) the level commodity
prices would explain. It measures the exchange rate's part in monetary
conditions, not the policy stance (see MODEL_NOTES).
"""

import pandas as pd

from src.paths import CHARTS

CHART_DIR = CHARTS / "twi-gap"

# Start of the fit: inflation targeting is in place, and the sample is the one
# the specification was chosen on.
START = pd.Period("1993Q1", "Q")

# Backplane bands, in standard deviations of the gap. Within NEUTRAL_SD is left
# clear (around half the quarters); beyond DARK_SD is shaded dark (around the
# most extreme tenth).
NEUTRAL_SD = 0.75
DARK_SD = 1.6

# Start years for the robustness check on the fit and the cointegration test.
# The US$ commodity index begins in 1982Q3.
ROBUSTNESS_STARTS = (1983, 1986, 1990, 1993, 1996, 1999, 2002)
