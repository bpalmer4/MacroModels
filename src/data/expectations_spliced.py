"""Measured inflation expectations back to 1970, PIE_RBAQ spliced to the model series.

The expectations model cannot start before 1983Q1. PIE_RBAQ runs from 1970Q1
to 2019Q1, so splicing the two covers 1970-2026.

PIE_RBAQ is MARTIN's inflation expectations variable, from the MacroDave
database (github.com/MacroDave/MARTIN), held as a static CSV in `input_data/`.
It is a LONG-RUN ANCHOR, built after Cusbert (2017) as a random walk in trend
inflation, so it lags a fast climb: it sits 6.52 below year-ended headline
across 1974-79 and 0.97 above it across 1983-92. Anything reading it as a
near-term expectation inherits a surprise of +5 to +11 through the 1970s.
UNVERIFIED: how its pre-1983 values were produced.

**The two series agree on shape and disagree on level.** Over the 145
overlapping quarters the correlation is 0.990, but PIE_RBAQ sits above the
model series by +0.87 on average across 1983-87, +0.72 across 1988-92, +0.15
across 1993-98 and +0.01 across 2010-19. The gap closes as inflation falls, so
it is neither a constant bias nor a constant ratio, and no single offset is
right.

**Which is what the offset window is for.** Spliced raw, the join drops 9.93
(1982Q4) to 7.97 (1983Q1), and since PIE_RBAQ's own move that quarter is
-0.92, that leaves an excess step of -1.04. Removing a constant offset fitted
on the join window kills the step but transfers the 1980s-sized gap to the
whole 1970-82 stretch, putting 1970Q1 at 0.5-0.8 against raw PIE_RBAQ's 1.84.
Sweep it; 0 gives the raw splice.
"""
import pandas as pd

from src.data.dataseries import DataSeries
from src.data.expectations_model import get_model_expectations
from src.data.rba_loader import get_inflation_expectations

# First quarter taken from the model series. PIE_RBAQ covers everything before.
SPLICE_AT = pd.Period("1983Q1", freq="Q")

# Quarters of overlap the level offset is measured on, counting from SPLICE_AT.
# 8 leaves a residual step of 0.24 at the join, inside the 0.44 sd of the
# quarter-to-quarter difference between the two series, and is steadier than
# the single-quarter offset that makes the join exactly continuous.
DEFAULT_OFFSET_QUARTERS = 8


def get_spliced_expectations(offset_quarters: int = DEFAULT_OFFSET_QUARTERS) -> DataSeries:
    """Return expectations from 1970Q1: PIE_RBAQ before 1983Q1, the model series after.

    Args:
        offset_quarters: quarters of overlap used to measure the level offset
            removed from PIE_RBAQ. 0 splices the two raw, keeping PIE_RBAQ's
            own level and accepting the step at the join.

    """
    if offset_quarters < 0:
        raise ValueError(f"offset_quarters must be non-negative, got {offset_quarters}")

    rba = get_inflation_expectations().data.astype(float)
    model = get_model_expectations().data.astype(float)

    offset = 0.0
    if offset_quarters:
        window = model.loc[SPLICE_AT:].index[:offset_quarters]
        offset = float((rba.reindex(window) - model.reindex(window)).mean())

    early = rba.loc[rba.index < SPLICE_AT] - offset
    spliced = pd.concat([early, model.loc[model.index >= SPLICE_AT]]).sort_index()

    return DataSeries(
        data=spliced,
        source="RBA",
        units="%",
        description=(
            f"Inflation expectations, PIE_RBAQ less {offset:.2f} before {SPLICE_AT}, "
            "expectations model after"
        ),
        table="PIE_RBAQ",
        series_id="PIE_RBAQ_spliced",
        metadata={"offset": offset, "offset_quarters": offset_quarters, "splice_at": str(SPLICE_AT)},
    )
