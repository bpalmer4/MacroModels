"""Load inflation expectations from RBA PIE_RBAQ with target phase-in.

Provides RBA survey-based expectations for use in NAIRU Phillips curves,
with the same phase-in arrangement as the model-based expectations.
"""

import pandas as pd

from src.data.dataseries import DataSeries
from src.data.rba_loader import get_inflation_expectations

# --- Phase-in Configuration (matches observations.py) ---

INFLATION_TARGET = 2.5
PHASE_START = pd.Period("1992Q4")  # Last quarter using pure expectations
PHASE_END = pd.Period("1998Q4")    # First quarter using pure target


# --- Public API ---


def get_rba_expectations() -> DataSeries:
    """Load RBA PIE_RBAQ expectations with target phase-in.

    Applies the same phase-in arrangement as model expectations:
    - Pre-1993: Use RBA PIE_RBAQ series
    - 1993-1998: Linear phase from PIE_RBAQ to 2.5% target
    - Post-1998: Fixed at 2.5% target

    Returns:
        DataSeries with quarterly expectations (annual rate, %)

    """
    # Load raw RBA expectations (already annualized)
    rba_raw = get_inflation_expectations().data

    # Extend index to current quarter (post-PHASE_END is all target anyway)
    current_quarter = pd.Timestamp.today().to_period("Q")
    full_index = pd.period_range(rba_raw.index.min(), current_quarter, freq="Q")
    result = pd.Series(index=full_index, dtype=float)

    # Pre-PHASE_START: use RBA raw
    pre_phase = full_index <= PHASE_START
    result.loc[pre_phase] = rba_raw.reindex(full_index[pre_phase])

    # PHASE_START to PHASE_END: linear interpolation
    phase_periods = pd.period_range(PHASE_START, PHASE_END, freq="Q")
    n_periods = len(phase_periods)
    for i, period in enumerate(phase_periods):
        weight = i / (n_periods - 1)
        exp_value = rba_raw.loc[period] if period in rba_raw.index else INFLATION_TARGET
        result.loc[period] = (1 - weight) * exp_value + weight * INFLATION_TARGET

    # Post-PHASE_END: use target
    post_phase = full_index > PHASE_END
    result.loc[post_phase] = INFLATION_TARGET

    return DataSeries(
        data=result,
        source="RBA",
        units="%",
        description="Inflation Expectations (RBA PIE_RBAQ, target-anchored)",
        table="PIE_RBAQ",
        series_id="PIE_RBAQ_anchored",
    )


# --- Testing ---

if __name__ == "__main__":
    print("Loading RBA expectations with target phase-in...")
    print(f"Phase period: {PHASE_START} to {PHASE_END}")

    exp = get_rba_expectations()
    print(f"\nExpectations: {exp.data.first_valid_index()} to {exp.data.last_valid_index()}")
    print(f"Latest: {exp.data.dropna().iloc[-1]:.2f}%")

    # Show phase region
    phase_region = exp.data.loc["1992Q1":"1999Q2"]
    print(f"\nPhase region:")
    print(phase_region.to_string())
