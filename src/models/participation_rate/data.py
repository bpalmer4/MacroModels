"""The monthly frame: participation, labour market controls and rate decisions."""

import numpy as np
import pandas as pd

from src.data.abs_loader import load_series
from src.data.cash_rate import get_cash_rate_monthly
from src.data.series_specs import (
    EMPLOYMENT_PERSONS,
    PARTICIPATION_RATE,
    PARTICIPATION_RATE_FEMALES,
    PARTICIPATION_RATE_MALES,
    UNEMPLOYMENT_RATE,
)
from src.models.participation_rate import CYCLE_GAP, START

# Smallest cash rate move counted as a decision (pp). Every move since 1990 is
# a multiple of 0.1, so half of that separates a move from no move.
MIN_MOVE = 0.05

# How far back the lagged controls reach, so the first sample month has them.
LAG_REACH = 24

# Pre-decision controls, every one dated t-1 or earlier.
CONTROLS = ["dpr12_lag", "dpr3_lag", "dur12_lag", "dur3_lag", "demp12_lag", "ur_lag", "dcr12_lag"]


def _monthly(series: pd.Series) -> pd.Series:
    """Coerce to a monthly PeriodIndex."""
    if not isinstance(series.index, pd.PeriodIndex):
        series.index = pd.PeriodIndex(series.index, freq="M")
    return series


def _first_of_cycle(moves: pd.Series) -> pd.Series:
    """1 where a move has no move the same way in the previous CYCLE_GAP months."""
    prior = moves.shift(1).rolling(CYCLE_GAP, min_periods=CYCLE_GAP).sum()
    return ((moves == 1) & (prior == 0)).astype(float)


def build_frame() -> pd.DataFrame:
    """Return the monthly frame from LAG_REACH months before START.

    Columns: pr, pr_m, pr_f (participation, %), ur (unemployment rate, %),
    emp (employment), cr (cash rate, %), hike, cut, first_hike, first_cut
    (0/1), and the CONTROLS.
    """
    d = pd.DataFrame(
        {
            "pr": _monthly(load_series(PARTICIPATION_RATE).data),
            "pr_m": _monthly(load_series(PARTICIPATION_RATE_MALES).data),
            "pr_f": _monthly(load_series(PARTICIPATION_RATE_FEMALES).data),
            "ur": _monthly(load_series(UNEMPLOYMENT_RATE).data),
            "emp": _monthly(load_series(EMPLOYMENT_PERSONS).data),
            "cr": _monthly(get_cash_rate_monthly().data),
        }
    ).sort_index()
    d = d.loc[d.index >= START - LAG_REACH].dropna(subset=["pr", "cr"])

    dcr = d["cr"].diff()
    d["hike"] = (dcr > MIN_MOVE).astype(float)
    d["cut"] = (dcr < -MIN_MOVE).astype(float)
    d["first_hike"] = _first_of_cycle(d["hike"])
    d["first_cut"] = _first_of_cycle(d["cut"])

    d["dpr12_lag"] = d["pr"].shift(1) - d["pr"].shift(13)
    d["dpr3_lag"] = d["pr"].shift(1) - d["pr"].shift(4)
    d["dur12_lag"] = d["ur"].shift(1) - d["ur"].shift(13)
    d["dur3_lag"] = d["ur"].shift(1) - d["ur"].shift(4)
    d["demp12_lag"] = 100 * np.log(d["emp"].shift(1) / d["emp"].shift(13))
    d["ur_lag"] = d["ur"].shift(1)
    d["dcr12_lag"] = d["cr"].shift(1) - d["cr"].shift(13)
    return d


def sample_months(d: pd.DataFrame) -> pd.PeriodIndex:
    """Return the months at which decisions are tested: START onward."""
    idx = d.index
    if not isinstance(idx, pd.PeriodIndex):
        msg = "frame must have a PeriodIndex"
        raise TypeError(msg)
    return idx[idx >= START]
