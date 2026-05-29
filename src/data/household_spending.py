"""Household spending data loading (5682.0 quarterly CVM).

Provides the quarterly Total Household Spending Indicator in Chain Volume Measures
(real, seasonally adjusted) from ABS 5682.0 table 5682015.

The quarterly tables only ship in the monthly 5682.0 release that lands on a
quarter-end month (Mar/Jun/Sep/Dec). The loader always requests the most recent
quarter-end snapshot via the `history` parameter, so it reliably returns data
regardless of which month it is run in.

History begins 2014Q3 (~46 growth obs as of 2026), which is sufficient for a
bridge or factor-model indicator but too short for the BVAR's long-history panel.
"""

from functools import cache

import numpy as np
import pandas as pd
import readabs as ra
from readabs import metacol as mc

from src.data.dataseries import DataSeries

_TABLE = "5682015"
_DID_TOTAL_CVM = (
    "Household spending ;  Total (Household Spending Categories) ;"
    "  Australia ;  Chain Volume Measures ;"
)


def _latest_quarter_end_month() -> str:
    """Return the most recent quarter-end month as 'mmm-yyyy' (lowercase)."""
    today = pd.Timestamp.today()
    current_q_end = pd.Period(today, freq="Q-DEC").asfreq("M", how="end")
    if current_q_end > pd.Period(today, freq="M"):
        # Quarter end is in the future; use the previous quarter end
        current_q_end = (pd.Period(today, freq="Q-DEC") - 1).asfreq("M", how="end")
    return current_q_end.strftime("%b-%Y").lower()


def _empty_series() -> DataSeries:
    """Return an empty DataSeries placeholder when 5682015 cannot be loaded."""
    return DataSeries(
        data=pd.Series(dtype=float),
        source="ABS",
        units="Index Numbers",
        description="Household spending Total CVM SA (5682.0 table 5682015) — unavailable",
        cat="5682.0",
        table=_TABLE,
        stype="Seasonally Adjusted",
    )


@cache
def get_household_spending_cvm_qrtly() -> DataSeries:
    """Get Total Household Spending (quarterly, CVM, seasonally adjusted).

    Always fetches with the most recent quarter-end-month `history` so the
    quarterly table is present even when called mid-quarter. Returns an empty
    DataSeries on any failure (so downstream models can degrade gracefully).
    """
    history = _latest_quarter_end_month()
    try:
        data, meta = ra.read_abs_cat(
            "5682.0", history=history, single_excel_only=_TABLE, verbose=False
        )
        table_meta = meta[meta[mc.table] == _TABLE]
        match = table_meta[
            (table_meta[mc.did] == _DID_TOTAL_CVM)
            & (table_meta[mc.stype] == "Seasonally Adjusted")
        ]
        if len(match) == 0:
            return _empty_series()
        series_id = match[mc.id].iloc[0]
        series = data[_TABLE][series_id].dropna()
    except (KeyError, ValueError, OSError):
        return _empty_series()

    return DataSeries(
        data=series,
        source="ABS",
        units="Index Numbers",
        description="Household spending, Total, Chain Volume Measures, SA (5682.0)",
        series_id=series_id,
        table=_TABLE,
        cat="5682.0",
        stype="Seasonally Adjusted",
    )


def get_household_spending_cvm_growth_qrtly() -> DataSeries:
    """Get quarterly Total Household Spending CVM growth (log difference).

    Returns:
        DataSeries with household spending growth (% per quarter)

    """
    ds = get_household_spending_cvm_qrtly()
    if len(ds.data) == 0:
        return DataSeries(
            data=pd.Series(dtype=float),
            source=ds.source,
            units="% per quarter",
            description="Household spending CVM growth (unavailable)",
            cat=ds.cat,
            table=ds.table,
        )
    log_idx = np.log(ds.data) * 100
    growth = log_idx.diff(1)
    return DataSeries(
        data=growth,
        source=ds.source,
        units="% per quarter",
        description="Household spending Total CVM growth (quarterly, log difference)",
        cat=ds.cat,
        table=ds.table,
    )
