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
from readabs.download_cache import HttpError

from src.data.dataseries import DataSeries

_TABLE = "5682015"
_DID_TOTAL_CVM = (
    "Household spending ;  Total (Household Spending Categories) ;"
    "  Australia ;  Chain Volume Measures ;"
)


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
def get_household_spending_cvm_qrtly(history: str) -> DataSeries:
    """Get Total Household Spending (quarterly, CVM, seasonally adjusted).

    The quarterly CVM table (5682015) only ships with the monthly 5682.0 release
    that lands on a quarter-end month, so ABS requires the snapshot to be fetched by
    a specific quarter-end month. ``history`` is REQUIRED and must be that quarter's
    end month (e.g. ``"dec-2025"``) — there is deliberately no date-based default,
    because the nowcast target quarter is derived from the GDP data and a date-based
    guess can point at an unpublished future vintage. Returns an empty DataSeries on
    any failure (so downstream models can degrade gracefully).

    Args:
        history: quarter-end month as ``"mmm-yyyy"`` (lowercase) to fetch.

    """
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
    except (KeyError, ValueError, OSError, HttpError):
        # HttpError: ABS returns 503 (not 404) for an unpublished snapshot month.
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


def get_household_spending_cvm_growth_latest(target_quarter: pd.Period) -> DataSeries:
    """Get quarterly CVM growth from the target quarter's snapshot, with fallback.

    The target quarter's end-month snapshot (e.g. 2026Q2 -> "jun-2026") is not
    published until ~5 weeks after the quarter ends, so a T-0 nowcast run straight
    after a GDP release falls back to the previous quarter-end snapshot (growth
    history through the prior quarter).

    Args:
        target_quarter: nowcast target quarter; its end month is tried first.

    """
    target_month = target_quarter.asfreq("M", how="end").strftime("%b-%Y").lower()
    ds = get_household_spending_cvm_growth_qrtly(target_month)
    if len(ds.data) == 0:
        prev_month = (target_quarter - 1).asfreq("M", how="end").strftime("%b-%Y").lower()
        ds = get_household_spending_cvm_growth_qrtly(prev_month)
    return ds


def get_household_spending_cvm_growth_qrtly(history: str) -> DataSeries:
    """Get quarterly Total Household Spending CVM growth (log difference).

    Args:
        history: quarter-end month as ``"mmm-yyyy"`` to fetch (REQUIRED; ABS fetches
            the quarterly CVM table by quarter-end-month snapshot). Passed through to
            :func:`get_household_spending_cvm_qrtly`.

    Returns:
        DataSeries with household spending growth (% per quarter)

    """
    ds = get_household_spending_cvm_qrtly(history)
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
