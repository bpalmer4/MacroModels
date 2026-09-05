"""Global Supply Chain Pressure Index, fetched live from the NY Fed.

Source: https://www.newyorkfed.org/research/policy/gscpi

**Why this exists alongside `gscpi.py`.** That module reads a static workbook
committed at `input_data/gscpi_data.xls`, which goes stale silently: as at
December 2025 the checked-in copy stopped at April 2024 while the published
series ran to August 2026. A model whose answer is the last observation cannot
be fed a series that quietly stops two years early.

This module is deliberately separate rather than a rewrite of `gscpi.py`. The
`nairu` package reads the static file, and its published results depend on
those exact numbers; swapping the source underneath it would move them without
anyone asking for that. So the old loader keeps the old file, and anything new
uses this one.

Revisions between the two vintages are small — over the 316-month overlap the
largest was 0.16 and the mean 0.03 — so the separation is about not moving a
published model unbidden, not about the numbers being in dispute.

Like `gscpi.py`, this returns the series raw. Masking it to a COVID window,
lagging it, or squaring it are model-level decisions and belong in the model
that makes them.
"""

from functools import cache
from io import BytesIO
from pathlib import Path

import pandas as pd
import readabs as ra

from src.data.dataseries import DataSeries

GSCPI_URL = (
    "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xls"
)

_SHEET = "GSCPI Monthly Data"
_COLUMN = "GSCPI"

# Sits beside the ABS cache rather than in `input_data`, because it is a cache
# and not an input: it is re-fetched when the NY Fed updates the workbook, and
# is safe to delete.
_CACHE_DIR = Path(__file__).parent.parent.parent / ".readabs_cache"


@cache
def get_gscpi_monthly_live(verbose: bool = False) -> DataSeries:
    """Get the GSCPI (monthly), downloading from the NY Fed with a local cache.

    Uses `readabs.download_cache`, so the workbook is re-fetched only when the
    server reports a newer `Last-Modified`, and a cached copy is used with a
    warning when the network is unavailable.

    Returns:
        DataSeries with the monthly GSCPI, indexed by month-end timestamps.

    """
    content = ra.download_cache.get_file(
        GSCPI_URL,
        cache_dir=_CACHE_DIR,
        cache_prefix="gscpi",
        verbose=verbose,
    )

    frame = pd.read_excel(BytesIO(content), sheet_name=_SHEET, index_col=0, parse_dates=True)
    if _COLUMN not in frame.columns:
        raise ValueError(f"sheet {_SHEET!r} has no {_COLUMN!r} column — has the NY Fed layout changed?")

    # The workbook carries title rows above the data, which parse to NaT, and
    # trailing blanks below it. Drop both rather than assuming a fixed offset.
    series = frame[_COLUMN]
    series = series[series.index.notna()].dropna()
    series.index = pd.DatetimeIndex(series.index)
    series = series.sort_index()

    if series.empty:
        raise ValueError("GSCPI download parsed to an empty series")

    return DataSeries(
        data=series,
        source="NY Fed",
        units="Index (std devs from mean)",
        description="Global Supply Chain Pressure Index (monthly, live)",
    )


@cache
def get_gscpi_qrtly_live(verbose: bool = False) -> DataSeries:
    """Get the GSCPI as a quarterly mean, downloaded from the NY Fed.

    Returns:
        DataSeries with the quarterly GSCPI, on a December-ending PeriodIndex
        to match the rest of the package.

    """
    monthly = get_gscpi_monthly_live(verbose=verbose)
    quarterly = ra.monthly_to_qtly(monthly.data, q_ending="DEC", f="mean")
    quarterly.index = pd.PeriodIndex(quarterly.index, freq="Q")

    return DataSeries(
        data=quarterly,
        source="NY Fed",
        units="Index (std devs from mean)",
        description="Global Supply Chain Pressure Index (quarterly mean, live)",
    )
