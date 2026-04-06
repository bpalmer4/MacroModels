"""Inflation data loading.

Provides CPI trimmed mean and weighted median inflation from ABS (6401.0).
Includes a spliced monthly CPI index for deflation, combining three sources:
  1. Quarterly CPI (6401.0 Appendix 1a) interpolated to monthly (pre-2017)
  2. Monthly CPI Indicator (6484.0) — Sep 2017 to Sep 2025
  3. Monthly CPI (6401.0 table 640106) — Apr 2024 onwards
"""

from functools import cache

import readabs as ra

from src.data.abs_loader import ReqsTuple, load_series
from src.data.dataseries import DataSeries
from src.data.series_specs import (
    CPI_ALL_GROUPS,
    CPI_HEADLINE_ANNUAL,
    CPI_TRIMMED_MEAN_ANNUAL,
    CPI_TRIMMED_MEAN_QUARTERLY,
    CPI_WEIGHTED_MEDIAN,
)

# Monthly CPI All Groups SA index from table 640106 (available from Apr 2024)
_CPI_MONTHLY_SA_INDEX = ReqsTuple(
    cat="6401.0",
    table="640106",
    did="All groups CPI, seasonally adjusted",
    stype="S",
    unit="Index Numbers",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

# ABS Monthly CPI Indicator landing page (discontinued catalogue, needs explicit URL)
_CPI_6484_URL = (
    "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/"
    "monthly-consumer-price-index-indicator/latest-release"
)


def get_trimmed_mean_qrtly() -> DataSeries:
    """Get quarterly trimmed mean inflation.

    Returns percentage change from previous period.

    Returns:
        DataSeries with quarterly inflation (%)

    """
    return load_series(CPI_TRIMMED_MEAN_QUARTERLY)


def get_trimmed_mean_annual() -> DataSeries:
    """Get annual trimmed mean inflation.

    Returns percentage change from corresponding quarter of previous year.

    Returns:
        DataSeries with annual inflation (%)

    """
    return load_series(CPI_TRIMMED_MEAN_ANNUAL)


def get_weighted_median_annual() -> DataSeries:
    """Get annual weighted median inflation.

    Returns percentage change from corresponding quarter of previous year.

    Returns:
        DataSeries with annual weighted median inflation (%)

    """
    return load_series(CPI_WEIGHTED_MEDIAN)


def get_headline_annual() -> DataSeries:
    """Get annual headline CPI inflation.

    Returns percentage change from corresponding quarter of previous year.
    Available from 1949Q3.

    Returns:
        DataSeries with annual headline inflation (%)

    """
    return load_series(CPI_HEADLINE_ANNUAL)


@cache
def _load_6484_monthly_cpi() -> DataSeries:
    """Load Monthly CPI Indicator All Groups SA index from discontinued 6484.0."""
    data, _meta = ra.read_abs_cat("6484.0", url=_CPI_6484_URL, verbose=False)
    series = data["648401"]["A128481587A"].dropna()
    return DataSeries(
        data=series,
        source="ABS",
        units="Index Numbers",
        description="Monthly CPI Indicator All Groups SA (6484.0, Sep 2017 – Sep 2025)",
        cat="6484.0",
        table="648401",
        series_id="A128481587A",
    )


@cache
def get_genuine_monthly_cpi_index() -> DataSeries:
    """Get genuine (non-interpolated) monthly CPI SA index.

    Splices:
    - 6484.0 Monthly CPI Indicator (Sep 2017 – Sep 2025), rebased to match 640106
    - 6401.0 table 640106 (Apr 2024 onwards), takes precedence in overlap

    Returns:
        DataSeries with genuine monthly CPI observations only (no interpolation).

    """
    new_monthly = load_series(_CPI_MONTHLY_SA_INDEX).data.dropna()
    old_monthly = _load_6484_monthly_cpi().data

    # Rebase 6484.0 to match 640106 base period at their overlap start
    overlap = old_monthly.index.intersection(new_monthly.index)
    rebase_factor = new_monthly[overlap[0]] / old_monthly[overlap[0]]
    old_rebased = old_monthly * rebase_factor

    # New takes precedence where both exist
    spliced = new_monthly.combine_first(old_rebased)

    return DataSeries(
        data=spliced,
        source="ABS",
        units="Index Numbers",
        description="CPI All Groups SA index (monthly, genuine obs from 6484.0 + 640106)",
        cat="6401.0",
        table="648401 / 640106",
    )


@cache
def get_monthly_cpi_index() -> DataSeries:
    """Get monthly CPI All Groups SA index, spliced from quarterly and monthly sources.

    Combines three sources (each taking precedence over the previous):
    1. Quarterly CPI (6401.0 Appendix 1a) interpolated to monthly (pre-Sep 2017)
    2. Monthly CPI Indicator (6484.0) from Sep 2017 (rebased)
    3. Monthly CPI (6401.0 table 640106) from Apr 2024

    Returns:
        DataSeries with monthly CPI index (seasonally adjusted)

    """
    # Quarterly CPI index → interpolate to monthly (full history)
    quarterly_ds = load_series(CPI_ALL_GROUPS)
    quarterly_monthly = ra.qtly_to_monthly(quarterly_ds.data, interpolate=True)

    # Genuine monthly data (6484.0 + 640106, already spliced and rebased)
    genuine = get_genuine_monthly_cpi_index().data

    # Rebase quarterly-interpolated series to match genuine monthly base period
    splice_point = genuine.index[0]
    rebase_factor = genuine.iloc[0] / quarterly_monthly[splice_point]
    quarterly_rebased = quarterly_monthly * rebase_factor

    # Genuine monthly takes precedence, quarterly fills pre-2017 history
    spliced = genuine.combine_first(quarterly_rebased)

    return DataSeries(
        data=spliced,
        source="ABS",
        units="Index Numbers",
        description="CPI All Groups SA index (monthly, spliced from quarterly + 6484.0 + 640106)",
        cat="6401.0",
        table="640106 / 648401 / 64010Appendix1a",
    )
