"""Cash rate data loading.

Provides spliced cash rate combining modern OCR with historical interbank rate,
plus derived r* and real rate gap calculations.
"""

import pandas as pd

from src.data.dataseries import DataSeries
from src.data.henderson import hma
from src.data.rba_loader import get_cash_rate as rba_get_cash_rate
from src.data.rba_loader import get_historical_interbank_rate, get_inflation_anchor
from src.data.series_specs import HISTORICAL_RATE_FILE
from src.data.transforms import splice_series


def get_cash_rate_monthly() -> DataSeries:
    """Get cash rate spliced from OCR and historical interbank rate (monthly).

    Combines:
    - Modern OCR from RBA (1990-present)
    - Historical interbank overnight rate (pre-1990)

    Favours OCR where both have data.

    Returns:
        DataSeries with monthly cash rate (%)

    """
    ocr = rba_get_cash_rate()
    historical = get_historical_interbank_rate(HISTORICAL_RATE_FILE)

    # Exclude early OCR records with non-numeric (range) values
    ocr_clean = pd.to_numeric(ocr.data, errors="coerce").dropna()

    # Convert historical index to PeriodIndex if needed
    hist_data = historical.data
    if not isinstance(hist_data.index, pd.PeriodIndex):
        hist_data.index = pd.PeriodIndex(hist_data.index, freq="M")

    # Splice: favour OCR where both have data
    monthly = splice_series(ocr_clean, hist_data)

    return DataSeries(
        data=monthly,
        source="RBA",
        units="%",
        description="Cash rate (spliced OCR + historical interbank)",
    )


def get_cash_rate_qrtly() -> DataSeries:
    """Get cash rate as end-of-quarter value (quarterly).

    Returns:
        DataSeries with quarterly cash rate (%)

    """
    monthly = get_cash_rate_monthly()
    quarterly = monthly.data.sort_index().groupby(monthly.data.index.asfreq("Q")).last()

    return DataSeries(
        data=quarterly,
        source="RBA",
        units="%",
        description="Cash rate (quarterly, end of period)",
    )


def compute_r_star(
    capital_growth: pd.Series,
    lf_growth: pd.Series,
    mfp_growth: pd.Series,
    alpha: float | pd.Series = 0.3,
    hma_term: int = 13,
) -> DataSeries:
    """Compute deterministic r* as smoothed potential growth.

    r* ≈ α×g_K + (1-α)×g_L + g_MFP (annualized and smoothed)

    Args:
        capital_growth: Capital stock growth (quarterly, smoothed)
        lf_growth: Labour force growth (quarterly, smoothed)
        mfp_growth: MFP growth (quarterly, trend)
        alpha: Capital share (default 0.3), can be time-varying Series
        hma_term: Henderson MA smoothing term (default 13)

    Returns:
        DataSeries with r* estimate (%)

    """
    quarterly_growth = (
        alpha * capital_growth
        + (1 - alpha) * lf_growth
        + mfp_growth
    )
    annual_growth = quarterly_growth.rolling(4).sum()
    annual_growth = annual_growth.bfill()
    r_star = hma(annual_growth.dropna(), hma_term)

    return DataSeries(
        data=r_star,
        source="Derived",
        units="% per year",
        description=f"Deterministic r* (potential growth, α={alpha})",
    )


def get_real_rate_gap_qrtly(
    capital_growth: pd.Series,
    lf_growth: pd.Series,
    mfp_growth: pd.Series,
    alpha: float = 0.3,
    hma_term: int = 13,
) -> DataSeries:
    """Compute real rate gap: r - π_anchor - r*.

    Args:
        capital_growth: Capital stock growth (quarterly, smoothed)
        lf_growth: Labour force growth (quarterly, smoothed)
        mfp_growth: MFP growth (quarterly, trend)
        alpha: Capital share (default 0.3)
        hma_term: Henderson MA smoothing term (default 13)

    Returns:
        DataSeries with real rate gap (%)

    """
    cash_rate = get_cash_rate_qrtly().data
    π_anchor = get_inflation_anchor().data
    r_star = compute_r_star(
        capital_growth, lf_growth, mfp_growth, alpha, hma_term
    ).data

    # Align indices
    common_idx = cash_rate.index.intersection(π_anchor.index).intersection(r_star.index)
    r_gap = cash_rate.loc[common_idx] - π_anchor.loc[common_idx] - r_star.loc[common_idx]

    return DataSeries(
        data=r_gap,
        source="Derived",
        units="%",
        description="Real rate gap (r - π_anchor - r*)",
    )


def get_real_rate_gap_lagged_qrtly(
    capital_growth: pd.Series,
    lf_growth: pd.Series,
    mfp_growth: pd.Series,
    alpha: float = 0.3,
    hma_term: int = 13,
) -> DataSeries:
    """Compute lagged real rate gap.

    Args:
        capital_growth: Capital stock growth (quarterly, smoothed)
        lf_growth: Labour force growth (quarterly, smoothed)
        mfp_growth: MFP growth (quarterly, trend)
        alpha: Capital share (default 0.3)
        hma_term: Henderson MA smoothing term (default 13)

    Returns:
        DataSeries with lagged real rate gap (%)

    """
    r_gap = get_real_rate_gap_qrtly(
        capital_growth, lf_growth, mfp_growth, alpha, hma_term
    )
    r_gap_1 = r_gap.data.shift(1)

    return DataSeries(
        data=r_gap_1,
        source="Derived",
        units="%",
        description="Real rate gap lagged one quarter",
    )
