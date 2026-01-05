"""RBA (Reserve Bank of Australia) data loading utilities.

Provides functions for fetching RBA data including:
- Cash rate (official cash rate target)
- Inflation expectations (surveys and bond yields)
- Exchange rates

Uses the readabs library for RBA data access.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from readabs import read_rba_ocr

from src.data.dataseries import DataSeries

# --- Constants ---

PI_TARGET = 2.5  # RBA inflation target midpoint (%)
PI_TARGET_START = pd.Period("1993Q1")  # Inflation targeting introduced
PI_TARGET_FULL = pd.Period("1998Q1")  # Full credibility assumed


# --- Cash Rate ---


def get_cash_rate() -> DataSeries:
    """Get RBA Official Cash Rate (monthly).

    Returns:
        DataSeries with monthly OCR from RBA (1990-present)

    """
    ocr = read_rba_ocr()
    ocr = ocr.squeeze()

    return DataSeries(
        data=ocr,
        source="RBA",
        units="%",
        description="Official Cash Rate (Monthly)",
        table="OCR",
    )


def get_historical_interbank_rate(path: str | Path) -> DataSeries:
    """Get historical interbank overnight rate from parquet file.

    Args:
        path: Path to parquet file with historical rates

    Returns:
        DataSeries with monthly interbank rate (pre-1990)

    """
    historical = pd.read_parquet(path)
    if isinstance(historical, pd.DataFrame):
        historical = historical.iloc[:, 0]

    return DataSeries(
        data=historical,
        source="RBA",
        units="%",
        description="Interbank Overnight Cash Rate (Historical)",
        table="Historical",
    )


# --- Inflation Expectations ---


def get_inflation_expectations() -> DataSeries:
    """Get inflation expectations from RBA PIE_RBAQ series.

    Loads raw RBA series from CSV file and converts quarterly to annual rate.
    No extension needed since anchor is 2.5% target after 1998Q1 anyway.

    Returns:
        DataSeries with annual inflation expectations

    """
    # Load RBA PIE_RBAQ series from CSV (in project input_data/ directory)
    csv_path = Path(__file__).parent.parent.parent / "input_data" / "PIE_RBAQ.CSV"
    rba_pie = pd.read_csv(csv_path, index_col=0, parse_dates=False)["PIE_RBAQ"]
    rba_pie.index = pd.PeriodIndex(rba_pie.index, freq="Q")
    rba_pie = rba_pie.dropna()

    # Convert quarterly rate to annual rate
    rba_annual = ((1 + rba_pie / 100) ** 4 - 1) * 100
    rba_annual.name = "Inflation Expectations"

    return DataSeries(
        data=rba_annual,
        source="RBA",
        units="%",
        description="Inflation Expectations (annual rate)",
        table="PIE_RBAQ",
        series_id="PIE_RBAQ",
    )


# --- Inflation Anchor ---


def get_inflation_anchor() -> DataSeries:
    """Construct inflation anchor series for Phillips curve estimation.

    The anchor transitions from inflation expectations to the inflation target:
    - Pre-1993Q1: Uses inflation expectations
    - 1993Q1-1998Q1: Linear phase-in from expectations to target
    - Post-1998Q1: Fixed at inflation target (2.5%)

    Returns:
        DataSeries with annual inflation anchor

    """
    # Get expectations (only needed for pre-1998Q1 periods)
    exp_data = get_inflation_expectations()
    expectations = exp_data.data

    # Build full index from expectations start to present
    current_quarter = pd.Timestamp.today().to_period("Q")
    full_index = pd.period_range(
        start=expectations.index.min(),
        end=current_quarter,
        freq="Q",
    )

    # Start with target value for all periods
    anchor = pd.Series(PI_TARGET, index=full_index)

    # Pre-1993Q1: Use expectations
    pre_target = full_index < PI_TARGET_START
    anchor[pre_target] = expectations.reindex(full_index[pre_target])

    # 1993Q1-1998Q1: Linear phase-in from expectations to target
    phase_in = (full_index >= PI_TARGET_START) & (full_index < PI_TARGET_FULL)
    phase_periods = full_index[phase_in]
    n_periods = len(phase_periods)
    if n_periods > 0:
        weights = np.linspace(0, 1, n_periods + 1)[1:]  # exclude 0, include 1
        exp_values = expectations.reindex(phase_periods)
        anchor[phase_in] = (1 - weights) * exp_values + weights * PI_TARGET

    # Post-1998Q1: Already set to PI_TARGET above
    anchor.name = "Inflation Anchor"

    return DataSeries(
        data=anchor,
        source="RBA",
        units="%",
        description="Inflation Anchor (Expectations → Target Transition)",
        metadata={
            "target_start": str(PI_TARGET_START),
            "target_full": str(PI_TARGET_FULL),
            "target_rate": PI_TARGET,
        },
    )


# --- Bond Yields (F2) ---

F2_HIST_URL = "https://www.rba.gov.au/statistics/tables/xls-hist/f02histhist.xls"
F2_CURRENT_URL = "https://www.rba.gov.au/statistics/tables/xls/f02hist.xlsx"


def _load_f2_series(col: str) -> pd.Series:
    """Load a series from RBA F2 bond yield tables (spliced historical + current).

    Args:
        col: Column name (e.g., "FCMYGBAG10", "FCMYGBAGI")

    Returns:
        Combined historical and current series with DatetimeIndex

    """
    # Historical (1969-2013)
    hist = pd.read_excel(F2_HIST_URL, sheet_name="Data", skiprows=10, index_col=0)
    hist.index = pd.to_datetime(hist.index)

    # Current (2013+)
    curr = pd.read_excel(F2_CURRENT_URL, sheet_name="Data", skiprows=10, index_col=0)
    curr.index = pd.to_datetime(curr.index)

    # Splice: current takes precedence
    hist_s = hist[col] if col in hist.columns else pd.Series(dtype=float)
    curr_s = curr[col] if col in curr.columns else pd.Series(dtype=float)

    combined = curr_s.combine_first(hist_s)
    return combined.dropna()


def get_bond_yield_10y() -> DataSeries:
    """Get 10-year nominal government bond yield (spliced 1969-present).

    Returns:
        DataSeries with monthly 10-year bond yield (%)

    """
    series = _load_f2_series("FCMYGBAG10")

    return DataSeries(
        data=series,
        source="RBA",
        units="%",
        description="10-year Government Bond Yield",
        table="F2",
        series_id="FCMYGBAG10",
    )


def get_indexed_bond_yield() -> DataSeries:
    """Get indexed (inflation-linked) bond yield (spliced 1986-present).

    Returns:
        DataSeries with monthly indexed bond yield (%)

    """
    series = _load_f2_series("FCMYGBAGI")

    return DataSeries(
        data=series,
        source="RBA",
        units="%",
        description="Indexed Bond Yield",
        table="F2",
        series_id="FCMYGBAGI",
    )


# --- Exchange Rates ---


def _load_f11_series(col_pattern: str) -> pd.Series:
    """Load a series from RBA F11 exchange rate tables.

    Args:
        col_pattern: Column pattern to match (e.g., "FXRUSD", "FXRTWI")

    Returns:
        Combined historical and current series with DatetimeIndex

    """
    # Historical exchange rates
    hist_url = "https://www.rba.gov.au/statistics/tables/xls-hist/f11hist-1969-2009.xls"
    hist_rates = pd.read_excel(hist_url, sheet_name="Data", index_col=0, skiprows=10)

    # Current exchange rates
    now_url = "https://www.rba.gov.au/statistics/tables/xls-hist/f11hist.xls"
    current_rates = pd.read_excel(now_url, sheet_name="Data", index_col=0, skiprows=10)

    # Find the relevant column
    hist_col = [c for c in hist_rates.columns if col_pattern in c]
    curr_col = [c for c in current_rates.columns if col_pattern in c]

    hist_series = hist_rates[hist_col[0]] if hist_col else pd.Series(dtype=float)
    curr_series = current_rates[curr_col[0]] if curr_col else pd.Series(dtype=float)

    combined = curr_series.combine_first(hist_series)
    combined.index = pd.to_datetime(combined.index)
    return combined


def get_exchange_rate(currency: str = "USD") -> DataSeries:
    """Get exchange rate from RBA F11 table.

    Args:
        currency: Currency code (default "USD")

    Returns:
        DataSeries with monthly exchange rate

    """
    combined = _load_f11_series(f"FXRU{currency}")

    return DataSeries(
        data=combined,
        source="RBA",
        units=f"AUD/{currency}",
        description=f"Exchange Rate AUD/{currency}",
        table="F11",
        series_id=f"FXRU{currency}",
    )


def get_twi() -> DataSeries:
    """Get Trade-Weighted Index from RBA F11 table.

    The TWI measures the value of the Australian dollar against a basket
    of currencies weighted by trade shares. Base: May 1970 = 100.

    Returns:
        DataSeries with monthly TWI (index)

    """
    combined = _load_f11_series("FXRTWI")

    return DataSeries(
        data=combined,
        source="RBA",
        units="Index",
        description="Trade-Weighted Index (May 1970 = 100)",
        table="F11",
        series_id="FXRTWI",
    )


# --- Testing ---

if __name__ == "__main__":
    print("Testing RBA loader...\n")

    # Test cash rate
    print("Cash rate (modern only):")
    cash_rates = get_cash_rate()
    print(f"Monthly: {cash_rates['monthly']}")
    print(f"Quarterly: {cash_rates['quarterly']}")

    # Test inflation expectations
    print("\nInflation expectations:")
    exp = get_inflation_expectations()
    print(f"Expectations: {exp}")
    print(f"Latest: {exp.data.tail()}")

    # Test inflation anchor
    print("\nInflation anchor:")
    anchor = get_inflation_anchor()
    print(f"Anchor: {anchor}")
    print(f"Latest: {anchor.data.tail()}")
