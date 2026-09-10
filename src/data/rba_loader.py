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
    squeezed = read_rba_ocr().squeeze()
    if not isinstance(squeezed, pd.Series):
        raise TypeError(f"read_rba_ocr() did not squeeze to a Series, got {type(squeezed).__name__}")

    return DataSeries(
        data=squeezed,
        source="RBA",
        units="%",
        description="Official Cash Rate (Monthly)",
        table="OCR",
    )


F4_URL = "https://www.rba.gov.au/statistics/tables/xls/f04hist.xlsx"
F5_URL = "https://www.rba.gov.au/statistics/tables/xls/f05hist.xlsx"

# Retail deposit rates (F4) and housing lending rates (F5). Together with the
# cash rate they measure what the RBA's policy rate actually costs a borrower
# and pays a depositor, which stopped tracking the cash rate after the GFC:
# term deposits moved from 1.68 below it to 0.39 above between 2004-07 and
# 2015-19, and mortgages from 1.24 above to 2.99 above.
DEPOSIT_RATES = {
    "term_deposit": "FRDIRBTD10KAR",   # banks' term deposits, average of all terms
    "online_saver": "FRDIRSAO10K",
    "transaction": "FRDIRTAB5K",
}
LENDING_RATES = {
    "housing_oo": "FILRHLBVD",         # discounted variable, owner-occupier
    "housing_oo_standard": "FILRHLBVS",
    "housing_investor": "FILRHLBVDI",
    "housing_investor_io": "FILRHLBVDO",
}


def _load_rba_series(url: str, col: str, table: str, description: str) -> DataSeries:
    """Load one monthly series from an RBA rates table."""
    frame = pd.read_excel(url, sheet_name="Data", skiprows=10, index_col=0)
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[frame.index.notna()]
    if col not in frame.columns:
        raise KeyError(f"{col} is not in RBA {table}; columns are {list(frame.columns)}")

    return DataSeries(
        data=frame[col].dropna().astype(float),
        source="RBA",
        units="%",
        description=description,
        table=table,
        series_id=col,
    )


def get_deposit_rate(kind: str = "term_deposit") -> DataSeries:
    """Get a retail deposit rate from RBA F4 (monthly, 1981-present).

    Args:
        kind: One of `DEPOSIT_RATES` — "term_deposit" (banks' term deposits,
            average of all terms), "online_saver", or "transaction".

    """
    if kind not in DEPOSIT_RATES:
        raise ValueError(f"kind must be one of {sorted(DEPOSIT_RATES)}, got {kind!r}")
    return _load_rba_series(F4_URL, DEPOSIT_RATES[kind], "F4", f"Deposit rate: {kind}")


def get_lending_rate(kind: str = "housing_oo") -> DataSeries:
    """Get a housing lending rate from RBA F5 (monthly).

    Args:
        kind: One of `LENDING_RATES`. "housing_oo" is the discounted variable
            owner-occupier rate, the closest thing to what a borrower pays.

    """
    if kind not in LENDING_RATES:
        raise ValueError(f"kind must be one of {sorted(LENDING_RATES)}, got {kind!r}")
    return _load_rba_series(F5_URL, LENDING_RATES[kind], "F5", f"Lending rate: {kind}")


F1_URL = "https://www.rba.gov.au/statistics/tables/xls/f01hist.xlsx"

BAB_TENORS = (30, 90, 180)


def get_bank_bill_rate(tenor: int = 90) -> DataSeries:
    """Get the bank-accepted bill / negotiable CD rate from RBA F1 (monthly).

    The short rate that leads the cash rate. The overnight rate cannot move
    before the RBA moves it, so it says nothing about a tightening the market
    has priced but the Bank has not delivered; a 90-day bill covers the next
    quarter's expected path and does. The gap between them is small on average
    (+0.16 against the cash rate since 1993, sd 0.21) and opens exactly at the
    turning points: +0.90 in 1994Q4, +0.68 in 2008Q1, +0.57 in 2018Q2, +0.87 in
    2022Q2.

    3-month OIS (`FIRMMOIS3`) is the cleaner measure of the same thing, but the
    series ends in November 2022, so it cannot carry a current-vintage model.

    Args:
        tenor: Bill tenor in days (30, 90, or 180).

    Returns:
        DataSeries with the monthly bill rate (%), 1969-present for 90 days and
        1992-present for the other two tenors.

    """
    if tenor not in BAB_TENORS:
        raise ValueError(f"tenor must be one of {BAB_TENORS}, got {tenor}")

    col = f"FIRMMBAB{tenor}"
    frame = pd.read_excel(F1_URL, sheet_name="Data", skiprows=10, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    if col not in frame.columns:
        raise KeyError(f"{col} is not in RBA F1; columns are {list(frame.columns)}")

    return DataSeries(
        data=frame[col].dropna().astype(float),
        source="RBA",
        units="%",
        description=f"{tenor}-day Bank-Accepted Bill Rate",
        table="F1",
        series_id=col,
    )


def get_historical_interbank_rate(path: str | Path) -> DataSeries:
    """Get historical interbank overnight rate from parquet file.

    Args:
        path: Path to parquet file with historical rates

    Returns:
        DataSeries with monthly interbank rate (pre-1990)

    """
    loaded = pd.read_parquet(path)
    historical = loaded.iloc[:, 0] if isinstance(loaded, pd.DataFrame) else loaded
    if not isinstance(historical, pd.Series):
        raise TypeError(f"expected a Series from {path}, got {type(historical).__name__}")

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


def get_cgs_yield(maturity: int = 5) -> DataSeries:
    """Get a Commonwealth Government Securities yield at a given maturity.

    Args:
        maturity: Tenor in years (2, 3, 5, or 10).

    Returns:
        DataSeries with monthly CGS yield (%)

    """
    col = f"FCMYGBAG{maturity}"
    series = _load_f2_series(col)
    return DataSeries(
        data=series,
        source="RBA",
        units="%",
        description=f"{maturity}-year Government Bond Yield",
        table="F2",
        series_id=col,
    )


# --- Corporate Bond Yields (F3) ---

F3_URL = "https://www.rba.gov.au/statistics/tables/xls/f03hist.xlsx"


def get_corporate_bond_yield(rating: str = "A", maturity: int = 5) -> DataSeries:
    """Get non-financial corporate bond yield from RBA F3 (2005-present).

    Args:
        rating: "A" or "BBB".
        maturity: Target tenor in years (3, 5, 7, or 10).

    Returns:
        DataSeries with monthly corporate bond yield (%)

    """
    col = f"FNFY{rating}{maturity}M"
    df = pd.read_excel(F3_URL, sheet_name="Data", skiprows=10, index_col=0)
    df.index = pd.to_datetime(df.index)
    series = df[col].dropna()
    return DataSeries(
        data=series,
        source="Bloomberg; RBA",
        units="%",
        description=f"Non-financial corporate {rating}-rated bond yield, {maturity}y target tenor",
        table="F3",
        series_id=col,
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


def get_real_twi() -> DataSeries:
    """Get the real Trade-Weighted Index from RBA F15 table.

    The real TWI is the nominal TWI multiplied by relative consumer price
    levels (AU over trade-weighted partners), so it measures what a dollar
    buys in foreign goods relative to Australian goods. It is *not* the
    nominal TWI deflated: it rises when Australian prices rise faster than
    partners', which is the opposite direction to deflating a nominal value.

    Published quarterly (quarter-average), base March 1995 = 100, from
    1970Q2. Only the .xlsx path exists for this table.

    Returns:
        DataSeries with quarterly real TWI (index), DatetimeIndex

    """
    url = "https://www.rba.gov.au/statistics/tables/xls/f15hist.xlsx"
    table = pd.read_excel(url, sheet_name="Data", index_col=0, skiprows=10)

    series = table["FRERTWI"].astype(float).dropna()
    series.index = pd.to_datetime(series.index)

    return DataSeries(
        data=series,
        source="RBA",
        units="Index, March 1995 = 100",
        description="Real Trade-Weighted Index (March 1995 = 100)",
        table="F15",
        series_id="FRERTWI",
    )


# --- Testing ---

if __name__ == "__main__":
    print("Testing RBA loader...\n")

    # Test cash rate
    print("Cash rate (modern only):")
    cash_rates = get_cash_rate()
    print(f"Monthly: {cash_rates.description}")
    print(f"Latest: {cash_rates.data.tail()}")

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
