"""Debt servicing data loading from ABS National Accounts.

Computes household debt servicing ratio (DSR) from ABS 5206.0:
- Interest payments on dwellings + consumer debt
- Divided by gross disposable income

Also loads RBA E13 data for mortgage buffers (excess payments) where available.
"""

import pandas as pd
import readabs as ra
from readabs import metacol as mc

from src.data.dataseries import DataSeries

# --- Constants ---

# ABS 5206.0 Household Income Account series IDs (Seasonally Adjusted)
INTEREST_DWELLINGS_ID = "A2302926A"  # Property income payable - Interest - Dwellings
INTEREST_CONSUMER_ID = "A2302927C"  # Property income payable - Interest - Consumer debt
GROSS_DISP_INCOME_ID = "A2302939L"  # Gross Disposable Income

# RBA E13 series for buffers (only available from 2009Q1)
E13_URL = "https://www.rba.gov.au/statistics/tables/xls/e13hist.xlsx"
EXCESS_PAYMENTS_ID = "LPHTEXRI"  # Excess payments / Household disposable income


# --- Private Helpers ---


def _load_household_income_data() -> pd.DataFrame:
    """Load ABS 5206.0 Household Income Account table.

    Returns:
        DataFrame with household income account series

    """
    dictionary, _meta = ra.read_abs_cat(
        "5206.0", single_excel_only="5206020_Household_Income", verbose=False
    )
    return dictionary["5206020_Household_Income"]


def _load_rba_e13() -> pd.DataFrame:
    """Load RBA E13 Housing Loan Payments table.

    Returns:
        DataFrame with E13 series (DatetimeIndex)

    """
    df = pd.read_excel(E13_URL, sheet_name="Data", index_col=0, skiprows=10)
    df.index = pd.to_datetime(df.index)
    return df


# --- Public Functions ---


def get_dsr_qrtly() -> DataSeries:
    """Get debt servicing ratio (interest payments / disposable income).

    Computed from ABS 5206.0 Household Income Account:
    DSR = (Interest on Dwellings + Interest on Consumer Debt) / Gross Disposable Income

    Returns:
        DataSeries with quarterly DSR (%, seasonally adjusted)

    """
    data = _load_household_income_data()

    interest_dwellings = data[INTEREST_DWELLINGS_ID]
    interest_consumer = data[INTEREST_CONSUMER_ID]
    gdi = data[GROSS_DISP_INCOME_ID]

    # Total household interest payments
    total_interest = interest_dwellings + interest_consumer

    # DSR as percentage
    dsr = (total_interest / gdi) * 100

    return DataSeries(
        data=dsr,
        source="ABS",
        units="%",
        description="Debt Servicing Ratio (Interest Payments / Disposable Income)",
        cat="5206.0",
        table="5206020_Household_Income",
        stype="Seasonally Adjusted",
    )


def get_dsr_change_qrtly() -> DataSeries:
    """Get change in debt servicing ratio (ΔDSR).

    Quarterly change in DSR. Positive values indicate rising debt burden.

    Returns:
        DataSeries with quarterly ΔDSR (percentage points)

    """
    dsr = get_dsr_qrtly().data
    delta_dsr = dsr.diff()

    return DataSeries(
        data=delta_dsr,
        source="Derived",
        units="pp",
        description="Change in Debt Servicing Ratio",
    )


def get_dsr_change_lagged_qrtly() -> DataSeries:
    """Get lagged change in debt servicing ratio (ΔDSR at t-1).

    Returns:
        DataSeries with lagged quarterly ΔDSR (percentage points)

    """
    delta_dsr = get_dsr_change_qrtly().data
    delta_dsr_1 = delta_dsr.shift(1)

    return DataSeries(
        data=delta_dsr_1,
        source="Derived",
        units="pp",
        description="Change in Debt Servicing Ratio (lagged)",
    )


def get_mortgage_buffers_qrtly() -> DataSeries:
    """Get mortgage buffers (excess payments / income) from RBA E13.

    This is the ratio of excess payments (payments above scheduled)
    to household disposable income. Higher values indicate larger
    mortgage payment buffers that can absorb rate rises.

    Note: Only available from 2009Q1.

    Returns:
        DataSeries with quarterly excess payments ratio (%)

    """
    df = _load_rba_e13()

    # Find column containing the series ID
    cols = [c for c in df.columns if EXCESS_PAYMENTS_ID in c]
    if not cols:
        raise ValueError(f"Series {EXCESS_PAYMENTS_ID} not found in RBA E13")

    series = df[cols[0]].dropna()

    # Convert to quarterly PeriodIndex
    quarterly = series.resample("QE").last()
    quarterly.index = quarterly.index.to_period("Q")

    return DataSeries(
        data=quarterly,
        source="RBA",
        units="%",
        description="Excess Payments / Disposable Income (Mortgage Buffers)",
        table="E13",
        series_id=EXCESS_PAYMENTS_ID,
    )


# --- Testing ---

if __name__ == "__main__":
    print("Testing debt servicing loader...\n")

    print("=== DSR from ABS 5206.0 ===")
    dsr = get_dsr_qrtly()
    print(f"{dsr}")
    print(f"Latest values:\n{dsr.data.tail(4).round(2)}\n")

    print("=== ΔDSR ===")
    delta_dsr = get_dsr_change_qrtly()
    print(f"{delta_dsr}")
    print(f"Latest values:\n{delta_dsr.data.tail(4).round(3)}\n")

    print("=== ΔDSR (lagged) ===")
    delta_dsr_1 = get_dsr_change_lagged_qrtly()
    print(f"{delta_dsr_1}")
    print(f"Latest values:\n{delta_dsr_1.data.tail(4).round(3)}\n")

    print("=== Mortgage Buffers from RBA E13 ===")
    try:
        buffers = get_mortgage_buffers_qrtly()
        print(f"{buffers}")
        print(f"Latest values:\n{buffers.data.tail(4).round(2)}")
    except Exception as e:
        print(f"Failed: {e}")
