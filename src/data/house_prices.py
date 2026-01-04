"""Housing wealth data loading from ABS 5232.0 National Financial Accounts.

Loads household residential property values (land + dwellings) for computing
housing wealth effects on consumption.

Wealth channel mechanism:
    Rate ↑ → House prices ↓ → Household wealth ↓ → Consumption ↓

Components (2025Q3):
    - Land (non-produced):     ~$9.1T (80%)
    - Dwellings (structures):  ~$3.1T (20%)
    - Total residential:       ~$11.5T

Data sources:
    - 5232.0 (quarterly, 1988Q3+): Actual total housing wealth (land + dwellings)
    - 5204.0 (annual, 1960+): Dwelling capital stock (structures only)

For pre-1988 data, we backcast using annual dwelling stock scaled by an
extrapolated land/dwelling ratio. The ratio grew from ~2.1 (1988) to ~3.6 (2024)
as land values appreciated faster than structures.

References:
- RBA estimates ~25% of monetary policy GDP transmission via housing/wealth channel
- House prices typically fall 2-4% per 100bp rate increase
- Marginal propensity to consume out of housing wealth: ~2-3 cents per dollar

"""

import numpy as np
import pandas as pd
import readabs as ra
from scipy import interpolate

from src.data.dataseries import DataSeries

# --- Constants ---

# ABS 5232.0 National Financial Accounts - Household Balance Sheet (quarterly)
WEALTH_CAT = "5232.0"
WEALTH_TABLE = "5232035"

# Series IDs from household balance sheet
RESIDENTIAL_LAND_DWELLINGS_ID = "A83728305F"  # Residential land and dwellings (combined)
DWELLINGS_ONLY_ID = "A83722686L"  # Dwellings (structures only)
LAND_ONLY_ID = "A83722664X"  # Land (non-produced assets)

# ABS 5204.0 National Accounts - Capital Stock (annual, back to 1960)
CAPITAL_CAT = "5204.0"
CAPITAL_TABLE = "5204056"
DWELLING_CAPITAL_STOCK_ID = "A2422529T"  # Dwellings; End-year net capital stock

# Ratio trend: Total housing wealth / Dwelling structures
# Observed: 2.08 (1988) → 3.60 (2024), ~0.042/year growth
# Extrapolating back: ~1.9 in 1984
RATIO_1988 = 2.08
RATIO_ANNUAL_GROWTH = 0.042


# --- Private Helpers ---


def _load_household_balance_sheet() -> pd.DataFrame:
    """Load ABS 5232.0 Household Balance Sheet table.

    Returns:
        DataFrame with household asset series

    """
    dictionary, _meta = ra.read_abs_cat(
        WEALTH_CAT, single_excel_only=WEALTH_TABLE, verbose=False
    )
    return dictionary[WEALTH_TABLE]


def _load_dwelling_capital_stock() -> pd.Series:
    """Load annual dwelling capital stock from ABS 5204.0 (back to 1960).

    Returns:
        Series with annual dwelling net capital stock ($ millions)

    """
    dictionary, _meta = ra.read_abs_cat(
        CAPITAL_CAT, single_excel_only=CAPITAL_TABLE, verbose=False
    )
    dwelling_stock = dictionary[f"{CAPITAL_TABLE}_Capital_Stock_By_AssetType"][
        DWELLING_CAPITAL_STOCK_ID
    ]
    return dwelling_stock


def _interpolate_annual_to_quarterly(annual: pd.Series) -> pd.Series:
    """Interpolate annual data to quarterly using cubic spline.

    Annual data is end-of-June (Y-JUN), so we interpolate to get
    quarterly values at Q1, Q2, Q3, Q4.

    Args:
        annual: Annual series with Y-JUN frequency

    Returns:
        Quarterly series with Q-DEC frequency

    """
    # Convert to numeric for interpolation
    years = annual.index.year + 0.5  # Mid-year (June)
    values = annual.to_numpy()

    # Create quarterly target dates
    start_year = annual.index.min().year
    end_year = annual.index.max().year
    quarters = pd.period_range(
        start=f"{start_year}Q1", end=f"{end_year}Q4", freq="Q"
    )
    # Quarter midpoints: Q1=0.125, Q2=0.375, Q3=0.625, Q4=0.875
    quarter_numeric = quarters.year + (quarters.quarter - 0.5) / 4

    # Cubic spline interpolation
    spline = interpolate.CubicSpline(years, values, extrapolate=True)
    quarterly_values = spline(quarter_numeric)

    return pd.Series(quarterly_values, index=quarters)


def _backcast_housing_wealth(
    backcast_start: str = "1984Q3", actual_start: str = "1988Q3"
) -> pd.Series:
    """Backcast total housing wealth before 5232.0 data starts.

    Uses annual dwelling capital stock (5204.0) scaled by extrapolated
    land/dwelling ratio. The ratio grew from ~2.1 (1988) to ~3.6 (2024).

    Args:
        backcast_start: Start period for backcast output
        actual_start: Period where actual 5232.0 data begins (excluded from backcast)

    Returns:
        Series with backcasted total housing wealth ($ billions, matching 5232.0)

    """
    # Load and interpolate dwelling stock (in $ millions)
    dwelling_annual = _load_dwelling_capital_stock()
    dwelling_qtr = _interpolate_annual_to_quarterly(dwelling_annual)

    # Calculate time-varying ratio (years before 1988.5 = June 1988)
    # Apply floor of 1.5 (minimum reasonable land/dwelling ratio)
    years_from_1988 = (dwelling_qtr.index.year + (dwelling_qtr.index.quarter - 0.5) / 4) - 1988.5
    ratio = np.maximum(RATIO_1988 + years_from_1988 * RATIO_ANNUAL_GROWTH, 1.5)

    # Apply ratio to get total housing wealth, convert millions → billions
    total_wealth = (dwelling_qtr * ratio) / 1000

    # Filter to backcast period [backcast_start, actual_start)
    start = pd.Period(backcast_start, freq="Q")
    end = pd.Period(actual_start, freq="Q")
    return total_wealth[(total_wealth.index >= start) & (total_wealth.index < end)]


# --- Public Functions ---


def get_housing_wealth_qrtly(include_backcast: bool = True) -> DataSeries:
    """Get total household residential property value (land + dwellings).

    From ABS 5232.0 National Financial Accounts - Household Balance Sheet.
    This is the combined value of residential land and dwelling structures.

    For pre-1988Q3, uses backcasted data from annual dwelling capital stock
    (5204.0) scaled by extrapolated land/dwelling ratio.

    Args:
        include_backcast: If True, include backcasted data from 1984Q3.
            If False, only return actual 5232.0 data (from 1988Q3).

    Returns:
        DataSeries with quarterly housing wealth ($ millions)

    """
    data = _load_household_balance_sheet()
    housing_wealth = data[RESIDENTIAL_LAND_DWELLINGS_ID]

    # Convert to quarterly PeriodIndex if needed
    if isinstance(housing_wealth.index, pd.DatetimeIndex):
        housing_wealth.index = housing_wealth.index.to_period("Q")

    if include_backcast:
        # Get backcast data for 1984Q1-1988Q2 (model starts 1984Q3, need earlier for lags)
        backcast = _backcast_housing_wealth(backcast_start="1984Q1", actual_start="1988Q3")
        # Combine: backcast + actual
        housing_wealth = pd.concat([backcast, housing_wealth])
        housing_wealth = housing_wealth.sort_index()
        source = "ABS + Backcast"
        description = "Household Residential Property Value (Land + Dwellings, backcast pre-1988)"
    else:
        source = "ABS"
        description = "Household Residential Property Value (Land + Dwellings)"

    return DataSeries(
        data=housing_wealth,
        source=source,
        units="$bn",
        description=description,
        cat=WEALTH_CAT,
        table=WEALTH_TABLE,
        series_id=RESIDENTIAL_LAND_DWELLINGS_ID,
    )


def get_housing_wealth_growth_qrtly() -> DataSeries:
    """Get quarterly housing wealth growth (log difference × 100).

    This captures changes in household property values which affect
    consumption through the wealth effect.

    Returns:
        DataSeries with quarterly housing wealth growth (%)

    """
    wealth = get_housing_wealth_qrtly().data
    growth = np.log(wealth).diff() * 100

    return DataSeries(
        data=growth,
        source="Derived",
        units="%",
        description="Quarterly Housing Wealth Growth",
    )


def get_housing_wealth_growth_annual() -> DataSeries:
    """Get annual housing wealth growth (4-quarter log difference × 100).

    Returns:
        DataSeries with annual housing wealth growth (%)

    """
    wealth = get_housing_wealth_qrtly().data
    growth = np.log(wealth).diff(4) * 100

    return DataSeries(
        data=growth,
        source="Derived",
        units="%",
        description="Annual Housing Wealth Growth (4-quarter)",
    )


def get_housing_wealth_growth_lagged_qrtly() -> DataSeries:
    """Get lagged quarterly housing wealth growth.

    Returns:
        DataSeries with lagged quarterly housing wealth growth (%)

    """
    growth = get_housing_wealth_growth_qrtly().data
    growth_lag = growth.shift(1)

    return DataSeries(
        data=growth_lag,
        source="Derived",
        units="%",
        description="Quarterly Housing Wealth Growth (lagged)",
    )


# --- Testing ---

if __name__ == "__main__":
    print("Testing housing wealth loader...\n")

    print("=== Housing Wealth (with backcast) ===")
    wealth = get_housing_wealth_qrtly(include_backcast=True)
    print(f"{wealth}")
    print(f"Date range: {wealth.data.index.min()} to {wealth.data.index.max()}")
    print("\nEarly values ($ billions) - BACKCAST:")
    print(wealth.data.head(8).round(1))
    print("\nSplice point 1988 ($ billions):")
    print(f"  1988Q2: {wealth.data['1988Q2']:.1f} (backcast)")
    print(f"  1988Q3: {wealth.data['1988Q3']:.1f} (actual)")
    print("\nLatest values ($ trillions):")
    print((wealth.data.tail(4) / 1000).round(2))
    print()

    print("=== Housing Wealth (actual only, no backcast) ===")
    wealth_actual = get_housing_wealth_qrtly(include_backcast=False)
    print(f"Date range: {wealth_actual.data.index.min()} to {wealth_actual.data.index.max()}")
    print()

    print("=== Quarterly Growth ===")
    growth_q = get_housing_wealth_growth_qrtly()
    print(f"{growth_q}")
    print(f"Date range: {growth_q.data.dropna().index.min()} to {growth_q.data.index.max()}")
    print(f"Latest values:\n{growth_q.data.tail(8).round(2)}\n")

    print("=== Annual Growth ===")
    growth_a = get_housing_wealth_growth_annual()
    print(f"{growth_a}")
    print(f"Latest values:\n{growth_a.data.tail(8).round(2)}\n")
