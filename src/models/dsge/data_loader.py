"""Data loading for DSGE model estimation.

Loads Australian macroeconomic data for estimating the NK DSGE model:
- Output gap (derived from GDP)
- Inflation (CPI trimmed mean)
- Interest rate (RBA cash rate)

The data is transformed to match model requirements:
- Output gap: HP-filtered log GDP deviation from trend
- Inflation: Quarterly rate (annualized)
- Interest rate: Quarterly average, deviation from sample mean
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from src.data.abs_loader import load_series
from src.data.rba_loader import get_cash_rate
from src.data.series_specs import GDP_CVM, CPI_TRIMMED_MEAN_QUARTERLY, COMPENSATION_OF_EMPLOYEES


def hp_filter(y: np.ndarray, lamb: float = 1600) -> tuple[np.ndarray, np.ndarray]:
    """Apply Hodrick-Prescott filter.

    Args:
        y: Time series
        lamb: Smoothing parameter (1600 for quarterly data)

    Returns:
        (trend, cycle) components

    """
    T = len(y)
    # Build second-difference matrix
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1

    # Solve for trend: (I + lamb * D'D) * trend = y
    I = np.eye(T)
    trend = np.linalg.solve(I + lamb * D.T @ D, y)
    cycle = y - trend

    return trend, cycle


def load_estimation_data(
    start: str = "1993Q1",
    end: str | None = None,
    n_observables: int = 2,
) -> pd.DataFrame:
    """Load data for DSGE estimation.

    Args:
        start: Start period (default 1993Q1, when inflation targeting began)
        end: End period (default: latest available)
        n_observables: Number of observables to include
            2: [output_gap, inflation]
            3: [output_gap, inflation, interest_rate]
            4: [output_gap, inflation, interest_rate, wage_inflation]

    Returns:
        DataFrame indexed by quarter

    """
    # Load GDP (chain volume measures)
    gdp_series = load_series(GDP_CVM)
    gdp = gdp_series.data

    # Ensure quarterly frequency
    if not isinstance(gdp.index, pd.PeriodIndex):
        gdp.index = pd.PeriodIndex(gdp.index, freq="Q")

    # Compute output gap via HP filter on log GDP
    log_gdp = np.log(gdp.values)
    trend, cycle = hp_filter(log_gdp, lamb=1600)
    output_gap = pd.Series(cycle * 100, index=gdp.index, name="output_gap")  # Percent

    # Load inflation (trimmed mean, quarterly rate)
    inf_series = load_series(CPI_TRIMMED_MEAN_QUARTERLY)
    inflation = inf_series.data

    if not isinstance(inflation.index, pd.PeriodIndex):
        inflation.index = pd.PeriodIndex(inflation.index, freq="Q")

    # Annualize quarterly inflation: (1 + q/100)^4 - 1
    inflation_annual = ((1 + inflation / 100) ** 4 - 1) * 100
    inflation_annual.name = "inflation"

    # Build DataFrame
    data_dict = {
        "output_gap": output_gap,
        "inflation": inflation_annual,
    }

    # Add interest rate if requested
    if n_observables >= 3:
        cash_rate_series = get_cash_rate()
        cash_rate = cash_rate_series.data

        # Convert to datetime if needed, then resample to quarterly
        if isinstance(cash_rate.index, pd.PeriodIndex):
            cash_rate.index = cash_rate.index.to_timestamp()
        else:
            cash_rate.index = pd.to_datetime(cash_rate.index)

        cash_rate_q = cash_rate.resample("QE").mean()
        cash_rate_q.index = pd.PeriodIndex(cash_rate_q.index, freq="Q")
        cash_rate_q.name = "interest_rate"

        data_dict["interest_rate"] = cash_rate_q

    # Add wage inflation if requested (using Compensation of Employees growth)
    if n_observables >= 4:
        coe_series = load_series(COMPENSATION_OF_EMPLOYEES)
        coe = coe_series.data

        if not isinstance(coe.index, pd.PeriodIndex):
            coe.index = pd.PeriodIndex(coe.index, freq="Q")

        # Year-on-year growth rate
        wage_inflation = coe.pct_change(4) * 100
        wage_inflation.name = "wage_inflation"
        data_dict["wage_inflation"] = wage_inflation

    df = pd.DataFrame(data_dict)

    # Align to common dates
    df = df.dropna()

    # Filter to requested period
    start_period = pd.Period(start, freq="Q")
    if end is not None:
        end_period = pd.Period(end, freq="Q")
        df = df[(df.index >= start_period) & (df.index <= end_period)]
    else:
        df = df[df.index >= start_period]

    # Demean interest rate to match model (which is in deviations)
    if n_observables >= 3:
        df["interest_rate"] = df["interest_rate"] - df["interest_rate"].mean()

    return df


def get_estimation_arrays(
    start: str = "1993Q1",
    end: str | None = None,
    n_observables: int = 2,
) -> tuple[np.ndarray, pd.PeriodIndex]:
    """Get numpy arrays for estimation.

    Args:
        start: Start period
        end: End period
        n_observables: Number of observables (2, 3, or 4)

    Returns:
        (y, dates) where y is (T, n_obs) array

    """
    df = load_estimation_data(start=start, end=end, n_observables=n_observables)
    y = df.values
    dates = df.index

    return y, dates


if __name__ == "__main__":
    print("Loading DSGE estimation data...")

    df = load_estimation_data()
    print(f"\nData range: {df.index[0]} to {df.index[-1]}")
    print(f"Observations: {len(df)}")

    print("\nSummary statistics:")
    print(df.describe())

    print("\nFirst 5 observations:")
    print(df.head())

    print("\nLast 5 observations:")
    print(df.tail())
