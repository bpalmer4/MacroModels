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


import numpy as np
import pandas as pd

from src.data.abs_loader import load_series
from src.data.cash_rate import get_cash_rate_qrtly
from src.data.series_specs import (
    COMPENSATION_OF_EMPLOYEES,
    CPI_TRIMMED_MEAN_QUARTERLY,
    GDP_CVM,
    UNEMPLOYMENT_RATE,
)

# Inflation targeting parameters
PI_TARGET = 2.5  # RBA target midpoint
PI_TARGET_START = "1993Q1"  # Inflation targeting introduced
PI_TARGET_FULL = "1998Q1"  # Full credibility assumed
ALPHA_POST_TARGET = 0.2  # Residual backward-looking component after credibility


def compute_inflation_anchor(
    inflation: pd.Series,
    target: float = PI_TARGET,
    target_start: str = PI_TARGET_START,
    target_full: str = PI_TARGET_FULL,
    alpha_post: float = ALPHA_POST_TARGET,
) -> pd.Series:
    """Compute inflation anchor with regime-dependent expectations.

    The anchor transitions from backward-looking to target-anchored:
    - Pre-target_start: α = 1.0 (fully backward-looking)
    - target_start to target_full: α fades linearly from 1.0 to alpha_post
    - Post-target_full: α = alpha_post (mostly anchored to target)

    Formula: π_anchor = α × π_{t-1} + (1 - α) × target

    All values in annual percent terms.

    Args:
        inflation: Annualized inflation series with PeriodIndex
        target: Inflation target (annual %, default 2.5)
        target_start: When targeting began (default 1993Q1)
        target_full: When full credibility achieved (default 1998Q1)
        alpha_post: Backward-looking weight after credibility (default 0.2)

    Returns:
        Inflation anchor series

    """
    start_period = pd.Period(target_start, freq="Q")
    full_period = pd.Period(target_full, freq="Q")

    # Compute alpha for each period
    alpha = pd.Series(index=inflation.index, dtype=float)

    for period in inflation.index:
        if period < start_period:
            # Pre-targeting: fully backward-looking
            alpha[period] = 1.0
        elif period >= full_period:
            # Post-credibility: mostly anchored
            alpha[period] = alpha_post
        else:
            # Transition: linear fade from 1.0 to alpha_post
            n_transition = (full_period - start_period).n
            periods_in = (period - start_period).n
            alpha[period] = 1.0 - (1.0 - alpha_post) * (periods_in / n_transition)

    # Compute anchor: α × π_{t-1} + (1 - α) × target
    pi_lag = inflation.shift(1)
    anchor = alpha * pi_lag + (1 - alpha) * target

    # Fill first value with target (no lag available)
    anchor = anchor.fillna(target)

    anchor.name = "pi_anchor"
    return anchor


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
    anchor_inflation: bool = False,
) -> pd.DataFrame:
    """Load data for DSGE estimation.

    Args:
        start: Start period (default 1993Q1, when inflation targeting began)
        end: End period (default: latest available)
        n_observables: Number of observables to include
            2: [output_gap, inflation]
            3: [output_gap, inflation, interest_rate]
            4: [output_gap, inflation, interest_rate, wage_inflation]
            5: [output_gap, inflation, interest_rate, wage_inflation, u_gap]
        anchor_inflation: If True, use (π - π_anchor) instead of π
            where π_anchor = α×π_{t-1} + (1-α)×2.5
            with α transitioning from 1.0 (pre-1993) to 0.2 (post-1998)

    Returns:
        DataFrame indexed by quarter

    """
    # Load GDP (chain volume measures)
    gdp_series = load_series(GDP_CVM)
    gdp = gdp_series.data

    # Ensure quarterly frequency
    if not isinstance(gdp.index, pd.PeriodIndex):
        gdp.index = pd.PeriodIndex(gdp.index, freq="Q")

    # Output gap via HP filter on log GDP - use lower λ for less smoothing
    log_gdp = np.log(gdp.to_numpy())
    trend, cycle = hp_filter(log_gdp, lamb=400)  # λ=400 vs standard 1600
    output_gap = pd.Series(cycle * 100, index=gdp.index, name="output_gap")

    # Load inflation (trimmed mean, quarterly rate)
    inf_series = load_series(CPI_TRIMMED_MEAN_QUARTERLY)
    inflation = inf_series.data

    if not isinstance(inflation.index, pd.PeriodIndex):
        inflation.index = pd.PeriodIndex(inflation.index, freq="Q")

    # Annualize quarterly inflation: (1 + q/100)^4 - 1
    inflation_annual = ((1 + inflation / 100) ** 4 - 1) * 100
    inflation_annual.name = "inflation"

    # Compute anchor-adjusted inflation if requested
    if anchor_inflation:
        pi_anchor = compute_inflation_anchor(inflation_annual)
        inflation_adjusted = inflation_annual - pi_anchor
        inflation_adjusted.name = "inflation"  # π - π_anchor for Phillips curve
    else:
        inflation_adjusted = inflation_annual

    # Build DataFrame
    data_dict = {
        "output_gap": output_gap,
        "inflation": inflation_adjusted,
    }

    # Add interest rate if requested (spliced OCR + historical interbank rate)
    if n_observables >= 3:
        cash_rate_series = get_cash_rate_qrtly()
        cash_rate_q = cash_rate_series.data
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

    # Add unemployment gap if requested (HP-filtered unemployment rate)
    if n_observables >= 5:
        ur_series = load_series(UNEMPLOYMENT_RATE)
        ur = ur_series.data

        if not isinstance(ur.index, pd.PeriodIndex):
            ur.index = pd.PeriodIndex(ur.index, freq="Q")

        # Convert monthly to quarterly if needed
        if ur.index.freqstr == "M":
            ur = ur.resample("Q").mean()
            ur.index = pd.PeriodIndex(ur.index, freq="Q")

        # Unemployment gap via HP filter - lower λ for less smoothing
        ur_trend, ur_cycle = hp_filter(ur.to_numpy(), lamb=400)  # λ=400 vs standard 1600
        u_gap = pd.Series(ur_cycle, index=ur.index, name="u_gap")

        data_dict["u_gap"] = u_gap

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
    anchor_inflation: bool = False,
) -> tuple[np.ndarray, pd.PeriodIndex]:
    """Get numpy arrays for estimation.

    Args:
        start: Start period
        end: End period
        n_observables: Number of observables (2, 3, 4, or 5)
        anchor_inflation: If True, use (π - π_anchor) instead of π

    Returns:
        (y, dates) where y is (T, n_obs) array

    """
    df = load_estimation_data(
        start=start, end=end, n_observables=n_observables, anchor_inflation=anchor_inflation
    )
    y = df.to_numpy()
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
