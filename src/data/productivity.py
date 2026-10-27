"""Derived productivity measures from wage and national accounts data.

Calculates labour productivity and MFP using identities from wage data:
- Labour Productivity = Δhcoe - Δulc (from ULC = HCOE / LP identity)
- MFP = LP - α × capital_deepening (Solow residual)

And one measured directly, without the wage identity:
- GDP per hour worked, as a trailing trend growth rate, optionally with the
  COVID swing flattened out of the index

These are algebraically derived from data, not estimated by the model.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from src.data.abs_loader import load_series
from src.data.dataseries import DataSeries
from src.data.henderson import hma
from src.data.series_specs import GDP_CVM, HOURS_WORKED_INDEX

HP_LAMBDA = 1600  # Hodrick-Prescott smoothing parameter for quarterly data

QUARTERS_PER_YEAR = 4

# COVID lifted GDP per hour (composition: low-paid hours went first) and took it
# back. The log index was back to its 2019Q4 level by 2023Q1, and the quarters
# between are replaced by a straight line joining those two.
COVID_FLAT_FROM = "2019Q4"
COVID_FLAT_TO = "2023Q1"


def get_labour_productivity_growth(
    ulc_growth: pd.Series,
    hcoe_growth: pd.Series,
) -> DataSeries:
    """Derive labour productivity growth from wage data.

    From the identity: ULC = HCOE / LP
    Therefore: Δlp = Δhcoe - Δulc

    Args:
        ulc_growth: Unit labour cost growth (quarterly, %)
        hcoe_growth: Hourly COE growth (quarterly, %)

    Returns:
        DataSeries with derived labour productivity growth

    """
    labour_productivity = hcoe_growth - ulc_growth

    return DataSeries(
        data=labour_productivity,
        source="Derived",
        units="% per quarter",
        description="Labour productivity growth (derived: Δhcoe - Δulc)",
        cat="Derived from ABS 5206.0",
    )


def get_mfp_growth(
    ulc_growth: pd.Series,
    hcoe_growth: pd.Series,
    capital_growth: pd.Series,
    hours_growth: pd.Series,
    alpha: float | pd.Series = 0.3,
) -> DataSeries:
    """Derive MFP growth from wage data using Solow residual identity.

    Labour productivity growth = Δhcoe - Δulc (from wage equation identity)
    MFP growth = labour_productivity - α × capital_deepening

    Where capital_deepening = g_K - g_L (capital growth minus hours growth)

    Args:
        ulc_growth: Unit labour cost growth (quarterly)
        hcoe_growth: Hourly COE growth (quarterly)
        capital_growth: Capital stock growth (quarterly, smoothed)
        hours_growth: Hours worked growth (quarterly)
        alpha: Capital share (default 0.3), can be time-varying Series

    Returns:
        DataSeries with derived MFP growth (unfloored)

    """
    labour_productivity = hcoe_growth - ulc_growth
    capital_deepening = capital_growth - hours_growth
    mfp_growth = labour_productivity - alpha * capital_deepening

    alpha_desc = "time-varying" if isinstance(alpha, pd.Series) else f"{alpha}"
    return DataSeries(
        data=mfp_growth,
        source="Derived",
        units="% per quarter",
        description=f"MFP growth (Solow residual, α={alpha_desc})",
        cat="Derived from ABS 5206.0",
    )


def compute_mfp_trend_floored(
    ulc_growth: pd.Series,
    hcoe_growth: pd.Series,
    capital_growth: pd.Series,
    hours_growth: pd.Series,
    *,
    alpha: float | pd.Series = 0.3,
    hp_lambda: int = HP_LAMBDA,
) -> DataSeries:
    """Derive MFP trend growth, HP-filtered and floored at zero.

    This is the version used as model input for potential output.
    Flooring at zero reflects that negative MFP during recessions is
    capacity underutilization, not true technological regress.

    Args:
        ulc_growth: Unit labour cost growth (quarterly)
        hcoe_growth: Hourly COE growth (quarterly)
        capital_growth: Capital stock growth (quarterly, smoothed)
        hours_growth: Hours worked growth (quarterly)
        alpha: Capital share (default 0.3), can be time-varying Series
        hp_lambda: HP filter smoothing parameter (default 1600)

    Returns:
        DataSeries with HP-filtered, floored MFP trend growth

    """
    # Get raw MFP
    mfp_raw = get_mfp_growth(
        ulc_growth, hcoe_growth, capital_growth, hours_growth, alpha
    ).data

    # HP filter to extract smooth trend
    mfp_clean = mfp_raw.dropna()
    _, mfp_trend = hpfilter(mfp_clean, lamb=hp_lambda)

    # Floor at zero: true technological progress doesn't go negative
    mfp_trend = np.maximum(mfp_trend, 0)

    mfp_floored = pd.Series(mfp_trend, index=mfp_clean.index).reindex(mfp_raw.index)

    alpha_desc = "time-varying" if isinstance(alpha, pd.Series) else f"{alpha}"
    return DataSeries(
        data=mfp_floored,
        source="Derived",
        units="% per quarter",
        description=f"MFP trend growth (HP λ={hp_lambda}, floored at zero, α={alpha_desc})",
        cat="Derived from ABS 5206.0",
    )


def compute_mfp_trend_hma(
    ulc_growth: pd.Series,
    hcoe_growth: pd.Series,
    capital_growth: pd.Series,
    hours_growth: pd.Series,
    *,
    alpha: float | pd.Series = 0.3,
    hma_term: int = 51,
) -> DataSeries:
    """Derive MFP trend growth using Henderson moving average.

    HMA(51) on quarterly data is comparable to HMA(13) on annual data,
    stripping out cyclical noise while avoiding HP filter end-point bias.
    Zero floor applied: negative trend MFP is implausible and arises from
    endpoint asymmetry when the sample is short relative to the filter.

    Args:
        ulc_growth: Unit labour cost growth (quarterly)
        hcoe_growth: Hourly COE growth (quarterly)
        capital_growth: Capital stock growth (quarterly, smoothed)
        hours_growth: Hours worked growth (quarterly)
        alpha: Capital share (default 0.3), can be time-varying Series
        hma_term: Henderson MA term (default 51, must be odd)

    Returns:
        DataSeries with HMA-smoothed MFP trend growth (floored at zero)

    """
    # Get raw MFP
    mfp_raw = get_mfp_growth(
        ulc_growth, hcoe_growth, capital_growth, hours_growth, alpha
    ).data

    # HMA smooth then floor at zero
    mfp_clean = mfp_raw.dropna()
    mfp_trend = hma(mfp_clean, hma_term)
    mfp_trend = mfp_trend.clip(lower=0.0)
    mfp_smoothed = mfp_trend.reindex(mfp_raw.index)

    alpha_desc = "time-varying" if isinstance(alpha, pd.Series) else f"{alpha}"
    return DataSeries(
        data=mfp_smoothed,
        source="Derived",
        units="% per quarter",
        description=f"MFP trend growth (HMA {hma_term}, α={alpha_desc})",
        cat="Derived from ABS 5206.0",
    )


def get_real_wage_gap(
    ulc_growth: pd.Series,
    mfp_growth: pd.Series,
) -> DataSeries:
    """Compute real wage gap: ULC growth minus MFP growth.

    When wages grow faster than productivity, real labour costs rise,
    leading firms to reduce hiring (used in employment equation).

    Args:
        ulc_growth: Unit labour cost growth (quarterly)
        mfp_growth: MFP growth (quarterly, trend)

    Returns:
        DataSeries with real wage gap (% per quarter)

    """
    real_wage_gap = ulc_growth - mfp_growth

    return DataSeries(
        data=real_wage_gap,
        source="Derived",
        units="% per quarter",
        description="Real wage gap (Δulc - MFP trend)",
        cat="Derived from ABS 5206.0",
    )


def get_gdp_per_hour_trend_growth(years: int = 10) -> DataSeries:
    """Trend growth in GDP per hour worked, over a trailing window.

    The change in log GDP per hour over the window, per year: the same as the
    average quarterly growth across it. Trailing, so no end-point problem, and a
    swing that reverses inside the window cancels, as COVID's composition jump
    in productivity and its unwind do once both are in it.

    Measured directly, not through the wage identity (Δhcoe - Δulc): JobKeeper
    payments ran through compensation of employees, and would show up in a
    productivity backed out of it.

    Args:
        years: Length of the trailing window (default 10)

    Returns:
        DataSeries with trend productivity growth (% per year, log)

    """
    window = years * QUARTERS_PER_YEAR
    trend = _log_gdp_per_hour().diff(window) / years

    return DataSeries(
        data=trend,
        source="Derived",
        units="% per year",
        description=f"GDP per hour worked, trend growth (trailing {years} years)",
        cat="Derived from ABS 5206.0",
    )


def get_gdp_per_hour_trend_growth_ex_covid(years: int = 5) -> DataSeries:
    """Trend growth in GDP per hour worked, with the COVID swing flattened out.

    The log index is joined by a straight line from 2019Q4 to 2023Q1, the two
    quarters at which it stood at the same level, before the trailing change is
    taken. That lets a window shorter than the swing's reach be used: without
    it, a five-year window would straddle the rise or the unwind until 2028.

    The change is between annual averages of the level (the latest four
    quarters against the same four a window earlier), not single quarters: a
    short window divides end-point noise by few years. Trailing, so a published
    value never revises, unlike a Henderson or HP trend at the sample end.

    Args:
        years: Length of the trailing window (default 5)

    Returns:
        DataSeries with trend productivity growth (% per year, log)

    """
    log_per_hour = _log_gdp_per_hour().copy()
    start, end = pd.Period(COVID_FLAT_FROM, freq="Q"), pd.Period(COVID_FLAT_TO, freq="Q")
    log_per_hour.loc[(log_per_hour.index > start) & (log_per_hour.index < end)] = np.nan
    flat = log_per_hour.interpolate(method="linear", limit_area="inside")
    annual = flat.rolling(QUARTERS_PER_YEAR).mean()
    trend = annual.diff(years * QUARTERS_PER_YEAR) / years

    return DataSeries(
        data=trend,
        source="Derived",
        units="% per year",
        description=(f"GDP per hour worked, trend growth (trailing {years} years of annual averages, "
                     f"flat {COVID_FLAT_FROM}-{COVID_FLAT_TO})"),
        cat="Derived from ABS 5206.0",
    )


def _log_gdp_per_hour() -> pd.Series:
    """Log of real GDP per hour worked (x100), quarterly."""
    gdp = load_series(GDP_CVM).data
    hours = load_series(HOURS_WORKED_INDEX).data
    return np.log(gdp / hours) * 100
