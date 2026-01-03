"""Derived productivity measures from wage and national accounts data.

Calculates labour productivity and MFP using identities from wage data:
- Labour Productivity = Δhcoe - Δulc (from ULC = HCOE / LP identity)
- MFP = LP - α × capital_deepening (Solow residual)

These are algebraically derived from data, not estimated by the model.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from src.data.dataseries import DataSeries

HP_LAMBDA = 1600  # Hodrick-Prescott smoothing parameter for quarterly data


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
