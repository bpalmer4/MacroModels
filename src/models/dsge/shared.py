"""Shared utilities for DSGE models.

This module provides common functionality used across different DSGE models:
- Regime definitions for Australian macroeconomic history
- Common data loading patterns

For estimation, see estimation.py which provides ModelSpec and estimate_model().
"""

import numpy as np
import pandas as pd


# =============================================================================
# Regime Definitions
# =============================================================================

# Standard Australian macroeconomic regimes
REGIMES = [
    ("Pre-GFC", "1984Q1", "2008Q3"),
    ("GFC-COVID", "2008Q4", "2020Q4"),
    ("Post-COVID", "2021Q1", None),
]


def get_regime_dates(
    regime_name: str,
    start_override: str | None = None,
) -> tuple[str, str | None]:
    """Get start/end dates for a regime.

    Args:
        regime_name: Name of regime ("Pre-GFC", "GFC-COVID", "Post-COVID")
        start_override: Optional earlier start to use (takes max with regime start)

    Returns:
        (start, end) tuple of period strings
    """
    for name, reg_start, reg_end in REGIMES:
        if name == regime_name:
            if start_override:
                actual_start = max(start_override, reg_start)
            else:
                actual_start = reg_start
            return actual_start, reg_end
    raise ValueError(f"Unknown regime: {regime_name}")


# =============================================================================
# Data Loading Utilities
# =============================================================================

def ensure_period_index(series: pd.Series, freq: str = "Q") -> pd.Series:
    """Ensure series has PeriodIndex.

    Args:
        series: Input series
        freq: Frequency for PeriodIndex

    Returns:
        Series with PeriodIndex
    """
    if not isinstance(series.index, pd.PeriodIndex):
        series = series.copy()
        series.index = pd.PeriodIndex(series.index, freq=freq)
    return series


def filter_date_range(
    df: pd.DataFrame,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Filter DataFrame to date range.

    Args:
        df: DataFrame with PeriodIndex
        start: Start period string
        end: End period string (optional)

    Returns:
        Filtered DataFrame
    """
    start_period = pd.Period(start, freq="Q")
    if end is not None:
        end_period = pd.Period(end, freq="Q")
        return df[(df.index >= start_period) & (df.index <= end_period)]
    return df[df.index >= start_period]
