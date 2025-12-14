"""Time series growth and lag utilities."""

import numpy as np
import pandas as pd


def quarterly_growth(series: pd.Series) -> pd.Series:
    """Calculate quarter-on-quarter growth rate (percentage).

    Args:
        series: pandas Series with quarterly data

    Returns:
        pandas Series with percentage growth rates

    """
    return series.pct_change(periods=1) * 100.0


def annual_growth(series: pd.Series, periods: int = 4) -> pd.Series:
    """Calculate year-on-year growth rate (percentage).

    Args:
        series: pandas Series with quarterly or monthly data
        periods: Number of periods in a year (4 for quarterly, 12 for monthly)

    Returns:
        pandas Series with percentage growth rates

    """
    return series.pct_change(periods=periods) * 100.0


def log_growth(series: pd.Series, periods: int = 1, scale: float = 100.0) -> pd.Series:
    """Calculate log growth rate (log difference).

    Args:
        series: pandas Series
        periods: Number of periods for differencing
        scale: Scaling factor (100 for percentage points)

    Returns:
        pandas Series with log growth rates

    """
    return np.log(series).diff(periods) * scale


def annualize_quarterly(quarterly_rate: pd.Series) -> pd.Series:
    """Convert quarterly growth rate to annualized rate.

    Uses compound growth formula: (1 + r)^4 - 1

    Args:
        quarterly_rate: Quarterly growth rate in percentage points

    Returns:
        Annualized growth rate in percentage points

    """
    return ((1 + quarterly_rate / 100) ** 4 - 1) * 100


def lag(series: pd.Series, periods: int = 1) -> pd.Series:
    """Shift series by specified number of periods.

    Args:
        series: pandas Series
        periods: Number of periods to shift (positive = lag)

    Returns:
        Shifted pandas Series

    """
    return series.shift(periods)


def diff(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate difference over specified periods.

    Args:
        series: pandas Series
        periods: Number of periods for differencing

    Returns:
        Differenced pandas Series

    """
    return series.diff(periods)


def splice_series(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    """Splice two series, favouring primary where both have data.

    Args:
        primary: Series to use where available
        secondary: Series to fill gaps in primary

    Returns:
        Combined series with primary values taking precedence

    Raises:
        ValueError: If indices have different PeriodIndex frequencies

    """
    # Check both have PeriodIndex
    if not isinstance(primary.index, pd.PeriodIndex):
        raise ValueError("Primary series must have PeriodIndex")
    if not isinstance(secondary.index, pd.PeriodIndex):
        raise ValueError("Secondary series must have PeriodIndex")

    # Check same frequency
    if primary.index.freq != secondary.index.freq:
        raise ValueError(
            f"Index frequencies must match: primary={primary.index.freq}, "
            f"secondary={secondary.index.freq}"
        )

    return primary.combine_first(secondary).sort_index()
