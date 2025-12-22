"""Rate conversion utilities for quarterly and annual rates.

Uses compound conversion formulas rather than simple multiplication/division.
For small rates the difference is minor, but compound conversion is more accurate.

Example:
    - Simple: 2.5% annual / 4 = 0.625% quarterly
    - Compound: (1.025)^(1/4) - 1 = 0.619% quarterly
"""

import numpy as np
import pandas as pd


def annualize(quarterly_rate: np.ndarray | pd.Series | pd.DataFrame | float) -> (
    np.ndarray | pd.Series | pd.DataFrame | float
):
    """Convert quarterly rate to annual using compound formula.

    Args:
        quarterly_rate: Rate in percent (e.g., 0.6 for 0.6% quarterly)

    Returns:
        Annual rate in percent (e.g., 2.5 for 2.5% annual)

    """
    return ((1 + quarterly_rate / 100) ** 4 - 1) * 100


def quarterly(annual_rate: np.ndarray | pd.Series | pd.DataFrame | float) -> (
    np.ndarray | pd.Series | pd.DataFrame | float
):
    """Convert annual rate to quarterly using compound formula.

    Args:
        annual_rate: Rate in percent (e.g., 2.5 for 2.5% annual)

    Returns:
        Quarterly rate in percent (e.g., 0.619 for ~0.62% quarterly)

    """
    return ((1 + annual_rate / 100) ** (1 / 4) - 1) * 100
