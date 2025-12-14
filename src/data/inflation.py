"""Inflation data loading.

Provides CPI trimmed mean inflation from ABS (6401.0).
"""

from src.data.abs_loader import load_series
from src.data.dataseries import DataSeries
from src.data.series_specs import (
    CPI_TRIMMED_MEAN_QUARTERLY,
    CPI_TRIMMED_MEAN_ANNUAL,
)


def get_inflation_qrtly() -> DataSeries:
    """Get quarterly trimmed mean inflation.

    Returns percentage change from previous period.

    Returns:
        DataSeries with quarterly inflation (%)
    """
    return load_series(CPI_TRIMMED_MEAN_QUARTERLY)


def get_inflation_annual() -> DataSeries:
    """Get annual trimmed mean inflation.

    Returns percentage change from corresponding quarter of previous year.

    Returns:
        DataSeries with annual inflation (%)
    """
    return load_series(CPI_TRIMMED_MEAN_ANNUAL)
