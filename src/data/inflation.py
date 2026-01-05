"""Inflation data loading.

Provides CPI trimmed mean and weighted median inflation from ABS (6401.0).
"""

from src.data.abs_loader import load_series
from src.data.dataseries import DataSeries
from src.data.series_specs import (
    CPI_HEADLINE_ANNUAL,
    CPI_TRIMMED_MEAN_ANNUAL,
    CPI_TRIMMED_MEAN_QUARTERLY,
    CPI_WEIGHTED_MEDIAN,
)


def get_trimmed_mean_qrtly() -> DataSeries:
    """Get quarterly trimmed mean inflation.

    Returns percentage change from previous period.

    Returns:
        DataSeries with quarterly inflation (%)

    """
    return load_series(CPI_TRIMMED_MEAN_QUARTERLY)


def get_trimmed_mean_annual() -> DataSeries:
    """Get annual trimmed mean inflation.

    Returns percentage change from corresponding quarter of previous year.

    Returns:
        DataSeries with annual inflation (%)

    """
    return load_series(CPI_TRIMMED_MEAN_ANNUAL)


def get_weighted_median_annual() -> DataSeries:
    """Get annual weighted median inflation.

    Returns percentage change from corresponding quarter of previous year.

    Returns:
        DataSeries with annual weighted median inflation (%)

    """
    return load_series(CPI_WEIGHTED_MEDIAN)


def get_headline_annual() -> DataSeries:
    """Get annual headline CPI inflation.

    Returns percentage change from corresponding quarter of previous year.
    Available from 1949Q3.

    Returns:
        DataSeries with annual headline inflation (%)

    """
    return load_series(CPI_HEADLINE_ANNUAL)
