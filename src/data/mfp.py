"""Multi-factor productivity data loading.

Provides MFP from ABS Estimates of Industry Multifactor Productivity (5204.0).
"""

from src.data.abs_loader import load_series
from src.data.dataseries import DataSeries
from src.data.series_specs import MFP_HOURS_WORKED


def get_mfp_annual() -> DataSeries:
    """Get multi-factor productivity growth from ABS (annual).

    Returns MFP percentage changes from ABS 5204.0 Productivity tables.
    This is annual data - models may need to convert to quarterly frequency.

    Returns:
        DataSeries with annual MFP growth (%)

    """
    return load_series(MFP_HOURS_WORKED)
