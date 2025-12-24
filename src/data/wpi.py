"""Wage Price Index data loading.

Loads WPI from ABS Labour Price Index.
"""

import numpy as np

from src.data.abs_loader import load_series
from src.data.dataseries import DataSeries
from src.data.series_specs import WPI_TOTAL


def get_wpi_growth_qrtly() -> DataSeries:
    """Get quarterly WPI growth.

    Returns log difference of WPI index as growth rate.

    Returns:
        DataSeries with WPI growth (% per quarter)

    """
    wpi = load_series(WPI_TOTAL).data

    log_wpi = np.log(wpi)
    delta_wpi = log_wpi.diff(1) * 100

    return DataSeries(
        data=delta_wpi,
        source="ABS",
        units="% per quarter",
        description="WPI growth (quarterly, log difference)",
        cat="6345.0",
    )
