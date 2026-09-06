"""Wage Price Index data loading.

Loads WPI from ABS Labour Price Index.
"""

import numpy as np

from src.data.abs_loader import load_series
from src.data.dataseries import DataSeries
from src.data.series_specs import WPI_PRIVATE, WPI_TOTAL


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


def get_wpi_private_growth_annual() -> DataSeries:
    """Get year-ended private-sector WPI growth.

    Four-quarter log difference of the private-sector index. Private only
    because public-sector wages are administered: see `WPI_PRIVATE` in
    `series_specs` for the correlations that motivate the split.

    The four-quarter rate is used rather than the quarterly one because wage
    setting is annual — award and enterprise agreement increases arrive once a
    year, mostly in the September quarter — so the quarterly rate is dominated
    by that timing pattern rather than by labour market pressure.

    Returns:
        DataSeries with private WPI growth (% per year)

    """
    wpi = load_series(WPI_PRIVATE).data

    growth = np.log(wpi).diff(4) * 100

    return DataSeries(
        data=growth,
        source="ABS",
        units="% per year",
        description="Private-sector WPI growth (year-ended, log difference)",
        cat="6345.0",
    )
