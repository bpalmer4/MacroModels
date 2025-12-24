"""Domestic Final Demand deflator data loading.

Loads the DFD implicit price deflator from ABS National Accounts.
"""

import numpy as np

from src.data.abs_loader import load_series
from src.data.dataseries import DataSeries
from src.data.series_specs import DFD_DEFLATOR


def get_dfd_deflator_growth_annual() -> DataSeries:
    """Get annual growth in the DFD implicit price deflator.

    Returns year-ended log difference of the ABS implicit price deflator.

    Returns:
        DataSeries with DFD deflator growth (% per year, year-ended)

    """
    deflator = load_series(DFD_DEFLATOR).data

    log_deflator = np.log(deflator)
    # Year-ended growth (4 quarters)
    delta_deflator = log_deflator.diff(4) * 100

    return DataSeries(
        data=delta_deflator,
        source="ABS",
        units="% per year",
        description="DFD implicit price deflator growth (year-ended)",
        cat="5206.0",
    )
