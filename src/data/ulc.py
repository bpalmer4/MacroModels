"""Unit labour cost data loading.

Calculates ULC from National Accounts components.
"""

import numpy as np

from src.data.abs_loader import get_abs_data
from src.data.dataseries import DataSeries
from src.data.series_specs import COMPENSATION_OF_EMPLOYEES, GDP_CVM


def get_ulc_growth_qrtly() -> DataSeries:
    """Get quarterly unit labour cost growth.

    Calculates ULC = Compensation of Employees / GDP (CVM),
    then returns log difference as growth rate.

    Returns:
        DataSeries with ULC growth (% per quarter)

    """
    data = get_abs_data({
        "GDP": GDP_CVM,
        "CoE": COMPENSATION_OF_EMPLOYEES,
    })

    gdp = data["GDP"].data
    coe = data["CoE"].data

    ulc = coe / gdp
    log_ulc = np.log(ulc)
    delta_ulc = log_ulc.diff(1) * 100

    return DataSeries(
        data=delta_ulc,
        source="ABS",
        units="% per quarter",
        description="Unit labour cost growth (quarterly, log difference)",
        cat="5206.0",
    )
