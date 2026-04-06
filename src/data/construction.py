"""Construction work done data loading.

Provides quarterly construction activity from ABS Construction Work Done (8755.0).
Published ~3 weeks before GDP, covering building and engineering construction
by sector (private and public).
"""

import numpy as np

from src.data.abs_loader import ReqsTuple, load_series
from src.data.dataseries import DataSeries

# --- Construction Work Done (8755.0) ---

TOTAL_CONSTRUCTION_CVM = ReqsTuple(
    cat="8755.0",
    table="8755001",
    did="Value of work done ;  Chain Volume Measures ;  Total Sectors ;  Total (Type of Construction) ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

PRIVATE_CONSTRUCTION_CVM = ReqsTuple(
    cat="8755.0",
    table="8755001",
    did="Value of work done ;  Chain Volume Measures ;  Private Sector ;  Total (Type of Construction) ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)


def get_total_construction_qrtly() -> DataSeries:
    """Get total construction work done (quarterly, SA, chain volume measures).

    Returns:
        DataSeries with quarterly total construction ($'000)

    """
    return load_series(TOTAL_CONSTRUCTION_CVM)


def get_total_construction_growth_qrtly() -> DataSeries:
    """Get quarterly total construction growth (log difference).

    Returns:
        DataSeries with construction growth (% per quarter)

    """
    construction = get_total_construction_qrtly()
    log_c = np.log(construction.data) * 100
    growth = log_c.diff(1)

    return DataSeries(
        data=growth,
        source=construction.source,
        units="% per quarter",
        description="Total construction work done growth (quarterly, log difference)",
        cat=construction.cat,
        table=construction.table,
    )


def get_private_construction_qrtly() -> DataSeries:
    """Get private sector construction work done (quarterly, SA, chain volume measures).

    Returns:
        DataSeries with quarterly private construction ($'000)

    """
    return load_series(PRIVATE_CONSTRUCTION_CVM)


def get_private_construction_growth_qrtly() -> DataSeries:
    """Get quarterly private construction growth (log difference).

    Returns:
        DataSeries with private construction growth (% per quarter)

    """
    construction = get_private_construction_qrtly()
    log_c = np.log(construction.data) * 100
    growth = log_c.diff(1)

    return DataSeries(
        data=growth,
        source=construction.source,
        units="% per quarter",
        description="Private construction work done growth (quarterly, log difference)",
        cat=construction.cat,
        table=construction.table,
    )
