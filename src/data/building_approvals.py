"""Building approvals data loading.

Provides dwelling approvals from ABS Monthly Building Approvals (8731.0).
"""

import numpy as np
import readabs as ra

from src.data.abs_loader import ReqsTuple, load_series
from src.data.dataseries import DataSeries

# --- Building Approvals (8731.0) ---

DWELLING_APPROVALS_TOTAL = ReqsTuple(
    cat="8731.0",
    table="8731006",
    did="Total number of dwelling units ;  Total (Type of Building) ;  Total Sectors ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)


def get_building_approvals_monthly() -> DataSeries:
    """Get total dwelling approvals (monthly, seasonally adjusted).

    Returns:
        DataSeries with monthly dwelling approvals

    """
    return load_series(DWELLING_APPROVALS_TOTAL)


def get_building_approvals_qrtly() -> DataSeries:
    """Get total dwelling approvals (quarterly sum, seasonally adjusted).

    Returns:
        DataSeries with quarterly dwelling approvals

    """
    monthly = get_building_approvals_monthly()
    quarterly = ra.monthly_to_qtly(monthly.data, q_ending="DEC", f="sum")

    return DataSeries(
        data=quarterly,
        source=monthly.source,
        units=monthly.units,
        description=f"{monthly.description} (quarterly sum)",
        series_id=monthly.series_id,
        table=monthly.table,
        cat=monthly.cat,
        stype=monthly.stype,
    )


def get_building_approvals_growth_qrtly() -> DataSeries:
    """Get quarterly dwelling approvals growth (log difference).

    Returns:
        DataSeries with approvals growth (% per quarter)

    """
    approvals = get_building_approvals_qrtly()
    log_approvals = np.log(approvals.data) * 100
    growth = log_approvals.diff(1)

    return DataSeries(
        data=growth,
        source=approvals.source,
        units="% per quarter",
        description="Building approvals growth (quarterly, log difference)",
        cat=approvals.cat,
        table=approvals.table,
    )
