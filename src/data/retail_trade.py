"""Retail trade data loading.

Provides retail turnover from ABS Monthly Household Spending Indicator (5682.0).
This replaced the former Monthly Retail Trade (8501.0) catalogue.
"""

import numpy as np
import readabs as ra

from src.data.abs_loader import ReqsTuple, load_series
from src.data.dataseries import DataSeries

# --- Monthly Household Spending Indicator (5682.0) ---

RETAIL_TURNOVER = ReqsTuple(
    cat="5682.0",
    table="5682019",
    did="Turnover ;  Total (Industry) ;  Australia ;  Current Price ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)


def get_retail_turnover_monthly() -> DataSeries:
    """Get total retail turnover (monthly, seasonally adjusted, current prices).

    Returns:
        DataSeries with monthly retail turnover

    """
    return load_series(RETAIL_TURNOVER)


def get_retail_turnover_qrtly() -> DataSeries:
    """Get total retail turnover (quarterly sum, seasonally adjusted).

    Returns:
        DataSeries with quarterly retail turnover

    """
    monthly = get_retail_turnover_monthly()
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


def get_retail_growth_qrtly() -> DataSeries:
    """Get quarterly retail turnover growth (log difference).

    Returns:
        DataSeries with retail growth (% per quarter)

    """
    retail = get_retail_turnover_qrtly()
    log_retail = np.log(retail.data) * 100
    growth = log_retail.diff(1)

    return DataSeries(
        data=growth,
        source=retail.source,
        units="% per quarter",
        description="Retail turnover growth (quarterly, log difference)",
        cat=retail.cat,
        table=retail.table,
    )
