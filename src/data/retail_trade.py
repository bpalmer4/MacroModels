"""Retail trade data loading.

Provides retail turnover from ABS Monthly Household Spending Indicator (5682.0).
This replaced the former Monthly Retail Trade (8501.0) catalogue.

Also provides CPI-deflated (real) variants for use as a real consumption proxy
in the GDP nowcast bridges — see `get_retail_turnover_real_monthly()`.
"""

import numpy as np
import pandas as pd
import readabs as ra

from src.data.abs_loader import ReqsTuple, load_series
from src.data.dataseries import DataSeries
from src.data.inflation import get_monthly_trimmed_mean_index

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


def _deflate_monthly(nominal: pd.Series) -> pd.Series:
    """Divide a monthly nominal series by the spliced monthly trimmed mean index.

    Trimmed mean is preferred over headline CPI: it strips volatile items
    (energy, food, government rebates) that would otherwise propagate as
    deflator noise into the bridge's OOS MSE.
    """
    deflator = get_monthly_trimmed_mean_index().data
    common = nominal.index.intersection(deflator.index)
    return (nominal.loc[common] / deflator.loc[common]).dropna()


def get_retail_turnover_real_monthly() -> DataSeries:
    """Get total retail turnover deflated by monthly trimmed mean (real, monthly, SA).

    Units are nominal $ / deflator-index, i.e. real $ in deflator base-year
    prices. The absolute level is not directly meaningful; this series is
    intended to be aggregated and log-differenced.
    """
    nominal = get_retail_turnover_monthly()
    real = _deflate_monthly(nominal.data)
    return DataSeries(
        data=real,
        source=nominal.source,
        units="Real $ (trimmed mean base year)",
        description="Retail turnover, real (monthly nominal / monthly trimmed mean), SA",
        series_id=nominal.series_id,
        table=nominal.table,
        cat=nominal.cat,
        stype=nominal.stype,
    )


def get_retail_real_growth_qrtly() -> DataSeries:
    """Get quarterly real retail turnover growth (log difference of quarterly sum).

    Returns:
        DataSeries with real retail growth (% per quarter)

    """
    real_monthly = get_retail_turnover_real_monthly()
    quarterly = ra.monthly_to_qtly(real_monthly.data, q_ending="DEC", f="sum")
    log_q = np.log(quarterly) * 100
    growth = log_q.diff(1)
    return DataSeries(
        data=growth,
        source=real_monthly.source,
        units="% per quarter",
        description="Real retail turnover growth (quarterly, log difference of trimmed-mean-deflated)",
        cat=real_monthly.cat,
        table=real_monthly.table,
    )
