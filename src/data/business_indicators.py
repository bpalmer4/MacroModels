"""Business indicators data loading.

Provides quarterly business indicators from ABS Quarterly Business Indicators (5676.0).
Includes company profits, inventories, wages, and total sales.

Note: Total sales is not published as a pre-aggregated series. It is constructed
by summing the industry breakdown from table 5676004 (chain volume measures, SA).
"""

import numpy as np
import readabs as ra
from readabs import metacol as mc

from src.data.abs_loader import ReqsTuple, load_series
from src.data.dataseries import DataSeries
from src.data.inflation import get_monthly_trimmed_mean_index

# --- Business Indicators (5676.0) ---

GROSS_OPERATING_PROFITS = ReqsTuple(
    cat="5676.0",
    table="56760015",
    did="Gross Operating Profits ;  Total (State) ;  Total (Industry) ;  Current Price ;  TOTAL (SCP_SCOPE) ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

INVENTORIES_CVM = ReqsTuple(
    cat="5676.0",
    table="5676001",
    did="Inventories ;  Total (State) ;  Total (Industry) ;  Chain Volume Measures ;  TOTAL (SCP_SCOPE) ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

WAGES_TOTAL = ReqsTuple(
    cat="5676.0",
    table="56760017",
    did="Wages ;  Total (State) ;  Total (Industry) ;  Current Price ;  TOTAL (SCP_SCOPE) ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)


def get_company_profits_qrtly() -> DataSeries:
    """Get gross operating profits (quarterly, seasonally adjusted, current prices).

    Returns:
        DataSeries with quarterly company profits

    """
    return load_series(GROSS_OPERATING_PROFITS)


def get_company_profits_growth_qrtly() -> DataSeries:
    """Get quarterly company profits growth (log difference).

    Returns:
        DataSeries with profits growth (% per quarter)

    """
    profits = get_company_profits_qrtly()
    log_profits = np.log(profits.data) * 100
    growth = log_profits.diff(1)

    return DataSeries(
        data=growth,
        source=profits.source,
        units="% per quarter",
        description="Company profits growth (quarterly, log difference)",
        cat=profits.cat,
        table=profits.table,
    )


def get_company_profits_real_qrtly() -> DataSeries:
    """Get gross operating profits deflated by trimmed mean (quarterly, SA).

    Quarterly nominal profits divided by the quarterly mean of the spliced
    monthly trimmed mean index. Trimmed mean is preferred over headline CPI
    to avoid feeding volatile-item noise into the bridge regression.
    """
    nominal = get_company_profits_qrtly()
    monthly_def = get_monthly_trimmed_mean_index().data
    deflator_qrtly = ra.monthly_to_qtly(monthly_def, q_ending="DEC", f="mean")
    common = nominal.data.index.intersection(deflator_qrtly.index)
    real = (nominal.data.loc[common] / deflator_qrtly.loc[common]).dropna()
    return DataSeries(
        data=real,
        source=nominal.source,
        units="Real $ (trimmed mean base year)",
        description="Gross Operating Profits, real (quarterly nominal / quarterly mean trimmed mean), SA",
        series_id=nominal.series_id,
        table=nominal.table,
        cat=nominal.cat,
        stype=nominal.stype,
    )


def get_company_profits_real_growth_qrtly() -> DataSeries:
    """Get quarterly real company profits growth (log diff of trimmed-mean-deflated profits)."""
    real = get_company_profits_real_qrtly()
    log_real = np.log(real.data) * 100
    growth = log_real.diff(1)
    return DataSeries(
        data=growth,
        source=real.source,
        units="% per quarter",
        description="Real company profits growth (quarterly, log difference of trimmed-mean-deflated)",
        cat=real.cat,
        table=real.table,
    )


def get_inventories_qrtly() -> DataSeries:
    """Get inventories (quarterly, seasonally adjusted, chain volume measures).

    Returns:
        DataSeries with quarterly inventories

    """
    return load_series(INVENTORIES_CVM)


def get_inventories_growth_qrtly() -> DataSeries:
    """Get quarterly inventories growth (log difference).

    Returns:
        DataSeries with inventories growth (% per quarter)

    """
    inventories = get_inventories_qrtly()
    log_inv = np.log(inventories.data) * 100
    growth = log_inv.diff(1)

    return DataSeries(
        data=growth,
        source=inventories.source,
        units="% per quarter",
        description="Inventories growth (quarterly, log difference)",
        cat=inventories.cat,
        table=inventories.table,
    )


def get_business_wages_qrtly() -> DataSeries:
    """Get total wages from business indicators (quarterly, seasonally adjusted, current prices).

    Returns:
        DataSeries with quarterly business wages

    """
    return load_series(WAGES_TOTAL)


def get_business_wages_growth_qrtly() -> DataSeries:
    """Get quarterly business wages growth (log difference).

    Returns:
        DataSeries with wages growth (% per quarter)

    """
    wages = get_business_wages_qrtly()
    log_wages = np.log(wages.data) * 100
    growth = log_wages.diff(1)

    return DataSeries(
        data=growth,
        source=wages.source,
        units="% per quarter",
        description="Business wages growth (quarterly, log difference)",
        cat=wages.cat,
        table=wages.table,
    )


def get_business_sales_qrtly() -> DataSeries:
    """Get total sales of goods and services (quarterly, SA, chain volume measures).

    Constructed by summing all industry series from table 5676004, since the ABS
    does not publish a pre-aggregated total.

    Returns:
        DataSeries with quarterly total business sales

    """
    table = "5676004"
    data, meta = ra.read_abs_cat("5676.0", single_excel_only=table, verbose=False)
    meta = meta[meta[mc.table] == table]

    # Select all SA industry sales series
    sa_sales = meta[
        (meta[mc.stype] == "Seasonally Adjusted")
        & (meta[mc.did].str.startswith("Sales ;  Total (State) ;"))
    ]

    # Sum across industries
    total = None
    for series_id in sa_sales[mc.id]:
        s = data[table][series_id]
        total = s if total is None else total.add(s, fill_value=0)

    return DataSeries(
        data=total,
        source="ABS",
        units="$ Millions",
        description="Total sales of goods and services (CVM, SA, summed across industries)",
        cat="5676.0",
        table=table,
    )


def get_business_sales_growth_qrtly() -> DataSeries:
    """Get quarterly total business sales growth (log difference).

    Returns:
        DataSeries with sales growth (% per quarter)

    """
    sales = get_business_sales_qrtly()
    log_sales = np.log(sales.data) * 100
    growth = log_sales.diff(1)

    return DataSeries(
        data=growth,
        source=sales.source,
        units="% per quarter",
        description="Business sales growth (quarterly, log difference)",
        cat=sales.cat,
        table=sales.table,
    )
