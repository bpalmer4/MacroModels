"""Goods trade data loading.

Provides monthly goods trade balance from ABS International Trade in Goods (5368.0).
"""

import readabs as ra

from src.data.abs_loader import ReqsTuple, load_series
from src.data.dataseries import DataSeries

# --- International Trade in Goods (5368.0) ---

GOODS_BALANCE = ReqsTuple(
    cat="5368.0",
    table="536801",
    did="Balance on goods ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

GOODS_CREDITS = ReqsTuple(
    cat="5368.0",
    table="536801",
    did="Credits, Total goods ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

GOODS_DEBITS = ReqsTuple(
    cat="5368.0",
    table="536801",
    did="Debits, Total goods ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)


def get_goods_balance_monthly() -> DataSeries:
    """Get balance on goods (monthly, seasonally adjusted).

    Returns:
        DataSeries with monthly goods trade balance

    """
    return load_series(GOODS_BALANCE)


def get_goods_balance_qrtly() -> DataSeries:
    """Get balance on goods (quarterly sum, seasonally adjusted).

    Returns:
        DataSeries with quarterly goods trade balance

    """
    monthly = get_goods_balance_monthly()
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
