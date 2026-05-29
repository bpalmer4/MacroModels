"""Goods trade data loading.

Provides monthly goods trade balance from ABS International Trade in Goods (5368.0).
Also provides a CPI-deflated (real) variant for use in the GDP nowcast bridges,
to strip out general price inflation from the nominal $ balance.
"""

import readabs as ra

from src.data.abs_loader import ReqsTuple, load_series
from src.data.dataseries import DataSeries
from src.data.inflation import get_monthly_trimmed_mean_index

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


def get_goods_balance_real_monthly() -> DataSeries:
    """Get monthly goods trade balance deflated by monthly trimmed mean.

    Strips out underlying inflation from the nominal $ balance. Not a perfect
    deflation — conceptually you'd deflate exports and imports by their own
    price indices separately — but those are quarterly-only, so the spliced
    monthly trimmed mean is the available proxy. Trimmed mean is preferred
    over headline CPI to avoid feeding volatile-item noise into the bridge.
    """
    nominal = get_goods_balance_monthly()
    deflator = get_monthly_trimmed_mean_index().data
    common = nominal.data.index.intersection(deflator.index)
    real = (nominal.data.loc[common] / deflator.loc[common]).dropna()
    return DataSeries(
        data=real,
        source=nominal.source,
        units="Real $ (trimmed mean base year)",
        description="Balance on goods, real (monthly nominal / monthly trimmed mean), SA",
        series_id=nominal.series_id,
        table=nominal.table,
        cat=nominal.cat,
        stype=nominal.stype,
    )


def get_goods_balance_real_qrtly() -> DataSeries:
    """Get balance on goods, real (quarterly sum of trimmed-mean-deflated monthly balance)."""
    real_monthly = get_goods_balance_real_monthly()
    quarterly = ra.monthly_to_qtly(real_monthly.data, q_ending="DEC", f="sum")
    return DataSeries(
        data=quarterly,
        source=real_monthly.source,
        units=real_monthly.units,
        description="Balance on goods, real (quarterly sum, trimmed-mean-deflated)",
        series_id=real_monthly.series_id,
        table=real_monthly.table,
        cat=real_monthly.cat,
        stype=real_monthly.stype,
    )
