"""Balance of payments data loading.

Provides quarterly goods and services trade balance from ABS Balance of Payments
and International Investment Position (5302.0). Published ~1 day before GDP,
this captures both goods and services trade on a BoP basis.
"""

import numpy as np

from src.data.abs_loader import ReqsTuple, load_series
from src.data.dataseries import DataSeries

# --- Balance of Payments (5302.0) ---

BOP_GOODS_SERVICES_BALANCE = ReqsTuple(
    cat="5302.0",
    table="530204",
    did="Goods and Services ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

BOP_GOODS_CREDITS = ReqsTuple(
    cat="5302.0",
    table="530204",
    did="Goods and Services credits ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

BOP_GOODS_DEBITS = ReqsTuple(
    cat="5302.0",
    table="530204",
    did="Goods and Services debits ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)


def get_bop_goods_services_balance_qrtly() -> DataSeries:
    """Get balance on goods and services (quarterly, SA, current prices, BoP basis).

    Returns:
        DataSeries with quarterly goods and services balance

    """
    return load_series(BOP_GOODS_SERVICES_BALANCE)


def get_bop_goods_services_change_qrtly() -> DataSeries:
    """Get quarterly change in goods and services balance.

    The change in the trade balance maps more directly to net exports'
    contribution to GDP growth than the balance level.

    Returns:
        DataSeries with quarterly change in goods and services balance ($M)

    """
    balance = get_bop_goods_services_balance_qrtly()
    change = balance.data.diff(1)

    return DataSeries(
        data=change,
        source=balance.source,
        units="$M change",
        description="Change in balance on goods and services (quarterly, BoP basis)",
        series_id=balance.series_id,
        table=balance.table,
        cat=balance.cat,
        stype=balance.stype,
    )


def get_bop_exports_growth_qrtly() -> DataSeries:
    """Get quarterly BoP exports growth (log difference).

    Returns:
        DataSeries with exports growth (% per quarter)

    """
    exports = load_series(BOP_GOODS_CREDITS)
    log_exports = np.log(exports.data) * 100
    growth = log_exports.diff(1)

    return DataSeries(
        data=growth,
        source=exports.source,
        units="% per quarter",
        description="BoP exports of goods and services growth (quarterly, log difference)",
        cat=exports.cat,
        table=exports.table,
    )


def get_bop_imports_growth_qrtly() -> DataSeries:
    """Get quarterly BoP imports growth (log difference).

    Returns:
        DataSeries with imports growth (% per quarter)

    """
    imports = load_series(BOP_GOODS_DEBITS)
    log_imports = np.log(imports.data) * 100
    growth = log_imports.diff(1)

    return DataSeries(
        data=growth,
        source=imports.source,
        units="% per quarter",
        description="BoP imports of goods and services growth (quarterly, log difference)",
        cat=imports.cat,
        table=imports.table,
    )
