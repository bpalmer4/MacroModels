"""Import price data loading.

Provides consumption goods import price index from ABS International Trade
Price Indexes (6457.0). Uses consumption goods (Table 3) rather than All groups
(Table 1) because the Phillips curve models a markup over input costs relevant
to consumer prices. All groups includes capital equipment and intermediate
inputs for exports which don't feed into consumer price setting.
Consistent with RBA MARTIN model (RDP 2019-07 s4.7.3; DavidAStephan/MARTIN).
"""

import numpy as np
import readabs as ra
from readabs import metacol as mc

from src.data.dataseries import DataSeries

CAT = "6457.0"
TABLE = "645703"


def get_import_price_index_qrtly() -> DataSeries:
    """Get consumption goods import price index (quarterly).

    From ABS 6457.0 Table 3 - Consumption goods total.

    Returns:
        DataSeries with consumption goods import price index

    """
    data, meta = ra.read_abs_cat(CAT, single_excel_only=TABLE, verbose=False)
    selector = {
        "Index Numbers": mc.did,
        "Consumption goods total": mc.did,
        "Original": mc.stype,
    }
    table, series_id, units = ra.find_abs_id(meta, selector)
    series = data[table][series_id]
    description = meta[meta[mc.id] == series_id][mc.did].iloc[0]

    return DataSeries(
        data=series,
        source="ABS",
        units=units,
        description=description,
        table=table,
        cat=CAT,
        stype="Original",
    )


def get_import_price_growth_annual() -> DataSeries:
    """Get annual import price growth (4-quarter log difference).

    Returns:
        DataSeries with annual import price growth (%)

    """
    index = get_import_price_index_qrtly()
    log_prices = np.log(index.data) * 100
    annual_growth = log_prices.diff(periods=4)

    return DataSeries(
        data=annual_growth,
        source="ABS",
        units="% per year",
        description="Consumption goods import price growth (annual, 4-quarter log difference)",
        cat=index.cat,
    )


def get_import_price_growth_lagged_annual() -> DataSeries:
    """Get lagged annual import price growth.

    Returns:
        DataSeries with annual import price growth lagged one quarter (%)

    """
    growth = get_import_price_growth_annual()
    growth_1 = growth.data.shift(1)

    return DataSeries(
        data=growth_1,
        source=growth.source,
        units="% per year",
        description="Consumption goods import price growth (annual) lagged one quarter",
        cat=growth.cat,
    )
