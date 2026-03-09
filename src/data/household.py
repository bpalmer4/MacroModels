"""Household sector data loading.

Provides household saving ratio from ABS National Accounts (5206.0).
"""

import readabs as ra
from readabs import metacol as mc

from src.data.dataseries import DataSeries

CAT = "5206.0"
TABLE = "5206001_Key_Aggregates"


def get_saving_ratio_qrtly() -> DataSeries:
    """Get household saving ratio (SA).

    From ABS 5206.0 Key Aggregates.
    Saving ratio = household saving / household disposable income.

    Higher values indicate households are saving more (less consumption).
    Lower values indicate households are spending/borrowing more.

    Returns:
        DataSeries with quarterly household saving ratio (proportion)

    """
    data, meta = ra.read_abs_cat(CAT, single_excel_only=TABLE, verbose=False)
    selector = {
        "Household saving ratio": mc.did,
        "Seasonally Adjusted": mc.stype,
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
        stype="Seasonally Adjusted",
    )


def get_saving_ratio_change_qrtly() -> DataSeries:
    """Get change in household saving ratio (quarterly).

    Positive change = households saving more = demand drag
    Negative change = households saving less (spending/borrowing) = demand boost

    Returns:
        DataSeries with quarterly change in saving ratio (pp)

    """
    saving_ratio = get_saving_ratio_qrtly()
    change = saving_ratio.data.diff(1)

    return DataSeries(
        data=change,
        source="ABS",
        units="pp change",
        description="Change in household saving ratio",
        cat=CAT,
    )
