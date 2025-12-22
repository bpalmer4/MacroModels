"""Household sector data loading.

Provides household saving ratio from ABS National Accounts (5206.0).
"""

import readabs as ra

from src.data.dataseries import DataSeries


def get_saving_ratio_qrtly() -> DataSeries:
    """Get household saving ratio (SA).

    From ABS 5206.0 Key Aggregates (series A2323382F).
    Saving ratio = household saving / household disposable income.

    Higher values indicate households are saving more (less consumption).
    Lower values indicate households are spending/borrowing more.

    Returns:
        DataSeries with quarterly household saving ratio (proportion)

    """
    cat = "5206.0"
    data_dict, _meta = ra.read_abs_cat(cat, single_excel_only="1", verbose=False)

    table = "5206001_Key_Aggregates"
    series_id = "A2323382F"
    series = data_dict[table][series_id]

    return DataSeries(
        data=series,
        source="ABS",
        units="proportion",
        description="Household saving ratio",
        series_id=series_id,
        table=table,
        cat=cat,
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
        series_id=f"{saving_ratio.series_id}_change",
        cat="5206.0",
    )
