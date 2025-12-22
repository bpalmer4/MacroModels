"""Terms of Trade data loading.

Provides Terms of Trade from ABS National Accounts (5206.0).
"""

import readabs as ra

from src.data.dataseries import DataSeries


def get_tot_change_qrtly() -> DataSeries:
    """Get Terms of Trade percentage change (quarterly).

    From ABS 5206.0 Key Aggregates (series A2302458X).
    Using pre-computed changes avoids base year issues.

    Returns:
        DataSeries with quarterly ToT percentage change

    """
    data_dict, _meta = ra.read_abs_cat("5206.0", single_excel_only="1", verbose=False)
    series = data_dict["5206001_Key_Aggregates"]["A2302458X"]

    return DataSeries(
        data=series,
        source="ABS",
        units="% change",
        description="Terms of Trade (quarterly percentage change)",
        series_id="A2302458X",
        cat="5206.0",
    )
