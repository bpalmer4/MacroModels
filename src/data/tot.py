"""Terms of Trade data loading.

Provides Terms of Trade from ABS National Accounts (5206.0).
"""

import readabs as ra
from readabs import metacol as mc

from src.data.dataseries import DataSeries

CAT = "5206.0"
TABLE = "5206001_Key_Aggregates"


def get_tot_change_qrtly() -> DataSeries:
    """Get Terms of Trade percentage change (quarterly).

    From ABS 5206.0 Key Aggregates.
    Using pre-computed changes avoids base year issues.

    Returns:
        DataSeries with quarterly ToT percentage change

    """
    data, meta = ra.read_abs_cat(CAT, single_excel_only=TABLE, verbose=False)
    selector = {
        "Terms of trade": mc.did,
        "Percentage changes": mc.did,
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
