"""Private new capital expenditure data loading.

Provides quarterly private capex from ABS Private New Capital Expenditure and
Expected Expenditure (5625.0). Published ~4 weeks before GDP.
"""

import numpy as np

from src.data.abs_loader import ReqsTuple, load_series
from src.data.dataseries import DataSeries

# --- Private New Capital Expenditure (5625.0) ---

_CAPEX_TABLE = "07_volume_measures_seasonally_adjusted_capex"

# fmt: off
TOTAL_CAPEX_CVM = ReqsTuple(
    cat="5625.0", table=_CAPEX_TABLE,
    did=(
        "Actual Expenditure ;  Total (State) ;"
        "  Total (Type of Asset - Detailed Level) ;"
        "  Chain Volume Measures ;  Total, including Education and Health ;"
    ),
    stype="SA", unit="", seek_yr_growth=False, calc_growth=False, zip_file="",
)

BUILDINGS_CAPEX_CVM = ReqsTuple(
    cat="5625.0", table=_CAPEX_TABLE,
    did=(
        "Actual Expenditure ;  Total (State) ;  Buildings and Structures ;"
        "  Chain Volume Measures ;  Total, including Education and Health ;"
    ),
    stype="SA", unit="", seek_yr_growth=False, calc_growth=False, zip_file="",
)

EQUIPMENT_CAPEX_CVM = ReqsTuple(
    cat="5625.0", table=_CAPEX_TABLE,
    did=(
        "Actual Expenditure ;  Total (State) ;  Equipment, Plant and Machinery ;"
        "  Chain Volume Measures ;  Total, including Education and Health ;"
    ),
    stype="SA", unit="", seek_yr_growth=False, calc_growth=False, zip_file="",
)
# fmt: on


def get_total_capex_qrtly() -> DataSeries:
    """Get total private capex (quarterly, SA, chain volume measures).

    Returns:
        DataSeries with quarterly total private capex

    """
    return load_series(TOTAL_CAPEX_CVM)


def get_total_capex_growth_qrtly() -> DataSeries:
    """Get quarterly total private capex growth (log difference).

    Returns:
        DataSeries with capex growth (% per quarter)

    """
    capex = get_total_capex_qrtly()
    log_c = np.log(capex.data) * 100
    growth = log_c.diff(1)

    return DataSeries(
        data=growth,
        source=capex.source,
        units="% per quarter",
        description="Total private capex growth (quarterly, log difference)",
        cat=capex.cat,
        table=capex.table,
    )
