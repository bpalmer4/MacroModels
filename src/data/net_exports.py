"""Net exports data loading.

Provides exports, imports, and net exports from National Accounts (5206.0).
"""

import numpy as np
import pandas as pd

from src.data.abs_loader import ReqsTuple, load_series
from src.data.dataseries import DataSeries


# --- National Accounts (5206.0) - Expenditure components ---

EXPORTS_CVM = ReqsTuple(
    cat="5206.0",
    table="5206002_Expenditure_Volume_Measures",
    did="Exports of goods and services ;",
    stype="SA",
    unit="$ Millions",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

IMPORTS_CVM = ReqsTuple(
    cat="5206.0",
    table="5206002_Expenditure_Volume_Measures",
    did="Imports of goods and services ;",
    stype="SA",
    unit="$ Millions",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

GDP_CVM = ReqsTuple(
    cat="5206.0",
    table="5206001_Key_Aggregates",
    did="Gross domestic product: Chain volume measures ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)


def get_exports_qrtly() -> DataSeries:
    """Get exports of goods and services (chain volume measures).

    Returns:
        DataSeries with quarterly exports (CVM)

    """
    return load_series(EXPORTS_CVM)


def get_imports_qrtly() -> DataSeries:
    """Get imports of goods and services (chain volume measures).

    Returns:
        DataSeries with quarterly imports (CVM)

    """
    return load_series(IMPORTS_CVM)


def get_net_exports_ratio_qrtly() -> DataSeries:
    """Get net exports as a ratio to GDP.

    NX/Y = (Exports - Imports) / GDP

    Returns:
        DataSeries with net exports ratio (%)

    """
    exports = load_series(EXPORTS_CVM).data
    imports = load_series(IMPORTS_CVM).data
    gdp = load_series(GDP_CVM).data

    # Align indices
    common_idx = exports.index.intersection(imports.index).intersection(gdp.index)
    exports = exports.loc[common_idx]
    imports = imports.loc[common_idx]
    gdp = gdp.loc[common_idx]

    nx_ratio = ((exports - imports) / gdp) * 100

    return DataSeries(
        data=nx_ratio,
        source="ABS",
        units="% of GDP",
        description="Net exports as ratio to GDP",
        cat="5206.0",
        table="5206001_Key_Aggregates",
    )


def get_net_exports_ratio_change_qrtly() -> DataSeries:
    """Get quarterly change in net exports ratio.

    Returns:
        DataSeries with Δ(NX/Y) (percentage points)

    """
    nx_ratio = get_net_exports_ratio_qrtly()
    delta_nx = nx_ratio.data.diff(1)

    return DataSeries(
        data=delta_nx,
        source=nx_ratio.source,
        units="pp",
        description="Change in net exports ratio (quarterly)",
        cat=nx_ratio.cat,
        table=nx_ratio.table,
    )


def get_export_growth_qrtly() -> DataSeries:
    """Get quarterly export growth (log difference).

    Returns:
        DataSeries with export growth (% per quarter)

    """
    exports = get_exports_qrtly()
    log_exports = np.log(exports.data) * 100
    export_growth = log_exports.diff(1)

    return DataSeries(
        data=export_growth,
        source=exports.source,
        units="% per quarter",
        description="Export growth (quarterly, log difference)",
        cat=exports.cat,
        table=exports.table,
    )


def get_import_growth_qrtly() -> DataSeries:
    """Get quarterly import growth (log difference).

    Returns:
        DataSeries with import growth (% per quarter)

    """
    imports = get_imports_qrtly()
    log_imports = np.log(imports.data) * 100
    import_growth = log_imports.diff(1)

    return DataSeries(
        data=import_growth,
        source=imports.source,
        units="% per quarter",
        description="Import growth (quarterly, log difference)",
        cat=imports.cat,
        table=imports.table,
    )
