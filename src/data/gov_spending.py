"""Government spending data loading.

Provides government final consumption expenditure from ABS National Accounts (5206.0).
"""

import numpy as np
import readabs as ra
from readabs import metacol as mc

from src.data.dataseries import DataSeries
from src.data.gdp import get_gdp_growth

CAT = "5206.0"
TABLE = "5206002_Expenditure_Volume_Measures"


def get_gov_consumption_qrtly() -> DataSeries:
    """Get general government final consumption expenditure (CVM, SA).

    From ABS 5206.0 Expenditure on GDP (Table 2).

    Returns:
        DataSeries with quarterly government consumption ($ millions, CVM)

    """
    data, meta = ra.read_abs_cat(CAT, single_excel_only=TABLE, verbose=False)
    selector = {
        "General government": mc.did,
        "Final consumption expenditure": mc.did,
        "Seasonally Adjusted": mc.stype,
    }
    # Exclude National/State breakdowns and percentage changes
    rows = ra.search_abs_meta(meta, selector)
    rows = rows[~rows[mc.did].str.contains("National|State|Percentage|Contribution|Revision", na=False)]
    if len(rows) != 1:
        raise ValueError(f"Expected 1 match for general govt FCE, got {len(rows)}")

    series_id = rows.index[0]
    table = rows[mc.table].iloc[0]
    series = data[table][series_id]
    description = rows[mc.did].iloc[0]

    return DataSeries(
        data=series,
        source="ABS",
        units=rows[mc.unit].iloc[0],
        description=description,
        table=table,
        cat=CAT,
        stype="Seasonally Adjusted",
    )


def get_gov_growth_qrtly() -> DataSeries:
    """Get government consumption growth (log difference, quarterly).

    Returns:
        DataSeries with quarterly government consumption growth

    """
    gov = get_gov_consumption_qrtly()
    log_gov = np.log(gov.data) * 100
    growth = log_gov.diff(1)

    return DataSeries(
        data=growth,
        source=gov.source,
        units="% per quarter",
        description="Government consumption growth (quarterly)",
        table=gov.table,
        cat=gov.cat,
        stype=gov.stype,
    )


def get_fiscal_impulse_qrtly() -> DataSeries:
    """Get fiscal impulse: government growth minus GDP growth.

    Fiscal impulse = Δlog(G) - Δlog(GDP)

    Positive values indicate expansionary fiscal stance (G growing faster than GDP).
    Negative values indicate contractionary fiscal stance (G growing slower than GDP).

    Returns:
        DataSeries with quarterly fiscal impulse

    """
    gov_growth = get_gov_growth_qrtly()
    gdp_growth = get_gdp_growth(periods=1)

    # Align series
    fiscal_impulse = gov_growth.data - gdp_growth.data

    return DataSeries(
        data=fiscal_impulse,
        source="ABS",
        units="% per quarter",
        description="Fiscal impulse (G growth - GDP growth)",
        cat="5206.0",
    )


def get_fiscal_impulse_lagged_qrtly() -> DataSeries:
    """Get lagged fiscal impulse.

    Returns:
        DataSeries with fiscal impulse lagged one quarter

    """
    impulse = get_fiscal_impulse_qrtly()
    impulse_1 = impulse.data.shift(1)

    return DataSeries(
        data=impulse_1,
        source=impulse.source,
        units="% per quarter",
        description="Fiscal impulse lagged one quarter",
        cat=impulse.cat,
    )
