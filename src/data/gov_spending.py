"""Government spending data loading.

Provides government final consumption expenditure from ABS National Accounts (5206.0).
"""

import numpy as np
import readabs as ra

from src.data.dataseries import DataSeries
from src.data.gdp import get_gdp_growth


def get_gov_consumption_qrtly() -> DataSeries:
    """Get general government final consumption expenditure (CVM, SA).

    From ABS 5206.0 Expenditure on GDP (Table 2).
    Series A2304080V: General government; Final consumption expenditure (SA)

    Returns:
        DataSeries with quarterly government consumption ($ millions, CVM)

    """
    cat = "5206.0"
    data_dict, _meta = ra.read_abs_cat(cat, single_excel_only="2", verbose=False)

    # Use specific series ID for general government final consumption (SA)
    table = "5206002_Expenditure_Volume_Measures"
    series_id = "A2304080V"
    series = data_dict[table][series_id]

    return DataSeries(
        data=series,
        source="ABS",
        units="$ Millions",
        description="General government final consumption expenditure (CVM)",
        series_id=series_id,
        table=table,
        cat=cat,
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
        series_id=gov.series_id,
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
        series_id=f"{gov_growth.series_id}_impulse",
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
        series_id=impulse.series_id,
        cat=impulse.cat,
    )
