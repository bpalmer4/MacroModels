"""GDP data loading.

Fetches Australian GDP from ABS National Accounts (5206.0).
"""

import numpy as np
import readabs as ra
from readabs import metacol as mc

from src.data.dataseries import DataSeries


def get_gdp(gdp_type: str = "CVM", seasonal: str = "SA") -> DataSeries:
    """Fetch ABS GDP Series from the key aggregates table.

    Args:
        gdp_type: Type of series - "CP" (Current price) or "CVM" (Chain volume measures)
        seasonal: Seasonal adjustment - "SA", "T" (Trend), or "O" (Original)

    Returns:
        DataSeries containing GDP data and metadata
    """
    did_cvm = "Gross domestic product: Chain volume measures ;"
    did_cp = "Gross domestic product: Current prices ;"
    gdp_types = {
        "CP": did_cp,
        "Current price": did_cp,
        "Current prices": did_cp,
        "CVM": did_cvm,
        "Volumetric": did_cvm,
    }
    seasonals = {
        "SA": "Seasonally Adjusted",
        "S": "Seasonally Adjusted",
        "T": "Trend",
        "O": "Original",
    }
    if gdp_type not in gdp_types:
        raise ValueError(f"Invalid GDP type: {gdp_type}")
    if seasonal not in seasonals:
        raise ValueError(f"Invalid seasonal adjustment type: {seasonal}")

    cat = "5206.0"
    seo = "5206001_Key_Aggregates"
    gdp_data, gdp_meta = ra.read_abs_cat(cat, single_excel_only=seo, verbose=False)
    selector = {
        gdp_types[gdp_type]: mc.did,
        seasonals[seasonal]: mc.stype,
    }
    table, series_id, units = ra.find_abs_id(gdp_meta, selector, verbose=False)
    gdp = gdp_data[table][series_id]

    return DataSeries(
        data=gdp,
        source="ABS",
        units=units,
        description=gdp_types[gdp_type].rstrip(" ;"),
        series_id=series_id,
        table=table,
        cat=cat,
        stype=seasonals[seasonal],
    )


def get_log_gdp() -> DataSeries:
    """Load GDP as log levels (scaled by 100).

    Returns:
        DataSeries with log GDP scaled for numerical stability
    """
    gdp = get_gdp(gdp_type="CVM", seasonal="SA")
    log_gdp = np.log(gdp.data) * 100

    return DataSeries(
        data=log_gdp,
        source=gdp.source,
        units="log points (x100)",
        description=f"Log {gdp.description}",
        series_id=gdp.series_id,
        table=gdp.table,
        cat=gdp.cat,
        stype=gdp.stype,
    )


def get_gdp_growth(periods: int = 1) -> DataSeries:
    """Load GDP growth as log difference.

    Args:
        periods: Number of periods for differencing (1=quarterly, 4=annual)

    Returns:
        DataSeries with GDP growth rate
    """
    log_gdp = get_log_gdp()
    growth = log_gdp.data.diff(periods)

    period_label = "quarterly" if periods == 1 else f"{periods}-period"

    return DataSeries(
        data=growth,
        source=log_gdp.source,
        units="% per period",
        description=f"GDP growth ({period_label})",
        series_id=log_gdp.series_id,
        table=log_gdp.table,
        cat=log_gdp.cat,
        stype=log_gdp.stype,
    )
