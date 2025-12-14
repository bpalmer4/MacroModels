"""Import price data loading.

Provides import price index from ABS International Trade Price Indexes (6457.0).
"""

import numpy as np
import readabs as ra

from src.data.dataseries import DataSeries


def get_import_price_index_qrtly() -> DataSeries:
    """Get import price index (quarterly).

    From ABS 6457.0 Import Price Index (series A2298279F).

    Returns:
        DataSeries with import price index
    """
    trade, _meta = ra.read_abs_series(cat="6457.0", series_id="A2298279F")
    series = trade["A2298279F"]

    return DataSeries(
        data=series,
        source="ABS",
        units="Index",
        description="Import Price Index",
        series_id="A2298279F",
        cat="6457.0",
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
        description="Import price growth (annual, 4-quarter log difference)",
        series_id=index.series_id,
        cat=index.cat,
    )
