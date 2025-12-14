"""Capital stock data loading.

Provides net capital stock from ABS Modellers Database (1364.0.15.003).
"""

import numpy as np

from src.data.abs_loader import load_series
from src.data.dataseries import DataSeries
from src.data.series_specs import CAPITAL_STOCK


def get_capital_stock_qrtly() -> DataSeries:
    """Get net capital stock from Modellers Database (quarterly).

    Non-financial and financial corporations net capital stock
    in chain volume measures.

    Returns:
        DataSeries with quarterly capital stock

    """
    return load_series(CAPITAL_STOCK)


def get_capital_growth_qrtly() -> DataSeries:
    """Get quarterly capital stock growth (log difference).

    Returns:
        DataSeries with capital growth (% per quarter)

    """
    capital = get_capital_stock_qrtly()
    log_capital = np.log(capital.data) * 100
    capital_growth = log_capital.diff(1)

    return DataSeries(
        data=capital_growth,
        source=capital.source,
        units="% per quarter",
        description="Capital stock growth (quarterly, log difference)",
        series_id=capital.series_id,
        table=capital.table,
        cat=capital.cat,
        stype=capital.stype,
    )
