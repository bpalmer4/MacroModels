"""Trade-Weighted Index (TWI) data loading and transforms.

Provides quarterly TWI series and log changes for exchange rate modeling.
"""

import numpy as np
import pandas as pd

from src.data.dataseries import DataSeries
from src.data.rba_loader import get_real_twi as rba_get_real_twi
from src.data.rba_loader import get_twi as rba_get_twi


def _as_period_index(index: pd.Index, freq: str) -> pd.PeriodIndex:
    """Return `index` as a PeriodIndex at `freq`, narrowing the type at runtime.

    Loaders hand back either a DatetimeIndex (straight from the RBA
    spreadsheets) or a PeriodIndex (once converted here).

    Args:
        index: the index to convert
        freq: target period frequency, e.g. "M" or "Q"

    Returns:
        The index as a PeriodIndex at the requested frequency

    """
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq(freq)
    if isinstance(index, pd.DatetimeIndex):
        return index.to_period(freq)
    raise TypeError(f"expected a DatetimeIndex or PeriodIndex, got {type(index).__name__}")


def get_twi_monthly() -> DataSeries:
    """Get TWI with monthly PeriodIndex.

    Returns:
        DataSeries with monthly TWI (index)

    """
    twi = rba_get_twi()
    data = twi.data.copy()
    data.index = _as_period_index(data.index, "M")

    return DataSeries(
        data=data,
        source="RBA",
        units="Index",
        description="Trade-Weighted Index (monthly)",
        table="F11",
        series_id="FXRTWI",
    )


def get_twi_qrtly() -> DataSeries:
    """Get TWI as quarterly average.

    Returns:
        DataSeries with quarterly TWI (index)

    """
    monthly = get_twi_monthly()
    quarterly = monthly.data.groupby(_as_period_index(monthly.data.index, "Q")).mean()

    return DataSeries(
        data=quarterly,
        source="RBA",
        units="Index",
        description="Trade-Weighted Index (quarterly average)",
        table="F11",
        series_id="FXRTWI",
    )


def get_log_twi_qrtly() -> DataSeries:
    """Get log TWI (× 100) quarterly.

    Returns:
        DataSeries with quarterly log TWI

    """
    twi = get_twi_qrtly()
    log_twi = np.log(twi.data) * 100

    return DataSeries(
        data=log_twi,
        source="RBA",
        units="Log index × 100",
        description="Log Trade-Weighted Index (quarterly)",
        table="F11",
        series_id="FXRTWI",
    )


def get_twi_change_qrtly() -> DataSeries:
    """Get quarterly percentage change in TWI (log difference × 100).

    Positive = AUD appreciation, Negative = AUD depreciation.

    Returns:
        DataSeries with quarterly TWI change (%)

    """
    log_twi = get_log_twi_qrtly()
    delta_twi = log_twi.data.diff()

    return DataSeries(
        data=delta_twi,
        source="RBA",
        units="%",
        description="TWI change (quarterly log diff × 100)",
        table="F11",
        series_id="FXRTWI",
    )


def get_twi_change_annual() -> DataSeries:
    """Get annual percentage change in TWI (4-quarter log difference).

    Positive = AUD appreciation over past year.

    Returns:
        DataSeries with annual TWI change (%)

    """
    log_twi = get_log_twi_qrtly()
    delta_4_twi = log_twi.data.diff(periods=4)

    return DataSeries(
        data=delta_4_twi,
        source="RBA",
        units="% per year",
        description="TWI change (annual, 4-quarter log diff)",
        table="F11",
        series_id="FXRTWI",
    )


def get_twi_change_lagged_qrtly() -> DataSeries:
    """Get lagged quarterly TWI change.

    Returns:
        DataSeries with quarterly TWI change lagged one quarter (%)

    """
    delta_twi = get_twi_change_qrtly()
    delta_twi_1 = delta_twi.data.shift(1)

    return DataSeries(
        data=delta_twi_1,
        source=delta_twi.source,
        units="%",
        description="TWI change (quarterly) lagged one quarter",
        table=delta_twi.table,
        series_id=delta_twi.series_id,
    )


def get_twi_change_lagged_annual() -> DataSeries:
    """Get lagged annual TWI change.

    Returns:
        DataSeries with annual TWI change lagged one quarter (%)

    """
    delta_4_twi = get_twi_change_annual()
    delta_4_twi_1 = delta_4_twi.data.shift(1)

    return DataSeries(
        data=delta_4_twi_1,
        source=delta_4_twi.source,
        units="% per year",
        description="TWI change (annual) lagged one quarter",
        table=delta_4_twi.table,
        series_id=delta_4_twi.series_id,
    )


def get_real_twi_qrtly() -> DataSeries:
    """Get the RBA's real TWI with a quarterly PeriodIndex.

    Real TWI = Nominal TWI × (P_AU / P_trade-partner), the RBA's F15
    measure. It is a relative *price level* adjustment, not a deflation:
    the index rises when Australian prices rise faster than partners'.

    Published quarterly at source, so there is no aggregation here.

    Returns:
        DataSeries with quarterly real TWI (index, March 1995 = 100)

    """
    real_twi = rba_get_real_twi()
    data = real_twi.data.copy()
    data.index = _as_period_index(data.index, "Q")

    return DataSeries(
        data=data,
        source=real_twi.source,
        units=real_twi.units,
        description=real_twi.description,
        table=real_twi.table,
        series_id=real_twi.series_id,
    )
