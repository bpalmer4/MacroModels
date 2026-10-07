"""RBA Index of Commodity Prices (ICP).

Source: RBA Statistical Tables, Table I2 — Commodity Prices.

The ICP is a chain-weighted index of Australian export commodity prices,
published in three currencies (AUD, USD, SDR). The USD version is the
world prices being paid for Australian commodity exports — most directly
the iron-ore, coal and LNG demand transmitted from Asia (China, Japan,
Korea). The AUD version is those prices converted at the exchange rate, so
it moves with the dollar too: it is what Australian producers receive.

For an SOE IS curve this is more upstream than terms of trade (no
import-price denominator) and more exogenous than net exports (no quantity
feedback from domestic demand).
"""

import numpy as np
import pandas as pd
from readabs import read_rba_table

from src.data.dataseries import DataSeries

_TABLE = "I2"
_SERIES_ID_AUD = "GRCPAIAD"  # Commodity prices – A$, monthly index, 2024/25 = 100
_SERIES_ID_USD = "GRCPAIUSD"  # Commodity prices – US$, monthly index, 2024/25 = 100


def _icp_qrtly(series_id: str) -> pd.Series:
    """Return one I2 series as a quarterly average, from its monthly values."""
    data, _ = read_rba_table(_TABLE)
    monthly = pd.to_numeric(data[series_id], errors="coerce").dropna()
    if not isinstance(monthly.index, pd.PeriodIndex):
        monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    return monthly.groupby(monthly.index.asfreq("Q")).mean()


def get_icp_aud_qrtly() -> DataSeries:
    """RBA Index of Commodity Prices in AUD, quarterly average level."""
    return DataSeries(
        data=_icp_qrtly(_SERIES_ID_AUD),
        source="RBA",
        units="Index, 2024/25 = 100",
        description="RBA Index of Commodity Prices (A$), quarterly average",
        table=_TABLE,
        series_id=_SERIES_ID_AUD,
    )


def get_icp_usd_qrtly() -> DataSeries:
    """RBA Index of Commodity Prices in USD, quarterly average level.

    World prices for Australia's commodity exports with the Australian dollar
    taken out, so the index can stand on the other side of an equation for the
    exchange rate. The A$ index is these prices converted at the exchange rate.
    """
    return DataSeries(
        data=_icp_qrtly(_SERIES_ID_USD),
        source="RBA",
        units="Index, 2024/25 = 100",
        description="RBA Index of Commodity Prices (US$), quarterly average",
        table=_TABLE,
        series_id=_SERIES_ID_USD,
    )


def get_icp_aud_change_qrtly() -> DataSeries:
    """Quarterly percentage change in the RBA ICP (AUD), log diff x 100."""
    icp = get_icp_aud_qrtly().data
    delta = (np.log(icp) - np.log(icp.shift(1))) * 100
    return DataSeries(
        data=delta,
        source="RBA",
        units="%",
        description="RBA ICP (A$) change (quarterly log diff x 100)",
        table=_TABLE,
        series_id=_SERIES_ID_AUD,
    )


def get_icp_aud_change_lagged_qrtly() -> DataSeries:
    """Quarterly ICP (AUD) change, lagged one quarter."""
    delta = get_icp_aud_change_qrtly()
    return DataSeries(
        data=delta.data.shift(1),
        source=delta.source,
        units=delta.units,
        description="RBA ICP (A$) change lagged one quarter",
        table=delta.table,
        series_id=delta.series_id,
    )


if __name__ == "__main__":
    s = get_icp_aud_change_lagged_qrtly()
    print(f"ICP A$ change (lag 1): {s.data.index[0]} to {s.data.index[-1]}")
    print(f"  range: [{s.data.min():.2f}, {s.data.max():.2f}]")
    print(f"  recent: {s.data.dropna().tail(5).to_string()}")
