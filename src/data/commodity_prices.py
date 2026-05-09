"""RBA Index of Commodity Prices (ICP).

Source: RBA Statistical Tables, Table I2 — Commodity Prices.

The ICP is a chain-weighted index of Australian export commodity prices,
published in three currencies (AUD, USD, SDR). The AUD version strips out
exchange-rate movements, leaving a clean signal for the world prices being
paid for Australian commodity exports — most directly the iron-ore, coal
and LNG demand transmitted from Asia (China, Japan, Korea).

For an SOE IS curve this is more upstream than terms of trade (no
import-price denominator) and more exogenous than net exports (no quantity
feedback from domestic demand).
"""

from readabs import read_rba_table

from src.data.dataseries import DataSeries

_TABLE = "I2"
_SERIES_ID_AUD = "GRCPAIAD"  # Commodity prices – A$, monthly index, 2024/25 = 100


def get_icp_aud_qrtly() -> DataSeries:
    """RBA Index of Commodity Prices in AUD, quarterly average level."""
    import pandas as pd  # noqa: PLC0415
    data, _ = read_rba_table(_TABLE)
    monthly = pd.to_numeric(data[_SERIES_ID_AUD], errors="coerce").dropna()
    if not isinstance(monthly.index, pd.PeriodIndex):
        monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    quarterly = monthly.groupby(monthly.index.asfreq("Q")).mean()
    return DataSeries(
        data=quarterly,
        source="RBA",
        units="Index, 2024/25 = 100",
        description="RBA Index of Commodity Prices (A$), quarterly average",
        table=_TABLE,
        series_id=_SERIES_ID_AUD,
    )


def get_icp_aud_change_qrtly() -> DataSeries:
    """Quarterly percentage change in the RBA ICP (AUD), log diff x 100."""
    import numpy as np  # noqa: PLC0415
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
