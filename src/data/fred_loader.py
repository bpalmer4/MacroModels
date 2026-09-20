"""FRED data retrieval, for the series that have no Australian equivalent.

Used for the world real rate series the `rstar` package needs as a market-priced
comparator for the Holston-Laubach-Williams anchor. HLW is a model output built
on an IS curve; these are prices. The point of having both is that they disagree.

The API key lives in `fred.api` in the project root, one line, and is
gitignored. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html.
The same file and convention are used in the ABS project.

The plain CSV endpoint (`fredgraph.csv`) needs no key but times out from here,
which is why this goes through the JSON API.

Caching is `readabs`', not ours: `get_file` keys each request by its full URL
into `.readabs_cache/` and refreshes on the server's Last-Modified header.
FRED sets that header per series, at the series' own revision time, so a
series is re-downloaded when it is actually revised rather than on a timer.
The key is part of the hashed URL but not of the cache filename or its
contents; rotating the key simply orphans the old entries.
"""

import json
from functools import cache
from urllib.parse import urlencode

import pandas as pd
from readabs.download_cache import get_file

from src.data.dataseries import DataSeries
from src.paths import ROOT

_KEY_FILE = ROOT / "fred.api"
_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def _api_key() -> str:
    """Return the FRED API key, or explain how to create one."""
    if not _KEY_FILE.exists():
        raise FileNotFoundError(
            f"No FRED API key at {_KEY_FILE}. Create that file containing the key on one "
            "line (it is gitignored). Free key: https://fred.stlouisfed.org/docs/api/api_key.html",
        )
    key = _KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"{_KEY_FILE} is empty")
    return key


@cache
def get_fred_series(series_id: str) -> DataSeries:
    """Fetch one FRED series, cached on disk until FRED revises it.

    Args:
        series_id: FRED series identifier, e.g. "DFII10".

    Returns:
        DataSeries with a DatetimeIndex at the series' native frequency.

    """
    query = urlencode({
        "series_id": series_id,
        "api_key": _api_key(),
        "file_type": "json",
    })
    payload = json.loads(get_file(f"{_BASE_URL}?{query}", cache_prefix="fred"))
    if not isinstance(payload, dict):
        raise TypeError(f"FRED returned a {type(payload).__name__}, not an object, for {series_id}")

    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError(f"FRED returned no observations for {series_id}")

    frame = pd.DataFrame(observations)
    # FRED marks missing values "."; to_numeric with coerce turns those into NaN.
    values = pd.to_numeric(frame["value"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.to_datetime(frame["date"])).dropna()

    return DataSeries(
        data=series,
        source="FRED",
        units="%",
        description=f"FRED {series_id}",
        table="FRED",
        series_id=series_id,
    )


def get_fred_quarterly(series_id: str, *, how: str = "mean") -> pd.Series:
    """Return a FRED series on a quarterly PeriodIndex.

    Args:
        series_id: FRED series identifier.
        how: "mean" to average within the quarter, "last" for the end value.
            Averaging suits a rate being compared with a quarterly model state;
            "last" suits anything read as a point-in-time price.

    """
    series = get_fred_series(series_id).data
    index = pd.PeriodIndex(series.index, freq="Q")
    grouped = series.groupby(index)
    return (grouped.mean() if how == "mean" else grouped.last()).astype(float)
