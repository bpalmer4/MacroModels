"""FRED data retrieval, for the series that have no Australian equivalent.

Used for the world real rate series the `rstar` package needs as a market-priced
comparator for the Holston-Laubach-Williams anchor. HLW is a model output built
on an IS curve; these are prices. The point of having both is that they disagree.

The API key lives in `fred.api` in the project root, one line, and is
gitignored. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html.
The same file and convention are used in the ABS project.

The plain CSV endpoint (`fredgraph.csv`) needs no key but times out from here,
which is why this goes through the JSON API.
"""

import json
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import pandas as pd
import requests

from src.data.dataseries import DataSeries

_ROOT = Path(__file__).parent.parent.parent
_KEY_FILE = _ROOT / "fred.api"
_CACHE_DIR = _ROOT / "input_data" / "fred"
_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT = 60
# A day is plenty: these are daily or monthly series and the model is quarterly.
_MAX_CACHE_AGE_DAYS = 1


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


def _cache_path(series_id: str) -> Path:
    return _CACHE_DIR / f"{series_id}.json"


def _cached(series_id: str) -> dict | None:
    """Return the cached payload if it is fresh enough, else None."""
    path = _cache_path(series_id)
    if not path.exists():
        return None
    age_days = (datetime.now(UTC).timestamp() - path.stat().st_mtime) / 86_400
    if age_days > _MAX_CACHE_AGE_DAYS:
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


@cache
def get_fred_series(series_id: str) -> DataSeries:
    """Fetch one FRED series, cached on disk for a day.

    Args:
        series_id: FRED series identifier, e.g. "DFII10".

    Returns:
        DataSeries with a DatetimeIndex at the series' native frequency.

    """
    payload = _cached(series_id)
    if payload is None:
        response = requests.get(
            _BASE_URL,
            params={
                "series_id": series_id,
                "api_key": _api_key(),
                "file_type": "json",
            },
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(series_id).write_text(json.dumps(payload), encoding="utf-8")

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
