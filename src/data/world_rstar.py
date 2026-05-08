"""World r-star data from NY Fed Holston-Laubach-Williams estimates.

Loads the published HLW r-star estimates for the US, Euro Area, and Canada.

Source:
    https://www.newyorkfed.org/research/policy/rstar
    Holston, Laubach, and Williams. "Measuring the Natural Rate of Interest
    Across Time and Space."

The Excel file is downloaded on first use and cached in ``input_data/``.
Pass ``force_download=True`` to refresh.
"""

from pathlib import Path

import pandas as pd
import requests

_HLW_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/economists/"
    "williams/data/Holston_Laubach_Williams_current_estimates.xlsx"
)
_INPUT_DIR = Path(__file__).parent.parent.parent / "input_data"
_CACHE_FILE = _INPUT_DIR / "Holston_Laubach_Williams_current_estimates.xlsx"


def _fetch_or_use_cache(*, force: bool = False) -> Path:
    if _CACHE_FILE.exists() and not force:
        return _CACHE_FILE
    _INPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching HLW data: {_HLW_URL}")
    try:
        response = requests.get(
            _HLW_URL,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MacroModels)"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download HLW data from {_HLW_URL}.\n"
            f"  Error: {exc}\n"
            f"Workaround: download the file manually from "
            f"https://www.newyorkfed.org/research/policy/rstar and place it at\n"
            f"  {_CACHE_FILE}",
        ) from exc
    _CACHE_FILE.write_bytes(response.content)
    print(f"  cached to {_CACHE_FILE}")
    return _CACHE_FILE


def get_world_rstar(*, force_download: bool = False) -> pd.DataFrame:
    """NY Fed HLW r* estimates for US, Euro Area, Canada.

    Args:
        force_download: Re-fetch the source Excel file even if cached.

    Returns:
        Quarterly PeriodIndex DataFrame with columns ``US``, ``Euro Area``,
        ``Canada`` (% per annum). NaN where a country's estimate is
        unavailable for a given period.

    """
    path = _fetch_or_use_cache(force=force_download)
    df = pd.read_excel(path, sheet_name="HLW Estimates", header=[4, 5])
    date = pd.to_datetime(df[("Unnamed: 0_level_0", "Date")], errors="coerce")
    out = pd.DataFrame({
        "US":        pd.to_numeric(df[("Natural Rate (r*)", "US")], errors="coerce"),
        "Euro Area": pd.to_numeric(df[("Natural Rate (r*)", "Euro Area")], errors="coerce"),
        "Canada":    pd.to_numeric(df[("Natural Rate (r*)", "Canada")], errors="coerce"),
    })
    out.index = date.dt.to_period("Q")
    out = out[out.index.notna()]
    return out.dropna(how="all")


if __name__ == "__main__":
    df = get_world_rstar()
    print(f"World r*: {df.index[0]} to {df.index[-1]} ({len(df)} obs)")
    print(df.tail())
