"""World r-star data from NY Fed Holston-Laubach-Williams estimates.

Loads the published HLW r-star estimates for the US, Euro Area, and Canada.

Source:
    https://www.newyorkfed.org/research/policy/rstar
    Holston, Laubach, and Williams. "Measuring the Natural Rate of Interest
    Across Time and Space."

Caching is `readabs`', not ours: `get_file` keys the workbook by URL into
`.readabs_cache/`, refreshes it on the server's Last-Modified header, and falls
back to the cached copy when the NY Fed is unreachable. Every call therefore
gets the currently published estimates without asking.
"""

from io import BytesIO

import pandas as pd
from readabs.download_cache import get_file

_HLW_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/economists/"
    "williams/data/Holston_Laubach_Williams_current_estimates.xlsx"
)


def get_world_rstar() -> pd.DataFrame:
    """NY Fed HLW r* estimates for US, Euro Area, Canada.

    Returns:
        Quarterly PeriodIndex DataFrame with columns ``US``, ``Euro Area``,
        ``Canada`` (% per annum). NaN where a country's estimate is
        unavailable for a given period.

    """
    workbook = BytesIO(get_file(_HLW_URL, cache_prefix="hlw"))
    df = pd.read_excel(workbook, sheet_name="HLW Estimates", header=[4, 5])
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
