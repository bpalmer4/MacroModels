"""US Treasury term premia from the New York Fed's ACM model.

Adrian, Crump and Moench (2013), "Pricing the Term Structure with Linear
Regressions". A no-arbitrage affine term structure model estimated by
three-step regressions on principal components of the Treasury curve. It
splits a nominal zero-coupon yield into a risk-neutral yield (the average
expected short rate over the term) and a term premium.

Source:
    https://www.newyorkfed.org/research/data_indicators/term-premia-tabs

Why this exists alongside `fred_loader`. The `rstar_bonds` package already
subtracts a published US term premium in two places, and until now both used
Kim-Wright (`THREEFYTP10`, the Federal Reserve Board three-factor model),
which FRED carries. ACM is the obvious second opinion and FRED does not carry
it, so it comes straight from the source. The two are the same object
estimated differently and they disagree materially: over 1993Q1-2026Q3 the
difference has a standard deviation of 0.66 points and era means ranging from
+0.82 (2008-2015) to -0.54 (2020-2022), against a 2026Q3 gap of -0.03.

The monthly file is used rather than the daily one: the model is quarterly.

Caching is `readabs`', not ours: `get_file` keys the workbook by URL into
`.readabs_cache/`, refreshes it on the server's Last-Modified header, and falls
back to the cached copy when the NY Fed is unreachable.

Note that this is a NOMINAL term premium, exactly as Kim-Wright is, so it
carries the same mismatch when subtracted from a real yield. Swapping one for
the other tests the provider, not that mismatch.
"""

from functools import cache
from io import BytesIO

import pandas as pd
from readabs.download_cache import get_file

from src.data.dataseries import DataSeries

_ACM_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/"
    "data_indicators/ACMTermPremium.xls"
)


@cache
def _acm_frame() -> pd.DataFrame:
    """Return the whole workbook on a month-end DatetimeIndex."""
    frame = pd.read_excel(BytesIO(get_file(_ACM_URL, cache_prefix="acm")))
    # Dates arrive as "30-Jun-1961" strings; anything unparseable is a footer row.
    dates = pd.to_datetime(frame["DATE"], format="%d-%b-%Y", errors="coerce")
    frame = frame.loc[dates.notna()].copy()
    frame.index = pd.DatetimeIndex(dates.loc[dates.notna()])
    return frame.drop(columns="DATE")


def get_acm_term_premium(maturity: int = 10) -> DataSeries:
    """ACM term premium on a zero-coupon US Treasury of the given maturity.

    Args:
        maturity: Tenor in years, 1 to 10.

    Returns:
        DataSeries with the monthly term premium (% per annum), 1961-06 onwards.

    """
    column = f"ACMTP{maturity:02d}"
    frame = _acm_frame()
    if column not in frame.columns:
        raise ValueError(
            f"No ACM term premium for a {maturity}-year tenor ({column} not in the file). "
            f"Available: {sorted(c for c in frame.columns if c.startswith('ACMTP'))}",
        )
    series = pd.to_numeric(frame[column], errors="coerce").dropna()

    return DataSeries(
        data=series,
        source="NY Fed",
        units="%",
        description=f"ACM term premium, {maturity}-year zero-coupon US Treasury",
        table="ACM",
        series_id=column,
    )
