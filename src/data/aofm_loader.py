"""Australian Treasury Bond yield-curve decomposition, from the AOFM.

The AOFM publishes a daily decomposition of the nominal Australian Treasury
Bond curve into a fitted zero-coupon yield, a term premium and a risk-neutral
yield, at every tenor from 1 to 10 years, from 1992-07-01. It accompanies their
working paper "Estimation of the Term Premium within Australian Treasury
Bonds".

Source:
    https://www.aofm.gov.au/data-hub

Why this matters to `rstar_bonds`. That package has to split the Australian
long yield into a natural rate and a term premium, and it does so by asserting
that the premium is stationary about a constant while r* is a random walk.
Nothing in the data enforces that split, which is why `wedge_0` and `mu_tp`
correlate at -0.87. This file is the first external Australian estimate of the
premium the package has had: the assertion can now be checked, and the premium
can be taken as data instead of being inferred.

It also supersedes the position recorded in `src/data/bonds.py` and the
`rstar_bonds` notes, that an Australian ACM-style decomposition was not
possible. It was not possible to *estimate* on RBA data: F2 carries only four
coupon tenors and F17's zero-coupon curve starts in 2017. The AOFM had already
done it, on a fitted zero-coupon curve running back to 1992.

TWO METHODS, AND WHY THE DEFAULT IS THE BIAS-CORRECTED ONE.

    "ols" — a five-factor Adrian, Crump and Moench (2013) decomposition.
    "bc"  — the same, with a bootstrap bias correction to the VAR parameters.

ACM's three-step regression inherits the small-sample downward bias in the
persistence of a highly autocorrelated VAR, which pushes variation out of the
expectations component and into the premium (Bauer, Rudebusch and Wu 2012). The
`rstar_bonds` package already acts on that: it tested plain ACM as the US
premium and rejected it, because it put the post-GFC US premium at 1.27 against
Kim-Wright's 0.45 and then turned it negative. The AOFM's "ols" sheet is the
same estimator on Australian data, so the same scepticism applies and "bc" is
the default here. The two disagree by enough to matter: over 1993Q1-2026Q3 the
10-year premium falls 2.99 points on "ols" against 1.99 on "bc".

THE SERIES IS REVISED IN FULL, EVERY MONTH. The AOFM's own note says so:
"estimations of decompositions can change when new data is included into the
sample as model parameter estimations are based on linear regressions." Every
historical value moves when the file is updated. That is fine for a level or a
shape comparison and it is NOT a real-time series, so it cannot be used inside
a truncated-sample exercise without putting future data into the premium.

NOMINAL ONLY. There is no indexed decomposition at any vintage, so subtracting
these from a real yield leaves the inflation risk premium behind. That mismatch
is the same one the package already carries with Kim-Wright, and this file
removes the country mismatch rather than that one.

Caching is `readabs`', as `acm_loader` does: `get_file` keys the workbook by URL
into `.readabs_cache/`, refreshes it on the server's Last-Modified header, and
falls back to the cached copy when the AOFM is unreachable.
"""

import re
from functools import cache
from io import BytesIO

import pandas as pd
from readabs.download_cache import get_file

from src.data.dataseries import DataSeries

# The datestamp in the path is the node's creation date, not the data vintage:
# the file is updated in place, and the copy behind this URL carries data well
# past 2025-06-06. `_resolve_url` re-finds it from the data hub if it moves.
_AOFM_URL = "https://www.aofm.gov.au/sites/default/files/2025-06-06/term%20premium.xlsx"
_AOFM_HUB = "https://www.aofm.gov.au/data-hub"

_METHOD_SHEETS = {"bc": "TermPremiumBC", "ols": "TermPremiumOLS"}
_QUANTITY_PREFIX = {"tp": "TP", "rny": "RNY", "fy": "FY"}


def _reachable(url: str) -> bool:
    """Return whether the workbook can be fetched (or served from cache) at `url`."""
    try:
        get_file(url, cache_prefix="aofm")
    except Exception as exc:
        print(f"  note: AOFM workbook not available at {url} ({type(exc).__name__}: {exc})")
        return False
    return True


@cache
def _resolve_url() -> str:
    """Return a working URL for the workbook, re-finding it if the constant is stale.

    The AOFM serves the file from a path containing the node's creation date, so
    a site rebuild would move it. Rather than fail, look the link up on the data
    hub page the way a reader would. The constant is tried first so the normal
    path costs one request, not two.
    """
    if _reachable(_AOFM_URL):
        return _AOFM_URL

    page = get_file(_AOFM_HUB, cache_prefix="aofm").decode("utf-8", errors="replace")
    hrefs = re.findall(r'href="([^"]*term[%20_ ]*premium[^"]*\.xlsx)"', page, flags=re.IGNORECASE)
    if not hrefs:
        raise ValueError(
            f"The AOFM term premium workbook is not at {_AOFM_URL} and no replacement link "
            f"was found on {_AOFM_HUB}. Check the data hub by hand.",
        )
    href = hrefs[0]
    return href if href.startswith("http") else f"https://www.aofm.gov.au{href}"


@cache
def _aofm_frame(method: str) -> pd.DataFrame:
    """Return one decomposition sheet on a DatetimeIndex, numeric throughout."""
    if method not in _METHOD_SHEETS:
        raise ValueError(
            f"Unknown AOFM method {method!r}; expected one of {', '.join(sorted(_METHOD_SHEETS))}",
        )
    # Row 1 is the method's title banner and row 2 carries the column names.
    raw = pd.read_excel(
        BytesIO(get_file(_resolve_url(), cache_prefix="aofm")),
        sheet_name=_METHOD_SHEETS[method],
        header=1,
    )
    frame = raw.rename(columns={raw.columns[0]: "DATE"})
    dates = pd.to_datetime(frame["DATE"], errors="coerce")
    frame = frame.loc[dates.notna()].copy()
    frame.index = pd.DatetimeIndex(dates.loc[dates.notna()])
    return frame.drop(columns="DATE").apply(pd.to_numeric, errors="coerce")


def _aofm_series(quantity: str, maturity: int, method: str) -> DataSeries:
    """Return one column of the decomposition, daily, as a `DataSeries`."""
    if quantity not in _QUANTITY_PREFIX:
        raise ValueError(
            f"Unknown quantity {quantity!r}; expected one of {', '.join(sorted(_QUANTITY_PREFIX))}",
        )
    column = f"{_QUANTITY_PREFIX[quantity]}{maturity}"
    frame = _aofm_frame(method)
    if column not in frame.columns:
        raise ValueError(
            f"No {quantity.upper()} for a {maturity}-year tenor ({column} not in the "
            f"{_METHOD_SHEETS[method]} sheet). Available: "
            f"{sorted(c for c in frame.columns if c.startswith(_QUANTITY_PREFIX[quantity]))}",
        )
    labels = {
        "tp": "term premium",
        "rny": "risk-neutral yield",
        "fy": "fitted zero-coupon yield",
    }
    return DataSeries(
        data=frame[column].dropna(),
        source="AOFM",
        units="%",
        description=(
            f"AOFM {labels[quantity]}, {maturity}-year zero-coupon Australian "
            f"Treasury Bond ({method.upper()} method)"
        ),
        table=_METHOD_SHEETS[method],
        series_id=column,
    )


def get_aofm_term_premium(maturity: int = 10, method: str = "bc") -> DataSeries:
    """Term premium on a nominal zero-coupon Australian Treasury Bond.

    Args:
        maturity: Tenor in years, 1 to 10.
        method: "bc" for the bias-corrected decomposition, "ols" for plain ACM.

    Returns:
        DataSeries with the daily term premium (% per annum), 1992-07 onwards.

    """
    return _aofm_series("tp", maturity, method)


def get_aofm_risk_neutral_yield(maturity: int = 10, method: str = "bc") -> DataSeries:
    """Risk-neutral yield: the fitted yield less the term premium.

    This is the average expected nominal short rate over the tenor, with the
    term premium removed by the AOFM rather than by a latent state. Subtracting
    expected inflation turns it into the real expected path a natural-rate model
    wants, without an inflation risk premium on top of a term premium.

    Args:
        maturity: Tenor in years, 1 to 10.
        method: "bc" for the bias-corrected decomposition, "ols" for plain ACM.

    Returns:
        DataSeries with the daily risk-neutral yield (% per annum), 1992-07 onwards.

    """
    return _aofm_series("rny", maturity, method)


def get_aofm_fitted_yield(maturity: int = 10, method: str = "bc") -> DataSeries:
    """Model-fitted zero-coupon yield, the sum of the other two.

    Carried so the decomposition can be checked against the curve it came from.
    The two methods fit the same curve, so this is near-identical across them.

    Args:
        maturity: Tenor in years, 1 to 10.
        method: "bc" for the bias-corrected decomposition, "ols" for plain ACM.

    Returns:
        DataSeries with the daily fitted zero-coupon yield (% per annum).

    """
    return _aofm_series("fy", maturity, method)


def get_aofm_5y5y_forward(method: str = "bc") -> DataSeries:
    """Return the 5y5y risk-neutral forward: the market's read on where rates settle.

    The average expected nominal short rate over the five years BEGINNING five
    years from now, with the term premium removed by AOFM. Five years out is far
    enough that the current cycle should have washed through, so what is left is
    close to the market's view of the neutral nominal cash rate.

    Derived from the zero-coupon risk-neutral curve by the standard forward
    identity, since a 10-year average is the 5-year average and the 5y5y forward
    in equal parts:

        5y5y = 2 * RNY10 - RNY5

    Verified 2026-09-16 to reproduce CBA's chart 5 on both methods: OLS 3.98
    (2010Q2), 3.12 (2020Q4), 3.87 (2026Q2); BC 3.94, 2.09, 3.70. CBA describe
    the series as "currently around 3.9%".

    WHAT IT IS NOT. It is the expected average POLICY RATE, not a neutral rate:
    it still contains whatever the market believes about the cyclical position
    over years five to ten, and whatever premium AOFM's model failed to strip.
    Anything using it as a measure of neutral is asserting those are small.
    """
    five = get_aofm_risk_neutral_yield(5, method).data
    ten = get_aofm_risk_neutral_yield(10, method).data
    forward = (2.0 * ten - five).dropna()
    return DataSeries(
        data=forward,
        source="AOFM",
        units="%",
        description=f"AOFM 5y5y risk-neutral forward rate ({method.upper()} method)",
        table=_METHOD_SHEETS[method],
        series_id="5Y5Y",
    )
