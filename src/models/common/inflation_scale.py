"""The one place the repo converts a neutral rate between real and nominal.

Every r* package needs this and they used to each do it themselves, by adding
or subtracting the 2.5% target. That was defensible, and it was also not what
anyone else does: the RBA and CBA both convert using long-run inflation
EXPECTATIONS, so a comparison against a published neutral rate was never quite
like for like. Over the 1990s the two conventions differ by up to a point.

WHICH SERIES, AND WHY IT IS THE ANCHORED ONE. `src/data/expectations_model.py`
publishes two. The UNANCHORED median has no target prior and moves with the
cycle: it reads 1.72 at its trough and 3.32 in 2023, and converting a neutral
rate with it drags the inflation cycle into r* (nominal r* would have fallen to
0.51 in 2020Q4 purely because expectations dipped). The TARGET-ANCHORED median
is the long-run measure, and over 1993Q1 onward it runs 2.13 to 3.50 with a
standard deviation of 0.28:

    1993-1999   2.98        the target was new and expectations had not converged
    2000-2007   2.50
    2008-2015   2.60
    2016-2021   2.32
    2022-       2.57        barely moved through the inflation spike

That last row is the test that matters. `rstar_rba`'s notes reject deflating by
REALISED inflation, because doing so made its real neutral hit -2.2 in 2022 for
no reason except that inflation peaked. Anchored expectations do not do that,
which is why they are usable here and realised inflation is not.

SO THE CHANGE IS SMALL AND IN THE RIGHT PLACE: it is worth about +0.5pp through
the 1990s re-anchoring, where expectations genuinely sat above target, and
close to nothing after 2000.

WHAT THIS IS NOT FOR. Deflating an actual borrowing rate, or the policy rate, to
get a real rate someone faced. That wants the unanchored series or realised
inflation, because what a borrower pays in real terms depends on what inflation
does and not on the target. This module converts the SCALE of a neutral rate,
which is a different operation: it answers "what nominal rate corresponds to
this real neutral, at the inflation people expect over the long run".

IT CREATES A DEPENDENCY. Any package converting scales now needs a completed
`./run-expectations.sh`. That is deliberate and it fails loudly: a silent
fallback to 2.5 would publish one convention under the label of another.
`scale="target"` restores the old behaviour explicitly, which keeps every
previously published number reproducible.
"""

import pandas as pd

from src.data.expectations_model import get_model_expectations

# The RBA's midpoint, used when `scale="target"` and as the label elsewhere.
TARGET = 2.5

SCALES = ("expectations", "target")


def long_run_expectations(index: pd.PeriodIndex | None = None) -> pd.Series:
    """Return target-anchored long-run inflation expectations, quarterly.

    Args:
        index: Reindex onto this if given. Quarters outside the expectations
            sample come back as NaN rather than being filled, so a model whose
            sample runs past the expectations run shows a gap instead of a
            silently flat tail.

    Returns:
        Series of expectations in per cent, on a quarterly PeriodIndex.

    Raises:
        FileNotFoundError: If the expectations model has not been run.

    """
    try:
        series = get_model_expectations().data.astype(float)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Converting between real and nominal needs the expectations model's "
            "target-anchored series, and it has not been run. Run ./run-expectations.sh, "
            "or pass scale='target' to use the 2.5% target as this repo used to.",
        ) from exc

    if not isinstance(series.index, pd.PeriodIndex):
        series.index = pd.PeriodIndex(series.index, freq="Q")
    series = series.dropna()
    return series.reindex(index) if index is not None else series


def _offset(index: pd.PeriodIndex, scale: str) -> pd.Series | float:
    """Return the inflation term to add or subtract, on `index`."""
    if scale == "target":
        return TARGET
    if scale != "expectations":
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    return long_run_expectations(index)


def to_nominal(real: pd.Series, *, scale: str = "expectations") -> pd.Series:
    """Convert a real neutral rate to nominal.

    Args:
        real: The real rate, on a quarterly PeriodIndex.
        scale: "expectations" (default) adds target-anchored long-run
            expectations; "target" adds the 2.5% target, the repo's old
            convention, kept so published numbers stay reproducible.

    Returns:
        The nominal rate on the same index.

    """
    index = real.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
        real = real.set_axis(index)
    return real + _offset(index, scale)


def to_real(nominal: pd.Series, *, scale: str = "expectations") -> pd.Series:
    """Convert a nominal neutral rate to real.

    The inverse of `to_nominal`, and the direction `rstar_rba` needs: it
    estimates a nominal neutral directly and reports the real one.

    Args:
        nominal: The nominal rate, on a quarterly PeriodIndex.
        scale: As `to_nominal`.

    Returns:
        The real rate on the same index.

    """
    index = nominal.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
        nominal = nominal.set_axis(index)
    return nominal - _offset(index, scale)


def scale_label(scale: str = "expectations") -> str:
    """Return a short phrase naming the convention, for chart headers."""
    if scale == "target":
        return f"the {TARGET:g}% target"
    if scale != "expectations":
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    return "long-run inflation expectations"
