"""The one place the repo converts a neutral rate between real and nominal.

A real neutral rate becomes nominal by adding inflation expectations, which is
what the RBA and CBA do, so a nominal line is comparable with a published
neutral rate. `scale="target"` adds the flat `TARGET` instead.

WHICH SERIES. Inflation expectations here are the expectations model's
unanchored median: no target observation, so expectations are free to move
away from the target, as they did in the post-pandemic inflation.

WHY NOT REALISED INFLATION. It swings with every price shock and would carry
those swings straight into a neutral rate. Expectations move far less.

IT CREATES A DEPENDENCY. Converting with expectations needs a completed
`./run-expectations.sh`, and fails loudly without one rather than falling back
to the target.
"""

from collections.abc import Callable

import pandas as pd

from src.data.dataseries import DataSeries
from src.data.expectations_model import get_model_expectations_unanchored

# The RBA's midpoint, used when `scale="target"` and as the label elsewhere.
TARGET = 2.5

SCALES = ("expectations", "target")


def _quarterly(loader: Callable[[], DataSeries], name: str, index: pd.PeriodIndex | None) -> pd.Series:
    """Return one of the expectations model's saved medians, quarterly.

    Quarters outside the expectations sample come back as NaN rather than being
    filled, so a model whose sample runs past the expectations run shows a gap
    instead of a silently flat tail.
    """
    try:
        series = loader().data.astype(float)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"The expectations model's {name} series has not been saved. Run "
            f"./run-expectations.sh, or pass scale='target' to convert with the {TARGET:g}% target.",
        ) from exc

    if not isinstance(series.index, pd.PeriodIndex):
        series.index = pd.PeriodIndex(series.index, freq="Q")
    series = series.dropna()
    return series.reindex(index) if index is not None else series


def get_unanchored_expectations(index: pd.PeriodIndex | None = None) -> pd.Series:
    """Return the plain median, quarterly: every input, no target observation.

    Args:
        index: Reindex onto this if given.

    Returns:
        Series of expectations in per cent, on a quarterly PeriodIndex.

    Raises:
        FileNotFoundError: If the expectations model has not been run.

    """
    return _quarterly(get_model_expectations_unanchored, "unanchored", index)


def _offset(index: pd.PeriodIndex, scale: str) -> pd.Series | float:
    """Return the inflation term to add or subtract, on `index`."""
    if scale == "target":
        return TARGET
    if scale != "expectations":
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    return get_unanchored_expectations(index)


def to_nominal(real: pd.Series, *, scale: str = "expectations") -> pd.Series:
    """Convert a real neutral rate to nominal.

    Args:
        real: The real rate, on a quarterly PeriodIndex.
        scale: "expectations" (default) adds inflation expectations;
            "target" adds `TARGET`.

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
    return "inflation expectations"
