"""Where each estimate of u* comes from, and how to load it.

NOT A MODEL. This gathers u* from the two models in the repo that estimate
one and puts several of their specifications on a single chart. Nothing here
is estimated.

SEVERAL LINES, AND NOT INDEPENDENT VOTES. Every one is fitted to the same
unemployment rate, the same trimmed mean inflation and the same expectations
series.

- `ustar` reads u* from one Phillips curve. Its three settings differ in the
  structure imposed on u*: a tapered random walk, or a spline with one or two
  knots.
- `ystar_ustar` estimates y* and u* in one likelihood. Only its tapered
  random walk, on the inflation-defined output gap, is charted here, beside
  the `ustar` walk it can be read against.

Each source is a specification from its package's `--compare`, found by
prefix, so the flags that reproduce it, its refresh and its chart directory
are defined once, there. A prefix missing from its package stops this at
import rather than quietly charting fewer lines.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from src.models.ustar import compare as ustar_compare
from src.models.ustar.config import ModelConfig
from src.models.ustar.results import load_results as load_ustar_results
from src.models.ystar_ustar import compare as joint_compare
from src.models.ystar_ustar.config import ModelConfig as JointConfig
from src.models.ystar_ustar.results import load_results as load_joint_results

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Either package's `--compare` specification. They are separate classes with
# the same interface: prefix, label, is_current(), refresh() and chart().
type Specification = ustar_compare.Specification | joint_compare.Specification


@dataclass(frozen=True)
class UstarSource:
    """One specification's u*, and how to get it.

    Attributes:
        model: the model's short name, prefixed to the chart label
        name: the u* structure, worded the same for every model so that one
            structure reads as one thing on the shared legend
        spec: the `--compare` specification that reproduces this run
        loader: returns the posterior median of u*, per cent
        note: what this u* is built from, for the run log

    """

    model: str
    name: str
    spec: Specification
    loader: Callable[[str], pd.Series]
    note: str

    @property
    def label(self) -> str:
        """How the chart names this line: the model, then its u* structure.

        Not the `--compare` label, which each package words for its own
        comparison and which would name one structure two ways here.
        """
        return f"{self.model}: {self.name}"

    @property
    def prefix(self) -> str:
        """The saved run this reads."""
        return self.spec.prefix

    def is_current(self) -> bool:
        """Return True if the saved trace was written today."""
        return self.spec.is_current()

    def refresh(self) -> None:
        """Re-estimate this specification, then redraw its own charts."""
        self.spec.refresh()
        self.spec.chart()


def _find(specifications: Sequence[Specification], prefix: str) -> Specification:
    """Return the specification saved under `prefix`, failing loudly if there is none."""
    for spec in specifications:
        if spec.prefix == prefix:
            return spec
    raise KeyError(f"no --compare specification with prefix {prefix!r}")


def _load_ustar(prefix: str) -> pd.Series:
    """u* from `ustar`, on whichever specification `prefix` was run with."""
    return load_ustar_results(prefix=prefix).ustar_posterior().median(axis=1)


def _load_joint(prefix: str) -> pd.Series:
    """u* from the joint y*/u* model."""
    return load_joint_results(prefix=prefix).ustar_posterior().median(axis=1)


_USTAR = "u*"
_JOINT = "y*/u*"

# u* structure names, set once so the legend cannot word one structure two
# ways when both models chart it.
_WALK = "Random walk"
_SPLINE_1 = "Spline 1 knot"
_SPLINE_2 = "Spline 2 knots"

# `ustar`'s default run, which is also where the unemployment rate is read.
_USTAR_DEFAULT = _find(ustar_compare.SPECIFICATIONS, "ustar")


def load_unemployment() -> pd.Series:
    """Return the unemployment rate every line is fitted to, from the default `ustar` run."""
    results = load_ustar_results(prefix=_USTAR_DEFAULT.prefix)
    return pd.Series(results.obs["u"], index=results.obs_index, name="Unemployment rate")


SOURCES: tuple[UstarSource, ...] = (
    UstarSource(
        model=_USTAR,
        name=_WALK,
        spec=_USTAR_DEFAULT,
        loader=_load_ustar,
        note=f"one Phillips curve, u* a random walk whose step size tapers to {ModelConfig.taper_end}; "
             "the default run",
    ),
    UstarSource(
        model=_USTAR,
        name=_SPLINE_1,
        spec=_find(ustar_compare.SPECIFICATIONS, "ustar_sum_k1"),
        loader=_load_ustar,
        note="one Phillips curve, u* a spline with one knot",
    ),
    UstarSource(
        model=_USTAR,
        name=_SPLINE_2,
        spec=_find(ustar_compare.SPECIFICATIONS, "ustar_sum_k2"),
        loader=_load_ustar,
        note="one Phillips curve, u* a spline with two knots",
    ),
    UstarSource(
        model=_JOINT,
        name=_WALK,
        spec=_find(joint_compare.SPECIFICATIONS, "yus_sum_taper"),
        loader=_load_joint,
        note="y* and u* in one likelihood, u* a random walk whose step size tapers to "
             f"{JointConfig.taper_end}",
    ),
)


def gather(
    *,
    allow_refresh: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return every specification's u* on a shared quarterly index.

    Args:
        allow_refresh: re-run any specification whose saved trace is not from
            today, which also redraws its own charts.
        verbose: print what was current, what was refreshed and what was read

    Returns:
        (frame, notes). `frame` holds one column per source; `notes` maps each
        label to what its u* is built from.

    """
    if verbose:
        print("Checking vintages (a trace counts as current if written today):")
        for source in SOURCES:
            state = "current" if source.is_current() else "STALE"
            print(f"  {state:>8}  {source.label}")

    if allow_refresh:
        for source in SOURCES:
            if not source.is_current():
                source.refresh()

    columns, notes = _load_all(verbose=verbose)
    if not columns:
        raise RuntimeError("no u* specifications could be loaded")
    return pd.DataFrame(columns).sort_index(), notes


def _as_quarterly(series: pd.Series) -> pd.Series:
    """Return `series` on a quarterly PeriodIndex."""
    index = series.index
    if isinstance(index, pd.PeriodIndex):
        series.index = index.asfreq("Q")
    elif isinstance(index, pd.DatetimeIndex):
        series.index = index.to_period("Q")
    return series.astype(float)


def _load_all(*, verbose: bool) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Load each source. One that cannot be read is named and skipped."""
    columns: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    if verbose:
        print("\nLoading:")
    for source in SOURCES:
        try:
            series = _as_quarterly(source.loader(source.prefix))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  SKIPPED {source.label}: {type(exc).__name__}: {exc}")
            continue
        columns[source.label] = series
        notes[source.label] = source.note
        if verbose:
            print(f"  {source.label:<52} {source.prefix}")
    return columns, notes
