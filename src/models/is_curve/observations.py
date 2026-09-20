"""Observation assembly for the IS-curve scatter.

Nothing here is estimated. The point is to put the two sides of the IS curve
on the same axes and look at them, so every series is either data or the
saved output of a completed run:

- the output gap, %          a completed `ystar_ustar` run (the joint model)
- the cash rate, %           RBA F1
- inflation expectations, %  the project's expectations model
- r*, %                      a completed `rstar` run, for one of the variants

The real cash rate is `cash_rate - expectations`. What turns it into a *gap*
is r*, and that is the whole question, so three x-axis variants are built:

    none      x = cash - pi_exp                 (no r* imposed at all)
    rstar     x = cash - pi_exp - r*_t          (the bond-market r*)
    constant  x = cash - pi_exp - mean(real)    (a flat r*, the sample mean)

Under `none` the fitted line's zero-gap crossing *is* an estimate of r*, read
off the scatter rather than imposed on it. Under the other two the intercept
measures how far the imposed r* sits from the one the scatter wants.
"""

from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.data.cash_rate import get_cash_rate_qrtly
from src.data.expectations_model import get_model_expectations_unanchored
from src.models.common.sources import SourceSet
from src.models.rstar_bonds.results import load_results as load_bonds_results
from src.models.rstar_rba.estimate import (
    load_results as load_rba_results,
)
from src.models.rstar_rba.estimate import (
    posterior_median,
)
from src.models.ystar_ustar.config import DEFAULT_EXCLUDE_WINDOW
from src.models.ystar_ustar.results import load_results as load_joint_results

if TYPE_CHECKING:
    from collections.abc import Sequence

# The x-axis variants, in the order they are charted and reported.
VARIANTS = ("none", "rstar", "rule", "constant")

# Inflation targeting begins 1993Q1. Before it the cash rate is not set by a
# reaction function the IS curve would recognise, and every model in this repo
# treats the pre-1993 period as a different regime.
DEFAULT_START = "1993Q1"

# The lockdown window `ystar` and `ystar_ustar` drop from their likelihoods.
DEFAULT_WINDOWS: tuple[tuple[str, str], ...] = (DEFAULT_EXCLUDE_WINDOW,)

# The GFC through to the end of the pandemic, as one stretch. Dated from
# 2008Q4, the quarter of the first Fed large-scale asset purchases and of
# `rstar`'s own largest GFC-era wedge move, to 2021Q3, the last lockdown
# quarter and the end of the RBA's own bond purchases and yield curve control.
# It subsumes the lockdown window rather than sitting beside it.
#
# The argument for dropping it is not that the gap is ill-measured but that
# the cash rate stops being the stance: with global QE, forward guidance and
# an effective lower bound, policy runs through balance sheets that the cash
# rate does not show, so the x-axis is mis-measured rather than merely noisy.
GFC_TO_PANDEMIC_WINDOW = ("2008Q4", "2021Q3")


@dataclass
class IsCurveData:
    """The two sides of the IS curve, aligned, plus the r* variants.

    Excluded quarters are *carried*, not removed. The rate enters the fit
    lagged, so dropping rows from the series here would make the lag step
    across the hole and pair a gap with the wrong quarter's rate. The fitting
    code applies `excluded` after shifting instead.
    """

    gap: pd.Series
    real_cash: pd.Series
    rstar: pd.Series
    variants: dict[str, pd.Series]
    constant_rstar: float
    excluded: pd.PeriodIndex
    sources: SourceSet = field(default_factory=SourceSet)

    @property
    def index(self) -> pd.PeriodIndex:
        """The quarters common to both sides."""
        index = self.gap.index
        if not isinstance(index, pd.PeriodIndex):
            raise TypeError(f"expected a PeriodIndex, got {type(index).__name__}")
        return index

    def xy(self, variant: str) -> tuple[pd.Series, pd.Series]:
        """Return (x, y) for one variant, both on the common index."""
        if variant not in self.variants:
            raise KeyError(f"unknown variant {variant!r}; expected one of {list(self.variants)}")
        return self.variants[variant], self.gap


def _excluded_quarters(
    index: pd.PeriodIndex,
    windows: Sequence[tuple[str, str]] | None,
) -> pd.PeriodIndex:
    """Return the quarters inside any of `windows`, which the fits leave out.

    Windows are dropped on the *gap's* quarter, not the rate's. The cash rate
    is observed through the lockdowns as accurately as ever; it is potential
    output, and so the gap, that is not well defined when large parts of the
    economy are closed by order. That is the same ground on which `ystar` and
    `ystar_ustar` drop these quarters from their likelihoods, which is where
    the lockdown window comes from.

    The QE window is excluded on different grounds: not that the gap is
    ill-measured, but that the cash rate stops being the stance. At the
    effective lower bound, with balance-sheet policy doing work the rate
    cannot show, the x-axis no longer measures what the IS curve needs.
    """
    if not windows:
        return pd.PeriodIndex([], freq="Q")
    mask = np.zeros(len(index), dtype=bool)
    for first_str, last_str in windows:
        first, last = pd.Period(first_str, freq="Q"), pd.Period(last_str, freq="Q")
        mask |= (index >= first) & (index <= last)
    return index[mask]


def blocks(index: pd.PeriodIndex, excluded: pd.PeriodIndex) -> list[pd.PeriodIndex]:
    """Split what survives the exclusions into runs of consecutive quarters.

    Excluding a long middle stretch leaves disconnected blocks at different
    rate levels, and a line fitted across them can take its slope from the
    difference *between* the blocks rather than from any response within
    them. Fitting each block separately is the check on that.
    """
    kept = index.difference(excluded).sort_values()
    if not len(kept):
        return []
    # Slice positions rather than np.split, which would turn a PeriodIndex
    # into plain arrays and lose the quarters.
    breaks = [0, *(np.flatnonzero(np.diff([period.ordinal for period in kept]) > 1) + 1).tolist(), len(kept)]
    parts = [kept[start:stop] for start, stop in pairwise(breaks)]
    return [part for part in parts if len(part)]


def _joint_gap(prefix: str, sources: SourceSet) -> pd.Series:
    """Return the median output gap from a completed joint y*/u* run."""
    results = load_joint_results(prefix=prefix)
    constants = getattr(results, "constants", None)
    if isinstance(constants, dict):
        recorded = SourceSet.from_records(constants.get("sources"))
        if recorded is not None:
            for source, cat in recorded.records:
                sources.add(source, cat)
    return results.output_gap_median()


def _rstar_median(prefix: str, sources: SourceSet) -> pd.Series:
    """Return the median r* from a completed rstar run, or an empty series."""
    try:
        results = load_bonds_results(prefix=prefix)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"  note: rstar output unavailable ({type(exc).__name__}); the 'rstar' variant will be skipped")
        return pd.Series(dtype=float)
    sources.add("NY Fed")
    return results.rstar_median()


def _rule_rstar_median(prefix: str, sources: SourceSet) -> pd.Series:
    """Return the *real* neutral rate implied by a completed `rstar_rba` run.

    Read off that model's recorded `neutral_real`: the slow intercept `b_t` less
    the target. NOT `prescribed_real`, which is `b_t + lambda·g_t` and carries
    the RBA's own inflation response on top, so a rate gap against it is close
    to that model's rule residual: high-frequency timing noise that fits nothing
    and says nothing.

    This variant used to read `prescribed_real`, which made it the least
    informative of the four. On `neutral_real` it becomes the strongest
    relationship in the package and the most wrongly signed. That is not an IS
    curve. `b_t` is a slow-moving neutral, so the rate gap against it is the
    policy STANCE, and the RBA sets a positive stance when the economy is
    running hot. The regression therefore recovers the reaction function with
    the sign reversed, which is the simultaneity this package exists to
    document. Read it as evidence for the central negative finding, not against
    it.
    """
    try:
        trace, frame, constants = load_rba_results(prefix=prefix)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"  note: rstar_rba output unavailable ({type(exc).__name__}); "
              "the 'rule' variant will be skipped")
        return pd.Series(dtype=float)

    recorded = SourceSet.from_records(constants.get("sources"))
    if recorded is not None:
        for source, cat in recorded.records:
            sources.add(source, cat)

    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    # Recorded by the model, so the deflator is its choice and not guessed at
    # here. Older traces predate it and are rebuilt from the nominal series.
    if "neutral_real" in getattr(trace, "posterior", {}):
        return posterior_median(trace, "neutral_real", index)
    nominal = posterior_median(trace, "neutral", index)
    return nominal - float(constants.get("anchor", 2.5))


def _as_quarterly(series: pd.Series) -> pd.Series:
    """Return `series` on a quarterly PeriodIndex."""
    index = series.index
    if isinstance(index, pd.PeriodIndex):
        series.index = index.asfreq("Q")
    elif isinstance(index, pd.DatetimeIndex):
        series.index = index.to_period("Q")
    else:
        raise TypeError(f"expected a DatetimeIndex or PeriodIndex, got {type(index).__name__}")
    return series.astype(float)


def build_observations(
    start: str = DEFAULT_START,
    end: str | None = None,
    *,
    joint_prefix: str = "ystar_ustar",
    rstar_prefix: str = "rstar_bonds",
    rule_prefix: str = "rstar_rba",
    exclude_windows: Sequence[tuple[str, str]] | None = DEFAULT_WINDOWS,
) -> IsCurveData:
    """Assemble the output gap and the three real-rate variants.

    Args:
        start: first quarter to keep
        end: last quarter to keep, or None for the latest available
        joint_prefix: prefix of the saved `ystar_ustar` run supplying the gap
        rstar_prefix: prefix of the saved `rstar` run supplying r*
        rule_prefix: prefix of the saved `rstar_rba` run supplying the
            reaction-function neutral `b_t`, read as `neutral_real`
        exclude_windows: (first, last) windows to leave out of the fits, or
            None to keep every quarter

    Returns:
        IsCurveData with both sides aligned on their common quarters

    """
    sources = SourceSet()

    print("Loading:")
    gap = _as_quarterly(_joint_gap(joint_prefix, sources))
    print(f"  {'output gap':<28}joint ystar_ustar run ({joint_prefix})")

    cash = _as_quarterly(sources.take(get_cash_rate_qrtly()))
    print(f"  {'cash rate':<28}RBA F1")

    expectations = _as_quarterly(sources.take(get_model_expectations_unanchored()))
    print(f"  {'inflation expectations':<28}expectations model (unanchored)")

    rstar = _as_quarterly(_rstar_median(rstar_prefix, sources))
    if not rstar.empty:
        print(f"  {'r*':<28}rstar run ({rstar_prefix})")

    rule_rstar = _as_quarterly(_rule_rstar_median(rule_prefix, sources))
    if not rule_rstar.empty:
        print(f"  {'r* (reaction function)':<28}rstar_rba run ({rule_prefix})")

    # The common window: every variant is plotted on the same quarters, so the
    # three charts differ only in what has been subtracted, not in the sample.
    common = gap.index.intersection(cash.index).intersection(expectations.index)
    if not rstar.empty:
        common = common.intersection(rstar.index)
    common = common[common >= pd.Period(start, freq="Q")]
    if end is not None:
        common = common[common <= pd.Period(end, freq="Q")]
    common = common.sort_values()

    gap, cash, expectations = gap.reindex(common), cash.reindex(common), expectations.reindex(common)
    rstar = rstar.reindex(common) if not rstar.empty else pd.Series(dtype=float)

    real_cash = (cash - expectations).dropna()
    common = common.intersection(real_cash.index)
    gap, real_cash = gap.reindex(common).dropna(), real_cash.reindex(common)
    common = common.intersection(gap.index)
    gap, real_cash = gap.reindex(common), real_cash.reindex(common)

    constant_rstar = float(real_cash.mean())
    variants = {
        "none": real_cash,
        "constant": real_cash - constant_rstar,
    }
    if not rstar.empty:
        variants["rstar"] = real_cash - rstar.reindex(common)
    if not rule_rstar.empty:
        variants["rule"] = real_cash - rule_rstar.reindex(common)

    # `intersection` widens the static type back to Index; the runtime object
    # is still quarterly, and the exclusion arithmetic needs that guaranteed.
    if not isinstance(common, pd.PeriodIndex):
        raise TypeError(f"expected a PeriodIndex after alignment, got {type(common).__name__}")
    excluded = _excluded_quarters(common, exclude_windows)

    print(f"\n  {len(common)} quarters, {common[0]} to {common[-1]}")
    print(f"  flat r* for the 'constant' variant: {constant_rstar:.2f}% (mean real cash rate)")
    if len(excluded):
        windows = ", ".join(f"{first}-{last}" for first, last in (exclude_windows or ()))
        print(f"  excluding {len(excluded)} quarters: {windows}")
        for block in blocks(common, excluded):
            print(f"    kept block: {block[0]} to {block[-1]} ({len(block)} quarters)")

    return IsCurveData(
        gap=gap,
        real_cash=real_cash,
        rstar=rstar.reindex(common) if not rstar.empty else rstar,
        variants={name: variants[name] for name in VARIANTS if name in variants},
        constant_rstar=constant_rstar,
        excluded=excluded,
        sources=sources,
    )
