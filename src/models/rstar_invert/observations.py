"""Observation assembly: the output gap and the real cash rate, aligned.

Nothing new is loaded here. Both series come from `is_curve.observations`, which
already assembles exactly this pair, so the two packages cannot drift apart on
the deflator, the exclusion windows or the vintage of the `ystar_ustar` run.
`is_curve` plots the pair and fits a line through it; this package inverts a
structural equation on it instead.

THE GAP IS A MODEL OUTPUT, taken at its posterior median, so this package
inherits everything `ystar_ustar` conditions on, including that model's imposed
`sigma_okun` of 0.20. In the current run roughly half of it is
`c x (inflation - 2.5)` with c = 0.275, so the left-hand side of the IS curve is
substantially a rescaling of inflation's deviation from target. That matters for
reading the result: a relationship recovered between the rate and THAT is part
reaction function.

Excluded quarters are CARRIED, not dropped. The rate enters lagged, so removing
rows would make the lag step across the hole and pair a gap with the wrong
quarter's rate. The likelihood mask is applied after the lag is built.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.common.sources import SourceSet
from src.models.is_curve.observations import (
    DEFAULT_WINDOWS,
    GFC_TO_PANDEMIC_WINDOW,
)
from src.models.is_curve.observations import (
    build_observations as build_is_curve_observations,
)
from src.models.rstar_invert.config import ModelConfig


@dataclass
class InversionData:
    """The two observed series, plus the lagged rate and the likelihood mask.

    Attributes:
        frame: `gap` and `real_cash` on the common quarterly index
        rate_lagged: (n, len(config.rate_lags)) real cash rate at each lag,
            NaN-filled at the start; one column per lag, in config order
        usable: rows carrying a likelihood contribution
        excluded: quarters removed by the exclusion windows
        sources: what was loaded, for the chart footers

    """

    frame: pd.DataFrame
    rate_lagged: np.ndarray
    usable: np.ndarray
    excluded: pd.PeriodIndex
    sources: SourceSet = field(default_factory=SourceSet)

    @property
    def index(self) -> pd.PeriodIndex:
        """The quarters the model runs over."""
        index = self.frame.index
        if not isinstance(index, pd.PeriodIndex):
            raise TypeError(f"expected a PeriodIndex, got {type(index).__name__}")
        return index


def _lagged(values: np.ndarray, lag: int) -> np.ndarray:
    """Return `values` shifted back `lag` quarters, NaN-filled at the start."""
    out = np.full_like(values, np.nan, dtype=float)
    if lag < len(values):
        out[lag:] = values[:-lag]
    return out


def build_observations(
    config: ModelConfig | None = None,
    *,
    verbose: bool = True,
) -> InversionData:
    """Assemble the gap, the real cash rate and the lagged rate.

    Args:
        config: the specification; a default one is built if omitted
        verbose: print what was loaded and what was excluded

    Returns:
        InversionData with both series aligned and the likelihood mask built

    """
    config = config or ModelConfig()

    windows = list(DEFAULT_WINDOWS)
    if config.exclude_qe:
        windows.append(GFC_TO_PANDEMIC_WINDOW)

    # The "none" variant is the raw real cash rate with no r* subtracted, which
    # is what this model needs: subtracting another model's r* here would be
    # conditioning the answer on the answer.
    data = build_is_curve_observations(
        start=config.start,
        end=config.end,
        joint_prefix=config.gap_prefix,
        exclude_windows=tuple(windows),
    )

    frame = pd.DataFrame({"gap": data.gap, "real_cash": data.variants["none"]})
    gap = frame["gap"].to_numpy(dtype=float)
    rate = frame["real_cash"].to_numpy(dtype=float)
    rate_lagged = np.column_stack([_lagged(rate, lag) for lag in config.rate_lags])

    kept = ~frame.index.isin(data.excluded)
    # Every lag the equation reaches for must exist, so the longest one sets
    # how many quarters are lost at the start.
    usable = np.isfinite(rate_lagged).all(axis=1) & np.isfinite(gap) & kept

    if verbose:
        lags = ", ".join(f"t-{lag}" for lag in config.rate_lags)
        print(f"\n  {len(frame)} quarters, {frame.index[0]} to {frame.index[-1]}")
        print(f"  rate enters at {lags}")
        print(f"  {int(usable.sum())} quarters carry a likelihood contribution "
              f"({max(config.rate_lags)} lost to the longest lag, "
              f"{int(len(frame) - kept.sum())} to exclusions)")
        print(f"  output gap sd {np.nanstd(gap):.3f}pp, real cash rate sd {np.nanstd(rate):.3f}pp, "
              f"mean {np.nanmean(rate):.3f}pp")

    return InversionData(
        frame=frame,
        rate_lagged=rate_lagged,
        usable=usable,
        excluded=data.excluded,
        sources=data.sources,
    )
