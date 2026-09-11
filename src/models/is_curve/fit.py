"""Ordinary least squares for the IS-curve scatter, and what its intercept means.

The line is

    gap_t = a + b . x_{t-lag} + e

where `x` is one of the real-rate variants. `b` is the IS curve's slope, the
thing `rstar_hlw` reports as a_r ~ -0.04 across eight specifications.

The intercept is the more interesting number here. Wherever the fitted line
crosses zero output gap, the real rate on that axis is neutral *by the
scatter's own reckoning*:

    x at zero gap = -a / b

Under the `none` variant, x is the plain real cash rate, so that crossing is
an estimate of r* read straight off the data. Under the variants that already
subtract an r*, x is a gap, so the crossing is the amount by which the imposed
r* is too low (positive) or too high (negative). A well-chosen r* puts the
crossing at zero, which is to say the intercept at zero.

The crossing is only meaningful if the line slopes the way an IS curve does.
Two guards, and `crossing` is None unless both pass:

- The slope must be distinguishable from zero. A flat line crosses zero gap
  either nowhere or everywhere, and -a/b explodes as b approaches 0, so the
  slope's t-statistic must reach `MIN_ABS_T`.
- The slope must be *negative*. A higher real rate opening a wider positive
  output gap is not an IS curve, so -a/b is not a natural rate: it is the
  point where a relationship the model does not believe in crosses zero.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Below this |t| on the slope, the zero-gap crossing is not worth quoting: the
# line is too flat for -a/b to mean anything. 2.0 is the conventional cut.
MIN_ABS_T = 2.0

# Lags swept by `lag_sweep`. The IS curve's rate term is lagged because policy
# is not thought to move output within the quarter: `nairu` uses t-2 and HLW
# averages t-1 and t-2.
DEFAULT_LAGS = tuple(range(9))

# SIX QUARTERS, and the reason is the sweep below rather than convention.
#
# This was 2, `nairu`'s choice. It moved because every piece of evidence in the
# package points further out. The slope here strengthens monotonically with the
# lag and only turns negative around 4 to 5 on the full sample, reaching -0.015
# by 6; on the sample that drops 2008Q4-2021Q3 it strengthens to -0.139 at five
# quarters and peaks at lag 6. `rstar_invert`'s single-lag sweep finds the same
# shape inside a state-space model, -0.007 at lag 1 rising to -0.034 at lag 5,
# with the posterior only coming off its sign bound at 5.
#
# The reason is simultaneity rather than fit-chasing. The RBA reacts to
# conditions within a quarter or two while output responds to rates over one to
# two years, so a short lag mostly measures the reaction function and returns
# the WRONG SIGN (+0.081 at lag 0). Reaching further back is a partial fix. It
# is only partial: the real cash rate is persistent, so r_{t-6} stays
# correlated with recent rates that are reacting.
#
# Six is also what `rstar_hlw` and `rstar_invert` now use, so the three are
# directly comparable on the timing. They are still NOT comparable on the
# coefficient: HLW has gap persistence, so its `a_r` is an impact coefficient
# whose level counterpart is a_r/(1 - a_y1 - a_y2), while this bench and
# `rstar_invert` fit level slopes.
DEFAULT_LAG = 6


@dataclass
class FitResult:
    """One OLS fit of the output gap on a lagged real-rate variant."""

    variant: str
    lag: int
    slope: float
    intercept: float
    se_slope: float
    t_slope: float
    r_squared: float
    n: int
    crossing: float | None

    def line(self, x: np.ndarray) -> np.ndarray:
        """Return the fitted values along `x`."""
        return self.intercept + self.slope * x

    def summary_line(self) -> str:
        """Return a one-line report of the fit."""
        crossing = "n/a" if self.crossing is None else f"{self.crossing:+.2f}"
        return (
            f"{self.variant:<9} lag {self.lag}  slope {self.slope:+.3f} "
            f"(t {self.t_slope:+.2f})  intercept {self.intercept:+.3f}  "
            f"R2 {self.r_squared:.3f}  n {self.n:3d}  zero-gap crossing {crossing}"
        )


def fit(
    x: pd.Series,
    y: pd.Series,
    variant: str,
    lag: int = DEFAULT_LAG,
    drop: pd.PeriodIndex | None = None,
) -> FitResult:
    """Regress the output gap on a lagged real-rate variant.

    `drop` is applied *after* the lag, never before. Removing quarters from
    the series first would make `shift` step across the hole and pair a gap
    with a rate from the wrong quarter.

    Args:
        x: the real-rate variant
        y: the output gap
        variant: name of the variant, carried into the result
        lag: quarters by which x is lagged before fitting
        drop: quarters to leave out of the fit, by the gap's own quarter

    Returns:
        FitResult for the fitted line

    """
    frame = pd.DataFrame({"x": x.shift(lag), "y": y}).dropna()
    if drop is not None:
        frame = frame.drop(index=drop, errors="ignore")
    if len(frame) < 3:  # noqa: PLR2004 — two parameters need a third point to have residuals
        raise ValueError(f"{variant}: only {len(frame)} usable quarters at lag {lag}")

    model = sm.OLS(frame["y"], sm.add_constant(frame["x"])).fit()
    slope = float(model.params["x"])
    intercept = float(model.params["const"])
    t_slope = float(model.tvalues["x"])

    is_curve_shaped = slope < 0 and abs(t_slope) >= MIN_ABS_T
    crossing = -intercept / slope if is_curve_shaped else None

    return FitResult(
        variant=variant,
        lag=lag,
        slope=slope,
        intercept=intercept,
        se_slope=float(model.bse["x"]),
        t_slope=t_slope,
        r_squared=float(model.rsquared),
        n=int(model.nobs),
        crossing=crossing,
    )


def lag_sweep(
    x: pd.Series,
    y: pd.Series,
    variant: str,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    drop: pd.PeriodIndex | None = None,
) -> pd.DataFrame:
    """Fit the same line at each lag, so a weak result cannot be a timing mistake.

    Args:
        x: the real-rate variant
        y: the output gap
        variant: name of the variant
        lags: lags to fit
        drop: quarters to leave out of every fit

    Returns:
        DataFrame indexed by lag, with slope, t, R2, n and the crossing

    """
    rows = {}
    for lag in lags:
        result = fit(x, y, variant, lag, drop)
        rows[lag] = {
            "slope": result.slope,
            "t": result.t_slope,
            "intercept": result.intercept,
            "R2": result.r_squared,
            "n": result.n,
            "crossing": np.nan if result.crossing is None else result.crossing,
        }
    frame = pd.DataFrame(rows).T
    frame.index.name = "lag"
    return frame
