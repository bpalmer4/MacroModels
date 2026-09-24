"""The natural cubic B-spline basis u* is built from, with knots at the regime dates.

This replaces the attractor state law. There, u* was a random walk pulled
towards one free constant per regime, which had two defects: the innovation sd
`sigma_ustar` was imposed and nothing measured it, and an exponential approach
to a level can only do one shape, so a regime spanning a peak and a descent
came out flat across both.

A spline fixes both. u* becomes deterministic given the coefficients, so the
imposed variance disappears rather than being chosen, and each segment gets a
cubic, which can ratchet, peak and roll over, or run an S-curve.

**It is a fit, not a filter.** The coefficients are estimated from the Phillips
likelihood, so the shape within each period is a finding. That is the
difference from putting a Henderson through the inversion, where the filter
decides the shape and the data only supply scatter around it. Henderson also
has negative outer weights, so on a noisy input its side lobes manufacture
oscillations at roughly the filter's own length; a least-squares fit of a
low-order polynomial has no side lobes and cannot invent a cycle. It can only
be too stiff to follow a real turn, which is a visible failure rather than an
invented feature.

**What it does not do** is add information. The likelihood is still one
equation on one series, so `u - u*` remains the inflation gap scaled by
`u/beta`. The spline changes how u* is parameterised, not what it is measured
against.

**Knot multiplicity.** A repeated knot drops the continuity there. See
`basis`; 1974Q1 carries multiplicity 3 by default, so the 1960s are not forced
to bend upward to meet the ratchet smoothly.

**Natural, not plain cubic.** Beyond the outer knots the curve is forced
linear. Without that the last segment, 2020Q1 onwards with no knot after it,
extrapolates as a free cubic against the largest inflation surprises in the
sample, and cubics are at their worst at a boundary.
"""

from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

DEGREE = 3

# A segment shorter than this has no shape to describe: a single quarter is a point.
_MIN_SEGMENT_QUARTERS = 2


def _augmented_knots(interior: np.ndarray, lo: float, hi: float, degree: int) -> np.ndarray:
    """Return the full knot vector: the interior knots with clamped boundaries."""
    return np.concatenate([np.repeat(lo, degree + 1), interior, np.repeat(hi, degree + 1)])


def basis(
    index: pd.PeriodIndex,
    breaks: tuple[str, ...],
    *,
    natural: bool = True,
    multiplicity: dict[str, int] | None = None,
    degree: int = DEGREE,
) -> np.ndarray:
    """Return the (T x J) spline basis evaluated on the sample's own quarters.

    Time is the integer position in the sample rather than a calendar value, so
    the basis does not depend on where the sample happens to start.

    `degree` is the polynomial degree, 3 (cubic) by default. The fitted
    growth path is one degree below the level, so a degree-4 basis with no
    interior knots gives a growth rate that is a single smooth cubic in time:
    it can rise, peak, fall and then flatten, which a cubic level cannot,
    and it does so without a knot date having to be chosen.

    `natural` imposes zero second derivative at both ends by reducing the basis
    rather than by adding a penalty: the two boundary-adjacent columns are
    folded into their neighbours so the fitted curve is linear beyond the outer
    knots. That costs two coefficients and buys an end segment that cannot
    swing.
    """
    t = np.arange(len(index), dtype=float)
    multiplicity = multiplicity or {}
    positions = []
    for b in breaks:
        # get_loc returns a slice or a mask for a non-unique index; a quarterly
        # PeriodIndex here is unique, so narrow rather than assume.
        where = index.get_loc(pd.Period(b, freq="Q"))
        if not isinstance(where, int):
            raise TypeError(f"break {b} does not sit on a single quarter of the sample")
        # A knot of multiplicity m in a degree-3 spline gives C^(3-m)
        # continuity there: 1 is the usual C2, 2 lets curvature kink, 3 leaves
        # only the level matched and frees the slope to turn a corner.
        #
        # 1974Q1 wants 3. Forcing matched slope and curvature across it makes
        # the 1960s bend upward years early, because a spline cannot turn a
        # corner and so has to begin the turn before the knot. The ratchet was
        # a break, not a transition, and a leap has a corner in it. C0 still
        # means u* does not jump, which is what "aligned handoff" requires.
        positions.extend([float(where)] * max(1, multiplicity.get(b, 1)))
    knots = _augmented_knots(np.asarray(positions, dtype=float), t[0], t[-1], degree)

    n_basis = len(knots) - degree - 1
    columns = []
    for j in range(n_basis):
        coef = np.zeros(n_basis)
        coef[j] = 1.0
        columns.append(BSpline(knots, coef, degree, extrapolate=False)(t))
    design = np.nan_to_num(np.column_stack(columns), nan=0.0)

    if natural:
        # Fold the second basis function at each end into the first, which
        # removes the curvature the boundary pair would otherwise supply.
        design[:, 0] += design[:, 1]
        design[:, -1] += design[:, -2]
        design = np.delete(design, [1, design.shape[1] - 2], axis=1)

    return design


def segment_shapes(index: pd.PeriodIndex, ustar: pd.Series, breaks: tuple[str, ...]) -> pd.DataFrame:
    """Describe each segment's fitted shape: where it starts, ends, and turns.

    The point of giving each period a cubic rather than a level is that it can
    turn inside the period. This is the table that says whether any of them
    did, which is the difference between the spline earning its extra
    coefficients and merely spending them.
    """
    cuts = [index[0], *[pd.Period(b, freq="Q") for b in breaks], index[-1]]
    rows = []
    for lo, hi in pairwise(cuts):
        seg = ustar.loc[lo:hi]
        if len(seg) < _MIN_SEGMENT_QUARTERS:
            continue
        slope = seg.diff()
        turns = int((np.sign(slope).diff().fillna(0) != 0).sum() - 1)
        rows.append({
            "segment": f"{seg.index[0]}-{seg.index[-1]}",
            "quarters": len(seg),
            "start": float(seg.iloc[0]),
            "end": float(seg.iloc[-1]),
            "min": float(seg.min()),
            "max": float(seg.max()),
            "turns": max(turns, 0),
        })
    return pd.DataFrame(rows)
