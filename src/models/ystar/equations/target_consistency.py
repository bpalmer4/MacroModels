"""Target-consistency identification: inflation's sign, at an estimated lag.

The proposition this implements:

    inflation subsequently above target  =>  the economy is running above potential
    inflation subsequently below target  =>  the economy is running below potential
    inflation subsequently at target     =>  the economy is at potential

and nothing more. In particular it makes **no claim about magnitude**: it does
not say that 1pp of excess inflation implies any particular size of output gap.
That is the difference between this and a Phillips curve, and it is the point.
A Phillips slope cannot be recovered from an inflation-targeting sample, because
what is observed is the sum of the gap's effect on inflation and the RBA's
response to the gap. On 1993Q1-2026Q2 data the correlation between the HP(1600)
gap and the trimmed mean deviation peaks at +0.28 after one quarter and is
**-0.29** by eight. The sign of a deviation at the right lag survives that; a
slope does not.

**The lag is estimated, not chosen.** Policy moves output before it moves
prices, so matching gap_t against inflation in the same quarter is wrong. Rather
than pick a lag by hand, the restriction uses a weighted average over lags
0..L with the weights free:

    d_t = sum_k w_k · (pi_{t+k} - anchor),    w ~ Dirichlet(1),  k = 0..L

The posterior over `w` is then the model's answer to "how long does it take",
and is reported as `pi_lag_weights`.

Implementation: a split (two-piece) normal prior on the gap.

    consistent side:   sd = s0 + k·|d_t|      permissive, magnitude is free
    inconsistent side: sd = s0                tight, this is the restriction

When inflation subsequently runs above target the gap may be as positive as the
data like, but is pushed away from being negative; below target, the reverse;
at target, both halves are `s0` and the gap is held near zero.

Two numerical points that matter:

- The switch between halves uses sigmoids rather than hard signs, so the log
  density is differentiable and NUTS does not have to negotiate a kink.
- Because `d` is now latent, the split normal's normalising constant depends on
  the parameters and **cannot be dropped**. Left out, the sampler would put all
  the weight on whichever lag makes |d| largest, since a wider permissive side
  is a cheaper likelihood. Conveniently sL + sR = 2·s0 + k·|d| exactly, so the
  constant is log(2·s0 + k·|d|) and is cheap to carry.

`s0` and `k` are imposed, not estimated, as every variance in this model is
(see `scale.py`). The weights are the one thing here the data get to choose.

**An earlier version used |d_t| only and discarded the sign.** That was wrong.
A term that can only ever say "the gap is zero", loudly or quietly, is a
smoothness penalty by another name, and it duly reproduced an HP(1600) cycle at
a correlation of 0.991.
"""

import math
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Width of the sigmoid switches, in the units of the quantity being switched on.
# Small enough to act as a sign, large enough to keep the gradient smooth.
_SWITCH_GAP = 0.25  # gap, in log x 100
_SWITCH_DEV = 0.25  # inflation deviation, in percentage points

_LOG_NORM = math.log(2.0) - 0.5 * math.log(2.0 * math.pi)


def target_consistency_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Restrict the sign of the output gap to match inflation's later deviation from target.

    Model: d_t  = sum_k w_k · (pi_{t+k} - anchor),   w ~ Dirichlet(1)
           gap_t ~ SplitNormal(0; consistent sd = s0 + k·|d_t|,
                                  inconsistent sd = s0)

    `anchor`, `gap_sd_on_target` (s0), `gap_sd_per_pp` (k) and `pi_lag_max` (L)
    must all be supplied via `constant`. Only the lag weights are estimated.
    """
    if constant is None:
        constant = {}

    required = ("anchor", "gap_sd_on_target", "gap_sd_per_pp", "pi_lag_max")
    missing = [name for name in required if name not in constant]
    if missing:
        raise ValueError(f"target_consistency_equation requires fixed {missing} — pass them via constant")

    # Scalar or one value per quarter: a phased anchor is a series, because
    # before target adoption "at target" is not a statement that can be made.
    anchor = np.asarray(constant["anchor"], dtype=float)
    s0 = float(constant["gap_sd_on_target"])
    k = float(constant["gap_sd_per_pp"])
    lag_max = int(constant["pi_lag_max"])
    if s0 <= 0:
        raise ValueError(f"gap_sd_on_target must be positive, got {s0}")
    if k < 0:
        raise ValueError(f"gap_sd_per_pp must be non-negative, got {k}")
    if lag_max < 0:
        raise ValueError(f"pi_lag_max must be non-negative, got {lag_max}")

    deviation_all = np.asarray(obs["pi"], dtype=float) - anchor
    n_periods = deviation_all.shape[0]
    if lag_max >= n_periods:
        raise ValueError(f"pi_lag_max {lag_max} leaves no observations (n={n_periods})")

    # Column j holds the deviation j quarters ahead. Rows are the quarters that
    # have a full set of future readings: the last `lag_max` carry no
    # restriction, which is exactly the quarters of most interest, and is the
    # honest price of allowing a lag at all.
    n_used = n_periods - lag_max
    lagged = np.column_stack([deviation_all[j : j + n_used] for j in range(lag_max + 1)])

    with model:
        if not hasattr(model, "_fixed_constants"):
            model._fixed_constants = {}  # noqa: SLF001 — our own metadata on the PyMC model
        model._fixed_constants.update({  # noqa: SLF001 — matching set_model_coefficients
            # The scalar where there is one; the series is recorded by
            # `estimate._record_anchor`, which is the only place that has it
            # alongside the sample index.
            "anchor": float(anchor) if anchor.ndim == 0 else float(anchor[-1]),
            "gap_sd_on_target": s0,
            "gap_sd_per_pp": k,
            "pi_lag_max": lag_max,
        })

        if lag_max == 0:
            deviation = pt.as_tensor_variable(lagged[:, 0])
        else:
            # Beta ("MIDAS") lag weights: two shape parameters instead of
            # lag_max+1 free ones. Free Dirichlet weights were tried first and
            # did not converge (r_hat 1.53 at L=4): the likelihood is bimodal,
            # with one mode at a contemporaneous relation and another at three
            # to four quarters, and chains settled in different modes. Two
            # parameters spanning decaying, humped and back-loaded shapes are
            # enough to say where the mass sits without that pathology.
            shape_a = 1.0 + pm.HalfNormal("pi_lag_a", sigma=2.0)
            shape_b = 1.0 + pm.HalfNormal("pi_lag_b", sigma=2.0)
            grid = (np.arange(lag_max + 1) + 1.0) / (lag_max + 2.0)
            unnormalised = pt.exp(
                (shape_a - 1.0) * np.log(grid) + (shape_b - 1.0) * np.log(1.0 - grid),
            )
            weights = pm.Deterministic("pi_lag_weights", unnormalised / pt.sum(unnormalised))
            deviation = pt.dot(pt.as_tensor_variable(lagged), weights)

        gap = latents["output_gap"][:n_used]

        # Which side of zero the deviation is on, smoothly.
        above = pm.math.sigmoid(deviation / _SWITCH_DEV)
        widening = k * pt.abs(deviation)
        sd_right = s0 + widening * above
        sd_left = s0 + widening * (1.0 - above)

        # Which side of zero the gap is on, smoothly.
        right = pm.math.sigmoid(gap / _SWITCH_GAP)
        precision = (1.0 - right) / pt.square(sd_left) + right / pt.square(sd_right)

        # sd_left + sd_right = 2*s0 + k*|d| identically, so the split-normal
        # normalising constant is this simple. It must be kept: it is what stops
        # the lag weights drifting toward whichever lag maximises |d|.
        log_norm = _LOG_NORM - pt.log(sd_left + sd_right)

        pm.Potential(
            "target_consistency",
            pt.sum(log_norm - 0.5 * pt.square(gap) * precision),
        )

        pm.Deterministic("inflation_deviation", deviation)

    lag_desc = "contemporaneous" if lag_max == 0 else f"weighted over lags 0..{lag_max}"
    return (
        f"gap_t ~ SplitNormal(0; consistent sd = {s0:g} + {k:g}·|d_t|, "
        f"inconsistent sd = {s0:g}),  d_t {lag_desc}"
    )
