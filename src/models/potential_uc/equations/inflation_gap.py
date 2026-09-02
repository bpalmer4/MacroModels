"""Potential output with the gap defined by inflation, and GDP fitted.

Three lines:

    y*_t      = y*_{t-1} + g_{t-1} + e_y      potential: a random walk (potential.py)
    gap_t     = c · (pi_t - anchor)           gap: defined by inflation
    log_gdp_t = y*_t + gap_t + e_c            output: fitted, with a residual

The gap is proportional to inflation's deviation from target. Inflation supplies
the sign, the timing and the relative magnitude: six per cent inflation means a
bigger gap than three. What it cannot supply is `c`, the conversion from
percentage points of inflation into per cent of output, because nothing in a
price index is denominated in units of GDP. `c` is estimated here.

**The third line is what makes this a model rather than a definition.** An
earlier version set `y* = log_gdp - c·d` outright. That reproduced GDP exactly
by construction, so there was no residual, no fit, and nothing the model could
get wrong: every wiggle in output that inflation did not account for was forced
into potential, and potential duly tracked GDP through the pandemic. With `e_c`
present, trend and inflation-implied gap no longer have to exhaust output, and
what they miss is measured instead of absorbed.

`sigma_e` is therefore the diagnostic the exercise needs. It says how much
Australian output variation the inflation-defined gap fails to explain, which is
the amplitude question answered with a number rather than an argument.

Note what is *not* here. No Phillips curve: `c` is not estimated by regressing
inflation on the gap. No IS curve and no policy rule. And no cycle dynamics:
`e_c` is white noise, so nothing is imposed about how the unexplained part of
output behaves over time, which is the opposite of assuming an AR(2) cycle.

One identification point. `sigma_ystar` and `sigma_e` compete for the same
variation, since a quarterly wiggle in GDP can be a shift in potential or a
residual. That is the Stock-Watson pile-up problem in its proper form, and one
of the two must be pinned. `sigma_ystar` is the one imposed (see `scale.py`),
which is what "potential is smooth" means operationally; `sigma_e` is free so
that it can report what is left over.
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.potential_uc.base import set_model_coefficients


def inflation_gap_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Gap as a scaled inflation deviation, with GDP fitted around it.

    Model: gap_t = c · (pi_t - anchor)
           log_gdp_t = y*_t + gap_t + e_c,   e_c ~ N(0, sigma_e)

    `anchor` must be supplied via `constant`. `latents` must already carry
    `potential_output` (see `potential.py`).
    """
    if constant is None:
        constant = {}

    if "anchor" not in constant:
        raise ValueError("inflation_gap_equation requires a fixed 'anchor'")
    if "potential_output" not in latents:
        raise ValueError("inflation_gap_equation requires potential_output — run potential.py first")

    anchor = float(constant["anchor"])
    deviation = np.asarray(obs["pi"], dtype=float) - anchor

    with model:
        # c is per cent of output per percentage point of inflation deviation.
        # Positive by construction: above-target inflation means output above
        # potential, which is the proposition being tested. The prior is
        # deliberately weak, since c is what the exercise is about.
        settings = {
            "c": {"sigma": 2.0},
            "sigma_e": {"sigma": 1.0},
        }
        mc = set_model_coefficients(model, settings, constant)

        potential_output = latents["potential_output"]
        output_gap = pm.Deterministic("output_gap", mc["c"] * deviation)

        pm.Normal(
            "observed_gdp",
            mu=potential_output + output_gap,
            sigma=mc["sigma_e"],
            observed=obs["log_gdp"],
        )

        pm.Deterministic("inflation_deviation", pt.as_tensor_variable(deviation))

    latents["output_gap"] = output_gap

    return f"gap_t = c · (pi_t - {anchor:g});  log_gdp_t = y*_t + gap_t + e_c"
