"""Output identity and the AR(2) cycle.

Adds latents `potential_output` (y*_t) and `output_gap` (log x 100), and
supplies the GDP observation equation.

Two things happen here.

1. The identity. y* = h* + lp*, and the gap is the residual
   gap_t = log_gdp_t - y*_t. The gap is *not* a separate latent state: making
   it deterministic saves a whole state series and matches the treatment used
   in the project's HLW model. It also means GDP is taken as measured without
   error, which is the standard normalisation — the hours equation carries the
   measurement error instead.

2. The cycle. The gap is restricted to a stationary AR(2), which is what makes
   it a *cycle* rather than a free residual. Without this restriction, any
   path of y* would fit GDP exactly and the trend/cycle split would rest
   entirely on the Phillips curve.

The AR(2) enters as an observation equation on log_gdp, in the same form the
HLW IS curve takes: predicted GDP is potential plus the AR projection of the
two lagged gaps.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.potential_uc.base import set_model_coefficients


def output_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Output identity plus AR(2) cycle dynamics.

    Model: y*_t   = h*_t + lp*_t
           gap_t  = log_gdp_t - y*_t
           log_gdp_t = y*_t + phi_1·gap_{t-1} + phi_2·gap_{t-2} + e_c

    phi priors are centred on a conventional hump-shaped cycle
    (phi_1 ~ 1.3, phi_2 ~ -0.4), which keeps the roots inside the unit circle
    for most of the prior mass without imposing a hard stationarity constraint.
    """
    if constant is None:
        constant = {}

    # With cycle_ar off, the gap is the bare identity log_gdp - y* and nothing
    # restricts it to look like a cycle. That is only sane when some other
    # equation pins the gap down (see `target_consistency.py`); with the
    # Phillips curve alone it would leave y* free to absorb all of output.
    cycle_ar = bool(constant.get("cycle_ar", True))

    with model:
        settings: dict[str, dict[str, float]] = {}
        if cycle_ar:
            settings = {
                "phi_1": {"mu": 1.30, "sigma": 0.30},
                "phi_2": {"mu": -0.40, "sigma": 0.30},
            }
        mc = set_model_coefficients(model, settings, constant)

        # The core (Y, pi) specification builds y* directly in potential.py.
        # The labour specification instead composes it from the two trends.
        composed = "potential_output" not in latents
        if not composed:
            potential_output = latents["potential_output"]
        else:
            potential_output = pm.Deterministic(
                "potential_output",
                latents["trend_hours"] + latents["trend_productivity"],
            )

        output_gap = pm.Deterministic(
            "output_gap",
            obs["log_gdp"] - potential_output,
        )

        if cycle_ar:
            predicted = (
                potential_output[2:]
                + mc["phi_1"] * output_gap[1:-1]
                + mc["phi_2"] * output_gap[:-2]
            )

            pm.Normal(
                "observed_gdp",
                mu=predicted,
                sigma=latents["sigma_c"],
                observed=obs["log_gdp"][2:],
            )

    latents["potential_output"] = potential_output
    latents["output_gap"] = output_gap
    if not cycle_ar:
        return "gap_t = log_gdp_t - y*_t   (identity only, no AR(2) restriction)"
    base = "log_gdp_t = y*_t + phi_1·gap_{t-1} + phi_2·gap_{t-2} + e_c"
    return f"{base},  y* = h* + lp*" if composed else base
