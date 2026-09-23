"""Potential output state equation for the HLW r-star model.

Adds latent `potential_output` (y*_t, log x 100) to the latents dict, with
drift driven by the latent annualised trend growth g_t.
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.nairu.base import set_model_coefficients


def potential_output_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    *,
    sigma_ystar: float | pt.TensorVariable,
    constant: dict[str, Any] | None = None,
) -> str:
    """Potential output with time-varying trend drift.

    Model: y*_t = y*_{t-1} + g_{t-1}/4 + e_ystar,  e_ystar ~ N(0, sigma_ystar)

    log_gdp is in log x 100 units; g is annualised %, so quarterly log
    growth in those units is g/4. Drift uses g_{t-1} (HLW lag form) to
    avoid simultaneity with the y* innovation.

    `sigma_ystar` IS CREATED BY THE CALLER, not here. It is hoisted into
    `estimate.py` because lambda_g ties sigma_g to it and trend growth is
    built first, so a ratio formed here would reference something that does
    not yet exist. It arrives as a float when imposed and as a random
    variable when estimated, and either works as the innovation scale.

    WHY IT IS USUALLY IMPOSED. Left free under a wide prior, the posterior
    piles up: at a HalfNormal scale of 0.55 it reached 1.112 and potential
    moved with a quarterly sd of 1.073 against log GDP's own 0.947, so
    POTENTIAL WAS MORE VOLATILE THAN OUTPUT and the implied gap had sd 1.92
    over a range of -3.0 to +7.6. That is not a trend-and-cycle
    decomposition, and every coefficient measured against it was measured
    against a gap that is not a gap.

    TIGHTENING THE PRIOR DOES NOT HOLD IT. Under HalfNormal(0.12), whose
    median of 0.081 matches the 0.078 that `ystar` imposes, the posterior
    still lands around 0.85, some seven standard deviations into the tail:
    the likelihood would rather let potential chase GDP than explain it, and
    a prior cannot win an argument the likelihood insists on.

    So the level is imposed at 0.078 by default. That is HLW's device applied
    to one variance rather than to the ratio it belongs to, and it has a
    price: the variance reappears in trend growth, where sigma_g returns
    0.124 against a prior scale of 0.04. Imposing lambda_g with this
    parameter free is the configuration that constrains the pair as HLW do.
    See the note at the top of `z_star.py` for the same gap on lambda_z.
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "initial_potential": {
                "mu": float(obs["log_gdp"][0]),
                "sigma": 2.0,
            },
        }
        mc = set_model_coefficients(model, settings, constant)

        g = latents["trend_growth"]
        n_periods = len(obs["log_gdp"])

        if isinstance(sigma_ystar, float):
            # Imposed scale: the innovations have a constant width, so there is
            # no funnel and the centred form samples fine.
            innovations = pm.Normal(
                "potential_innovations",
                mu=0,
                sigma=sigma_ystar,
                shape=n_periods - 1,
            )
        else:
            # Estimated scale: centred, every innovation's width is a parameter,
            # and the sampler has to move through a neck that narrows as
            # sigma_ystar falls. That is the funnel behind Resolution S's first
            # stage. Drawing standard normals and scaling them afterwards gives
            # the same model with the dependence taken out of the geometry.
            raw = pm.Normal(
                "potential_innovations_raw",
                mu=0,
                sigma=1.0,
                shape=n_periods - 1,
            )
            innovations = pm.Deterministic(
                "potential_innovations", sigma_ystar * raw,
            )

        # Quarterly drift from lagged trend growth (g_{t-1}/4 in log x 100 units)
        drift = g[:-1] / 4

        init_value = mc["initial_potential"]
        cumulative_growth = pt.cumsum(drift + innovations)
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[init_value], init_value + cumulative_growth]),
        )

    latents["potential_output"] = potential_output
    latents["sigma_ystar"] = sigma_ystar
    sigma_desc = (
        f"sigma_ystar = {sigma_ystar:.3f} imposed"
        if isinstance(sigma_ystar, float)
        else "sigma_ystar sampled"
    )
    return f"y*_t = y*_{{t-1}} + g_{{t-1}}/4 + e_ystar,  e ~ N(0, sigma_ystar);  {sigma_desc}"
