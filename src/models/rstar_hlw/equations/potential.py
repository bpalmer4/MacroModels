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
    constant: dict[str, Any] | None = None,
) -> str:
    """Potential output with time-varying trend drift.

    Model: y*_t = y*_{t-1} + g_{t-1}/4 + e_ystar,  e_ystar ~ N(0, sigma_ystar)

    log_gdp is in log x 100 units; g is annualised %, so quarterly log
    growth in those units is g/4. Drift uses g_{t-1} (HLW lag form) to
    avoid simultaneity with the y* innovation.
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "sigma_ystar": {"sigma": 0.55},
            "initial_potential": {
                "mu": float(obs["log_gdp"][0]),
                "sigma": 2.0,
            },
        }
        mc = set_model_coefficients(model, settings, constant)

        g = latents["trend_growth"]
        n_periods = len(obs["log_gdp"])

        innovations = pm.Normal(
            "potential_innovations",
            mu=0,
            sigma=mc["sigma_ystar"],
            shape=n_periods - 1,
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
    latents["sigma_ystar"] = mc["sigma_ystar"]
    return "y*_t = y*_{t-1} + g_{t-1}/4 + e_ystar,  e ~ N(0, sigma_ystar)"
