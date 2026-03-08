"""Potential output state equation (Cobb-Douglas production function).

State equation that adds 'potential_output' and 'potential_growth' to latents.
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
    """Potential output with Cobb-Douglas and Gaussian innovations.

    Model: Y*_t = Y*_{t-1} + alpha x dK + (1-alpha) x dL + dMFP + e
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "alpha_capital": {"mu": 0.3, "sigma": 0.05},
            "potential_innovation": {"mu": 0.1, "sigma": 0.05},
            "initial_potential": {
                "mu": obs["log_gdp"][0],
                "sigma": obs["log_gdp"][0] * 0.1,
            },
        }
        mc = set_model_coefficients(model, settings, constant)

        alpha = mc["alpha_capital"]
        drift = (
            alpha * obs["capital_growth"]
            + (1 - alpha) * obs["lf_growth"]
            + obs["mfp_growth"]
        )

        init_value = mc["initial_potential"]
        n_periods = len(obs["log_gdp"])
        innovations = pm.Normal(
            "potential_innovations",
            mu=0,
            sigma=mc["potential_innovation"],
            shape=n_periods - 1,
        )

        cumulative_growth = pt.cumsum(drift[1:] + innovations)
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[init_value], init_value + cumulative_growth]),
        )

        growth_rates = pt.concatenate([[drift[0]], drift[1:] + innovations])
        pm.Deterministic("potential_growth", growth_rates)

    latents["potential_output"] = potential_output
    return "Y*_t = Y*_{t-1} + alpha x dK + (1-alpha) x dL + dMFP + e,  e ~ N(0, sigma)"


def potential_output_skewnormal_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Potential output with Cobb-Douglas and SkewNormal innovations.

    Model: Y*_t = Y*_{t-1} + alpha_t x dK + (1-alpha_t) x dL + dMFP + e,  e ~ SkewNormal(alpha=1)
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "potential_innovation": {"mu": 0.05, "sigma": 0.02},
            "initial_potential": {
                "mu": obs["log_gdp"][0],
                "sigma": obs["log_gdp"][0] * 0.1,
            },
        }
        mc = set_model_coefficients(model, settings, constant)

        alpha = obs["alpha_capital"]
        drift = (
            alpha * obs["capital_growth"]
            + (1 - alpha) * obs["lf_growth"]
            + obs["mfp_growth"]
        )

        init_value = mc["initial_potential"]
        n_periods = len(obs["log_gdp"])
        skew_alpha = 1.0
        mean_offset = np.sqrt(2 / np.pi) * skew_alpha / np.sqrt(1 + skew_alpha**2)
        innovations = pm.SkewNormal(
            "potential_innovations",
            mu=-mc["potential_innovation"] * mean_offset,
            sigma=mc["potential_innovation"],
            alpha=skew_alpha,
            shape=n_periods,
        )

        growth_rates = drift + innovations
        cumulative_growth = pt.cumsum(growth_rates[1:])
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[init_value], init_value + cumulative_growth]),
        )
        pm.Deterministic("potential_growth", growth_rates)

    latents["potential_output"] = potential_output
    return "Y*_t = Y*_{t-1} + alpha_t x dK + (1-alpha_t) x dL + dMFP + e,  e ~ SkewNormal(alpha=1)"
