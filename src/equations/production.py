"""Production function equations for potential output."""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.base import set_model_coefficients


def potential_output_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    constant: dict[str, Any] | None = None,
) -> pm.Distribution:
    """Potential output with Cobb-Douglas production function.

    Potential growth equals Cobb-Douglas drift plus small innovation:
        g_potential_t = drift_t + ε_t

    Where drift is from Cobb-Douglas:
        drift_t = α × capital_growth_t + (1-α) × lf_growth_t + mfp_growth_t

    With tight innovation variance, potential is driven almost entirely by
    supply-side fundamentals (capital, labor, MFP).

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP (for initial value)
            - "capital_growth": Quarterly capital stock growth
            - "lf_growth": Quarterly labor force growth
            - "mfp_growth": Multi-factor productivity growth
        model: PyMC model context
        constant: Optional fixed values. Keys:
            - "alpha_capital": Fix capital share
            - "potential_innovation": Fix innovation std dev
            - "initial_potential": Fix initial potential output

    Returns:
        pm.Distribution: Potential output latent variable (log scale)

    Example:
        potential = potential_output_equation(inputs, model)

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "alpha_capital": {"mu": 0.3, "sigma": 0.05},
            # Innovation allows potential to absorb some high-frequency noise
            # while staying smooth through recessions
            "potential_innovation": {"mu": 0.05, "sigma": 0.02},
            "initial_potential": {
                "mu": inputs["log_gdp"][0],
                "sigma": inputs["log_gdp"][0] * 0.1,
            },
        }
        mc = set_model_coefficients(model, settings, constant)

        alpha = mc["alpha_capital"]

        # Cobb-Douglas drift: α×g_K + (1-α)×g_L + g_MFP
        drift = (
            alpha * inputs["capital_growth"]
            + (1 - alpha) * inputs["lf_growth"]
            + inputs["mfp_growth"]
        )

        init_value = mc["initial_potential"]

        # Build potential output: growth = drift + innovation
        n_periods = len(inputs["log_gdp"])
        innovations = pm.Normal(
            "potential_innovations",
            mu=0,
            sigma=mc["potential_innovation"],
            shape=n_periods - 1,
        )

        # Growth rates: g_t = drift_t + ε_t
        growth_rates = drift[1:] + innovations

        # Cumulative: Y*_t = Y*_0 + Σ(g_i)
        cumulative_growth = pt.cumsum(growth_rates)
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[init_value], init_value + cumulative_growth]),
        )

        # Expose growth rates for diagnostics
        pm.Deterministic(
            "potential_growth",
            pt.concatenate([[drift[0]], growth_rates]),
        )

    return potential_output
