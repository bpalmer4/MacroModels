"""Okun's Law equation linking output gap to unemployment.

Simple form:
    ΔU_t = β × OG_t + ε

The output gap directly affects the change in unemployment. A positive output gap
(economy above potential) leads to falling unemployment; a negative gap leads to
rising unemployment.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def okun_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,  # noqa: ARG001
    potential_output: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Simple Okun's Law.

    ΔU_t = β × OG_t + ε

    Where:
        - ΔU is the observed quarterly change in unemployment rate (pp)
        - OG = output_gap = (log_gdp - potential_output)
        - β is the Okun coefficient (negative: positive gap → falling U)

    Interpretation:
        - β ≈ -0.18: 1% output gap → -0.18pp effect on ΔU

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP
            - "ΔU": Change in unemployment rate
        model: PyMC model context
        nairu: NAIRU latent variable (unused, kept for API compatibility)
        potential_output: Potential output latent variable
        constant: Optional fixed values for coefficients

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "beta_okun": {"mu": -0.18, "sigma": 0.1, "upper": 0},
            "epsilon_okun": {"sigma": 0.3},
        }
        mc = set_model_coefficients(model, settings, constant)

        # Output gap: (Y - Y*) in log terms
        output_gap = inputs["log_gdp"] - potential_output

        # Okun's Law: ΔU_t = β × OG_t + ε
        pm.Normal(
            "okun",
            mu=mc["beta_okun"] * output_gap,
            sigma=mc["epsilon_okun"],
            observed=inputs["ΔU"],
        )
