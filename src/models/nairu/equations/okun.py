"""Okun's Law equation linking output gap to unemployment."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def okun_law_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    potential_output: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Okun's Law linking output gap to change in unemployment.

    Change form of Okun's Law:
        ΔU = β × output_gap + ε
        ΔU = β × (Y - Y*) + ε

    Where:
        - ΔU is the observed quarterly change in unemployment rate (pp)
        - output_gap = (log_gdp - potential_output)
        - β is negative: when Y > Y* (positive gap), unemployment falls

    Note: Regime-switching was tested (pre-GFC, post-GFC, post-COVID) but posteriors
    showed no meaningful difference - all regime medians fell within the other regimes'
    90% HDI. A single time-invariant coefficient is used for parsimony.

    Interpretation of β (beta_okun):
        - Expected to be negative
        - Typical values: β ≈ -0.1 to -0.3 for quarterly data
        - E.g., β = -0.2 means 1% output gap → -0.2pp change in unemployment

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP
            - "ΔU": Change in unemployment rate
        model: PyMC model context
        nairu: NAIRU latent variable (not directly used, but keeps interface consistent)
        potential_output: Potential output latent variable
        constant: Optional fixed values for coefficients

    Example:
        okun_law_equation(inputs, model, nairu, potential_output)

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "beta_okun": {"mu": -0.2, "sigma": 0.1, "upper": 0},
            "epsilon_okun": {"sigma": 0.5},
        }
        mc = set_model_coefficients(model, settings, constant)

        # Output gap: (Y - Y*) in log terms
        output_gap = inputs["log_gdp"] - potential_output

        # Okun's Law: ΔU = β × output_gap
        pm.Normal(
            "okun_law",
            mu=mc["beta_okun"] * output_gap,
            sigma=mc["epsilon_okun"],
            observed=inputs["ΔU"],
        )
