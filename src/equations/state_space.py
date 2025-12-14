"""State-space equations for latent variables."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.base import set_model_coefficients


def nairu_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    constant: dict[str, Any] | None = None,
) -> pm.Distribution:
    """Gaussian random walk for the NAIRU (Non-Accelerating Inflation Rate of Unemployment).

    The NAIRU is modeled as a random walk without drift:
        NAIRU_t = NAIRU_{t-1} + ε_t,  ε_t ~ N(0, σ²)

    This allows the natural rate of unemployment to evolve slowly over time,
    capturing structural changes in the labor market.

    Args:
        inputs: Must contain "U" (unemployment rate array) for dimensioning
        model: PyMC model context
        constant: Optional fixed values. Keys:
            - "nairu_innovation": Fix innovation std dev (recommended ~0.25)
            - "nairu": Fix entire NAIRU series (for counterfactuals)

    Returns:
        pm.GaussianRandomWalk: The NAIRU latent variable

    Example:
        nairu = nairu_equation(inputs, model, constant={"nairu_innovation": 0.25})

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "nairu_innovation": {"mu": 0.3, "sigma": 0.1},
        }
        mc = set_model_coefficients(model, settings, constant)

        nairu = (
            pm.GaussianRandomWalk(
                "nairu",
                mu=0,  # no drift
                sigma=mc["nairu_innovation"],
                init_dist=pm.Normal.dist(mu=15.0, sigma=8.0),
                steps=len(inputs["U"]) - 1,
            )
            if "nairu" not in constant
            else constant["nairu"]
        )

    return nairu
