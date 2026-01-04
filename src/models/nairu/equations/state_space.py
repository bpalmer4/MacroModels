"""NAIRU state equation (Student-t random walk for robustness to occasional large shifts)."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients

# Student-t degrees of freedom for NAIRU innovations
# nu=4 provides fat tails with finite variance (many small moves, occasional large moves)
NAIRU_NU = 4


def nairu_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    constant: dict[str, Any] | None = None,
) -> pm.Distribution:
    """Student-t random walk for the NAIRU (Non-Accelerating Inflation Rate of Unemployment).

    The NAIRU is modeled as a random walk with Student-t innovations:
        NAIRU_t = NAIRU_{t-1} + ε_t,  ε_t ~ StudentT(ν=4, 0, σ)

    Using Student-t instead of Gaussian provides:
    - Tighter central mass: most quarter-to-quarter changes are small
    - Fat tails: occasional large shifts (e.g., GFC, COVID) are accommodated
    - ν=4 gives fat tails with finite variance (robust to occasional large shifts)

    This allows the natural rate of unemployment to evolve slowly over time,
    with robustness to occasional larger shifts during structural changes.

    Args:
        inputs: Must contain "U" (unemployment rate array) for dimensioning
        model: PyMC model context
        constant: Optional fixed values. Keys:
            - "nairu_innovation": Fix innovation scale (recommended ~0.10)
            - "nairu": Fix entire NAIRU series (for counterfactuals)

    Returns:
        pm.RandomWalk: The NAIRU latent variable

    Example:
        nairu = nairu_equation(inputs, model, constant={"nairu_innovation": 0.10})

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "nairu_innovation": {"mu": 0.15, "sigma": 0.05},
        }
        mc = set_model_coefficients(model, settings, constant)

        nairu = (
            pm.RandomWalk(
                "nairu",
                innovation_dist=pm.StudentT.dist(
                    nu=NAIRU_NU,
                    mu=0,
                    sigma=mc["nairu_innovation"],
                ),
                init_dist=pm.Normal.dist(mu=6.0, sigma=2.0),  # Australian NAIRU ~5-7%
                steps=len(inputs["U"]) - 1,
            )
            if "nairu" not in constant
            else constant["nairu"]
        )

    return nairu
