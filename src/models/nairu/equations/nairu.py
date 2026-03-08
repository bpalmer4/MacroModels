"""NAIRU state equation (Gaussian or Student-t random walk).

State equation that adds 'nairu' to the latents dict.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients

# Student-t degrees of freedom for fat-tail variant
NAIRU_NU = 4


def nairu_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Gaussian random walk for the NAIRU.

    Model: NAIRU_t = NAIRU_{t-1} + e_t,  e ~ N(0, sigma)
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
                mu=0,
                sigma=mc["nairu_innovation"],
                init_dist=pm.Normal.dist(mu=15.0, sigma=8.0),
                steps=len(obs["U"]) - 1,
            )
            if "nairu" not in constant
            else constant["nairu"]
        )

    latents["nairu"] = nairu
    return "NAIRU_t = NAIRU_{t-1} + e_t,  e ~ N(0, sigma)"


def nairu_student_t_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Student-t random walk for the NAIRU (fat tails).

    Model: NAIRU_t = NAIRU_{t-1} + e_t,  e ~ StudentT(nu=4, 0, sigma)
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
                init_dist=pm.Normal.dist(mu=7.0, sigma=2.0),
                steps=len(obs["U"]) - 1,
            )
            if "nairu" not in constant
            else constant["nairu"]
        )

    latents["nairu"] = nairu
    return "NAIRU_t = NAIRU_{t-1} + e_t,  e ~ StudentT(nu=4, 0, sigma)"
