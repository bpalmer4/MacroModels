"""Participation rate equation (discouraged worker effect)."""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.nairu.base import set_model_coefficients


def participation_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Participation rate equation.

    Model: dpr = beta_pr x (U_{-1} - NAIRU_{-1}) + e
    """
    if constant is None:
        constant = {}

    nairu = latents["nairu"]

    with model:
        settings = {
            "beta_pr": {"mu": -0.05, "sigma": 0.03, "upper": 0},
            "epsilon_pr": {"sigma": 0.3},
        }
        mc = set_model_coefficients(model, settings, constant)

        U_lag1 = obs["U_1"]
        nairu_lag1 = pt.concatenate([[nairu[0]], nairu[:-1]])
        u_gap_lag1 = U_lag1 - nairu_lag1

        predicted_delta_pr = mc["beta_pr"] * u_gap_lag1

        pm.Normal(
            "observed_participation",
            mu=predicted_delta_pr,
            sigma=mc["epsilon_pr"],
            observed=obs["Δpr"],
        )

    return "dpr = beta_pr x (U_{-1} - NAIRU_{-1}) + e"
