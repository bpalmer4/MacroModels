"""Okun's Law equations linking output gap to unemployment."""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.nairu.base import set_model_coefficients


def okun_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Okun's Law (simple change form).

    Model: dU = beta x output_gap + e
    """
    if constant is None:
        constant = {}

    nairu = latents["nairu"]
    potential_output = latents["potential_output"]

    with model:
        settings = {
            "beta_okun": {"mu": -0.2, "sigma": 0.15},
            "epsilon_okun": {"sigma": 0.5},
        }
        mc = set_model_coefficients(model, settings, constant)

        output_gap = obs["log_gdp"] - potential_output

        pm.Normal(
            "okun_law",
            mu=mc["beta_okun"] * output_gap,
            sigma=mc["epsilon_okun"],
            observed=obs["ΔU"],
        )

    return "dU = beta x output_gap + e"


def okun_gap_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Okun's Law (gap-to-gap with persistence).

    Model: U_gap = tau2 x U_gap_{-1} + tau1 x Y_gap + e
    """
    if constant is None:
        constant = {}

    nairu = latents["nairu"]
    potential_output = latents["potential_output"]

    with model:
        settings = {
            "tau1_okun": {"mu": -0.18, "sigma": 0.1, "upper": 0},
            "tau2_okun": {"mu": 0.75, "sigma": 0.15, "lower": 0, "upper": 1},
            "epsilon_okun": {"sigma": 0.3},
        }
        mc = set_model_coefficients(model, settings, constant)

        output_gap = obs["log_gdp"] - potential_output
        nairu_lag1 = pt.concatenate([[nairu[0]], nairu[:-1]])

        U_lag1 = np.concatenate([[obs["U"][0]], obs["U"][:-1]])
        u_gap_lag1 = U_lag1 - nairu_lag1

        predicted_U = nairu + mc["tau2_okun"] * u_gap_lag1 + mc["tau1_okun"] * output_gap

        pm.Normal(
            "okun",
            mu=predicted_U,
            sigma=mc["epsilon_okun"],
            observed=obs["U"],
        )

    return "U_gap = tau2 x U_gap_{-1} + tau1 x Y_gap + e"
