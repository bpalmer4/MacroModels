"""Net exports equation linking trade balance to demand and exchange rates."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def net_exports_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Net exports equation.

    Model: d(NX/Y) = beta_ygap x output_gap + beta_twi x dtwi + e
    """
    if constant is None:
        constant = {}

    potential_output = latents["potential_output"]

    with model:
        settings = {
            "beta_nx_ygap": {"mu": -0.05, "sigma": 0.05, "upper": 0},
            "beta_nx_twi": {"mu": -0.02, "sigma": 0.02, "upper": 0},
            "epsilon_nx": {"sigma": 0.3},
        }
        mc = set_model_coefficients(model, settings, constant)

        output_gap = obs["log_gdp"] - potential_output

        predicted_dnx = mc["beta_nx_ygap"] * output_gap + mc["beta_nx_twi"] * obs["Δtwi"]

        pm.Normal(
            "observed_net_exports",
            mu=predicted_dnx,
            sigma=mc["epsilon_nx"],
            observed=obs["Δnx_ratio"],
        )

    return "d(NX/Y) = beta_ygap x output_gap + beta_twi x dtwi + e"
