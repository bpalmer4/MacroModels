"""Import price pass-through equation."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def import_price_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],  # noqa: ARG001 — common equation interface
    constant: dict[str, Any] | None = None,
) -> str:
    """Import price pass-through from TWI and oil prices.

    Model: d4pm = beta_pt x d4twi_{-1} + beta_oil x d4oil_{-1} + rho x d4pm_{-1} + e
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "beta_pt": {"mu": -0.3, "sigma": 0.15, "upper": 0},
            "beta_oil": {"mu": 0.10, "sigma": 0.05, "lower": 0},
            "rho_pt": {"mu": 0.4, "sigma": 0.15},
            "epsilon_pt": {"sigma": 2.5},
        }
        mc = set_model_coefficients(model, settings, constant)

        predicted_delta_pm = (
            mc["beta_pt"] * obs["Δ4twi_1"]
            + mc["beta_oil"] * obs["Δ4oil_1"]
            + mc["rho_pt"] * obs["Δ4ρm_1"]
        )

        pm.Normal(
            "observed_import_price",
            mu=predicted_delta_pm,
            sigma=mc["epsilon_pt"],
            observed=obs["Δ4ρm"],
        )

    return "d4pm = beta_pt x d4twi_{-1} + beta_oil x d4oil_{-1} + rho x d4pm_{-1} + e"
