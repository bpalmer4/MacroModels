"""Exchange rate equation (UIP-style TWI)."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def exchange_rate_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Exchange rate equation linking TWI to interest rate differential.

    Model: de = rho x de_{-1} + beta_r x r_gap_{-1} + e
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "rho_er": {"mu": 0.3, "sigma": 0.15},
            "beta_er_r": {"mu": 0.3, "sigma": 0.2, "lower": 0},
            "epsilon_er": {"sigma": 3.0},
        }
        mc = set_model_coefficients(model, settings, constant)

        predicted_delta_twi = (
            mc["rho_er"] * obs["Δtwi_1"]
            + mc["beta_er_r"] * obs["r_gap_1"]
        )

        pm.Normal(
            "observed_twi_change",
            mu=predicted_delta_twi,
            sigma=mc["epsilon_er"],
            observed=obs["Δtwi"],
        )

    return "de = rho x de_{-1} + beta_r x r_gap_{-1} + e"
