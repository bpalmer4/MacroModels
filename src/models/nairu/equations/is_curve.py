"""IS curve equation linking output gap to interest rates and fiscal impulse."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def is_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """IS curve linking output gap to interest rates and fiscal impulse.

    Model: y_gap = rho x y_gap_{-1} - beta x r_gap_{-2} + gamma x fiscal_{-1} + e
    """
    if constant is None:
        constant = {}

    potential_output = latents["potential_output"]

    with model:
        settings = {
            "rho_is": {"mu": 0.85, "sigma": 0.1},
            "beta_is": {"mu": 0.20, "sigma": 0.10, "lower": 0},
            "gamma_fi": {"mu": 0.05, "sigma": 0.2, "lower": 0},
            "epsilon_is": {"sigma": 0.4},
        }
        mc = set_model_coefficients(model, settings, constant)

        real_rate = obs["cash_rate"] - obs["π_exp"]
        rate_gap = real_rate - obs["det_r_star"]
        rate_gap_lag2 = rate_gap[:-2]

        output_gap = obs["log_gdp"] - potential_output
        output_gap_lag1 = output_gap[1:-1]
        potential_t = potential_output[2:]
        fiscal_impulse_lag1 = obs["fiscal_impulse_1"][2:]

        predicted_log_gdp = (
            potential_t
            + mc["rho_is"] * output_gap_lag1
            - mc["beta_is"] * rate_gap_lag2
            + mc["gamma_fi"] * fiscal_impulse_lag1
        )

        pm.Normal(
            "observed_is",
            mu=predicted_log_gdp,
            sigma=mc["epsilon_is"],
            observed=obs["log_gdp"][2:],
        )

    return "y_gap = rho x y_gap_{-1} - beta x r_gap_{-2} + gamma x fiscal_{-1} + e"
