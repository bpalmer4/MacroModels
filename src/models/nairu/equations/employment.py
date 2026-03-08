"""Employment equation linking labour demand to output and real wages."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def employment_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Employment equation - labour demand.

    Model: demp = alpha + beta_ygap x output_gap + beta_wage x (dulc - dmfp) + e
    """
    if constant is None:
        constant = {}

    potential_output = latents["potential_output"]

    with model:
        settings = {
            "alpha_emp": {"mu": 0.3, "sigma": 0.2},
            "beta_emp_ygap": {"mu": 0.15, "sigma": 0.1, "lower": 0},
            "beta_emp_wage": {"mu": -0.1, "sigma": 0.1, "upper": 0},
            "epsilon_emp": {"sigma": 0.4},
        }
        mc = set_model_coefficients(model, settings, constant)

        output_gap = obs["log_gdp"] - potential_output

        predicted_emp_growth = (
            mc["alpha_emp"]
            + mc["beta_emp_ygap"] * output_gap
            + mc["beta_emp_wage"] * obs["real_wage_gap"]
        )

        pm.Normal(
            "observed_employment",
            mu=predicted_emp_growth,
            sigma=mc["epsilon_emp"],
            observed=obs["emp_growth"],
        )

    return "demp = alpha + beta_ygap x output_gap + beta_wage x (dulc - dmfp) + e"
