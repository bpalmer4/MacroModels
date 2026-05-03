"""Phillips curve for the HLW r-star model.

Anchor-augmented form on annual (year-on-year) trimmed mean inflation. The
annual form is more identifying than quarterly inflation: it averages out
high-frequency noise so the b_y slope can do real work pinning the output gap.

The b_y prior is held away from zero (lower=0.02) so the Phillips curve does
not collapse — without it, y* could absorb all of output and z would wander.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def phillips_curve_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Anchor-augmented Phillips curve on annual trimmed mean.

    Model: pi_4_t = pi_exp_t + b_y * y_gap_{t-1} + e_pi

    pi_4 and pi_exp are both annualised %; y_gap is in log x 100 units, so b_y
    translates 1 log-point of output gap into pp of annual inflation.
    """
    if constant is None:
        constant = {}

    potential_output = latents["potential_output"]

    with model:
        settings = {
            "b_y": {"mu": 0.10, "sigma": 0.05, "lower": 0.02},
            "sigma_pi": {"sigma": 0.30},
        }
        mc = set_model_coefficients(model, settings, constant)

        output_gap = obs["log_gdp"] - potential_output

        predicted_pi = obs["pi_exp"][1:] + mc["b_y"] * output_gap[:-1]

        pm.Normal(
            "observed_pi",
            mu=predicted_pi,
            sigma=mc["sigma_pi"],
            observed=obs["pi_4"][1:],
        )

    return "pi_4_t = pi_exp_t + b_y * y_gap_{t-1} + e_pi"
