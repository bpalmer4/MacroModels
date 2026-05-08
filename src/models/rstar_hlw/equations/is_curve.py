"""HLW IS curve with latent r* and a fiscal-impulse regressor.

Observation equation indexed from t=2 onwards (needs two output-gap and two
real-rate-gap lags).

The fiscal impulse helps the IS curve do real work — without it, sigma_IS
absorbs both demand shocks and fiscal effects, leaving little explanatory
power for the real rate gap. With it, a_r can plausibly identify away from
zero.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def is_curve_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """HLW (2017) IS curve in level form, with fiscal impulse.

    Model:
        log_gdp_t = y*_t
                  + a_y1 * y_gap_{t-1}
                  + a_y2 * y_gap_{t-2}
                  + (a_r/2) * (r_gap_{t-1} + r_gap_{t-2})
                  + gamma_fi * fiscal_impulse_{t-1}
                  + e_IS

    where r_gap = (cash_rate - pi_exp) - r*, all annualised %.
    Output gap is in log x 100 units.
    """
    if constant is None:
        constant = {}

    r_star = latents["r_star"]
    potential_output = latents["potential_output"]

    with model:
        settings = {
            "a_y1": {"mu": 0.90, "sigma": 0.10, "lower": 0.0, "upper": 1.0},
            "a_y2": {"mu": -0.10, "sigma": 0.10, "upper": 0.0},
            "a_r": {"mu": -0.15, "sigma": 0.08, "upper": 0.0},
            "sigma_IS": {"sigma": 0.4},
        }
        if "fiscal_impulse_1" in obs:
            settings["gamma_fi"] = {"mu": 0.05, "sigma": 0.20, "lower": 0.0}
        mc = set_model_coefficients(model, settings, constant)

        real_rate = obs["cash_rate"] - obs["pi_exp"]
        r_gap = real_rate - r_star

        output_gap = obs["log_gdp"] - potential_output

        # t = 2 .. T-1 (we predict log_gdp from index 2 onwards)
        predicted_log_gdp = (
            potential_output[2:]
            + mc["a_y1"] * output_gap[1:-1]
            + mc["a_y2"] * output_gap[:-2]
            + (mc["a_r"] / 2) * (r_gap[1:-1] + r_gap[:-2])
        )

        if "fiscal_impulse_1" in obs:
            predicted_log_gdp = predicted_log_gdp + mc["gamma_fi"] * obs["fiscal_impulse_1"][2:]

        pm.Normal(
            "observed_IS",
            mu=predicted_log_gdp,
            sigma=mc["sigma_IS"],
            observed=obs["log_gdp"][2:],
        )

    parts = [
        "y_gap_t = a_y1*y_gap_{t-1} + a_y2*y_gap_{t-2} + (a_r/2)*(r_gap_{t-1}+r_gap_{t-2})",
    ]
    if "fiscal_impulse_1" in obs:
        parts.append("gamma_fi*fiscal_{t-1}")
    parts.append("e_IS")
    return " + ".join(parts)
