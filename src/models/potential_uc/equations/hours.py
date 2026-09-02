"""Hours observation equation.

The hours series is what separates trend hours from trend productivity. Without
it, only the sum h* + lp* would be identified and the model could not attribute
the gap.

lambda_h is the share of the output gap that shows up in hours. It does the
work that a second, correlated cycle state would otherwise do: labour hoarding
means firms adjust hours by less than output, so lambda_h < 1 is the expected
finding, and (1 - lambda_h) is the share of the cycle showing up in measured
productivity. It is the model's Okun coefficient in hours space.

eta is measurement error on hours, not part of the cycle. LFS hours is a noisy
survey series (sample rotation, holiday and weather timing), and without a
separate white-noise term that noise is read as labour underutilisation and
propagates into the gap and thence into potential. The separation is
identified off the autocovariance: the cycle is persistent, eta is not.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.potential_uc.base import set_model_coefficients


def hours_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Hours worked as trend plus a loading on the output gap.

    Model: log_hours_t = h*_t + lambda_h · gap_t + eta_t

    lambda_h is bounded to [0, 1.5]: negative would mean hours move against the
    cycle, and above ~1.5 hours would have to be more cyclical than output
    itself.
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "lambda_h": {"mu": 0.70, "sigma": 0.25, "lower": 0.0, "upper": 1.5},
            "sigma_eta": {"sigma": 0.50},
        }
        mc = set_model_coefficients(model, settings, constant)

        predicted = latents["trend_hours"] + mc["lambda_h"] * latents["output_gap"]

        pm.Normal(
            "observed_hours",
            mu=predicted,
            sigma=mc["sigma_eta"],
            observed=obs["log_hours"],
        )

        # The complement: share of the cycle landing in measured productivity.
        pm.Deterministic("lambda_lp", 1.0 - mc["lambda_h"])

    return "log_hours_t = h*_t + lambda_h · gap_t + eta_t,  eta ~ N(0, sigma_eta)"
