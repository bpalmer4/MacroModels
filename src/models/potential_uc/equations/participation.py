"""Participation rate observation equation.

The participation rate is *not* fed in as data. It is cyclical — the
discouraged and encouraged worker effects are exactly a business-cycle
phenomenon — so treating it as an input would inject cycle straight into the
trend and defeat the purpose. It gets its own trend/cycle split instead:

    log pr_t = pr*_t + lambda_pr · gap_t + e_pr

This does two jobs. It identifies pr*, which trend hours needs (see
`trend_hours.py`). And it gives the model a second labour-market observation
loading on the same output gap, which is the sharper contribution: Fleischman &
Roberts (2011) find labour-market variables are the most informative individual
indicators of the cycle, with inflation informative conditional on them. The
first specification of this model had only hours and inflation.

lambda_pr is expected positive — participation rises when the economy runs hot
(encouraged workers) and falls when it does not — and smaller than the hours
loading, since the participation margin adjusts more sluggishly than hours.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.potential_uc.base import set_model_coefficients


def participation_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Participation rate as trend plus a loading on the output gap.

    Model: log_pr_t = pr*_t + lambda_pr · gap_t + e_pr

    lambda_pr is bounded below at zero: a negative loading would mean
    participation falls when the economy runs above potential, which would be
    the discouraged-worker effect with its sign reversed.
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "lambda_pr": {"mu": 0.20, "sigma": 0.15, "lower": 0.0, "upper": 1.0},
            "sigma_pr_obs": {"sigma": 0.40},
        }
        mc = set_model_coefficients(model, settings, constant)

        predicted = (
            latents["trend_participation"]
            + mc["lambda_pr"] * latents["output_gap"]
        )

        pm.Normal(
            "observed_participation",
            mu=predicted,
            sigma=mc["sigma_pr_obs"],
            observed=obs["log_pr"],
        )

    return "log_pr_t = pr*_t + lambda_pr · gap_t + e_pr"
