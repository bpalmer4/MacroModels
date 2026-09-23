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
from src.models.rstar_hlw.equations.exclusion import drop_excluded

# The two priors on the Phillips slope. HLW2017 impose only that b_y is
# positive; this repo's default also centres it at 0.10. The lower bound of
# 0.02 is kept in both, because it is what stops the curve collapsing and
# letting y* absorb all of output, and HLW bound the slope away from zero
# too. See `_A_R_PRIOR` in `is_curve.py` for the sd of the sign-only form.
#
# Unlike a_r, this constraint is not binding: the posterior sits above 0.27,
# more than ten times the floor, so the two priors should give the same
# answer. That is the point of running it.
_B_Y_PRIOR = {"mu": 0.10, "sigma": 0.05, "lower": 0.02}
_B_Y_PRIOR_SIGN_ONLY = {"mu": 0.0, "sigma": 0.5, "lower": 0.02}


def phillips_curve_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    *,
    constant: dict[str, Any] | None = None,
    keep: np.ndarray | None = None,
    sign_prior_only: bool = False,
) -> str:
    """Anchor-augmented Phillips curve on annual trimmed mean.

    Model: pi_4_t = pi_exp_t + b_y * y_gap_{t-1} + e_pi

    pi_4 and pi_exp are both annualised %; y_gap is in log x 100 units, so b_y
    translates 1 log-point of output gap into pp of annual inflation.

    `keep` drops a window of quarters (see `exclusion.py`). It is applied here
    as well as in the IS curve, and on purpose: the lockdown output gap is a
    shuttered economy rather than deficient demand, so asking b_y to price it
    into inflation would pull potential down towards GDP through a second
    route, which is the thing the exclusion exists to stop. The gap is lagged
    one quarter here, so the mask is applied on the inflation date, meaning the
    quarter whose GAP is excluded is the one before each dropped row.
    """
    if constant is None:
        constant = {}

    potential_output = latents["potential_output"]

    with model:
        settings = {
            "b_y": dict(_B_Y_PRIOR_SIGN_ONLY if sign_prior_only else _B_Y_PRIOR),
            "sigma_pi": {"sigma": 0.30},
        }
        mc = set_model_coefficients(model, settings, constant)

        output_gap = obs["log_gdp"] - potential_output

        predicted_pi = obs["pi_exp"][1:] + mc["b_y"] * output_gap[:-1]

        # first=0: row i uses the gap dated i, so the mask is read on the gap's
        # own date rather than the inflation date one quarter later.
        fitted, observed = drop_excluded(
            keep, 0, predicted_pi, np.asarray(obs["pi_4"][1:], dtype=float),
        )

        pm.Normal(
            "observed_pi",
            mu=fitted,
            sigma=mc["sigma_pi"],
            observed=observed,
        )

    return "pi_4_t = pi_exp_t + b_y * y_gap_{t-1} + e_pi"
