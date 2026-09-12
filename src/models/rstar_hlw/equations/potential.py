"""Potential output state equation for the HLW r-star model.

Adds latent `potential_output` (y*_t, log x 100) to the latents dict, with
drift driven by the latent annualised trend growth g_t.
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.nairu.base import set_model_coefficients


def potential_output_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    *,
    constant: dict[str, Any] | None = None,
    sigma_ystar_prior: float = 0.12,
    sigma_ystar_fixed: float | None = 0.078,
) -> str:
    """Potential output with time-varying trend drift.

    Model: y*_t = y*_{t-1} + g_{t-1}/4 + e_ystar,  e_ystar ~ N(0, sigma_ystar)

    log_gdp is in log x 100 units; g is annualised %, so quarterly log
    growth in those units is g/4. Drift uses g_{t-1} (HLW lag form) to
    avoid simultaneity with the y* innovation.

    `sigma_ystar_prior` IS THE HalfNormal SCALE, AND IT WAS THE BUG.

    It was 0.55, effectively unconstrained, and the posterior piled up at
    1.112, three times that prior's median of 0.371. The consequence is not
    subtle: potential output then moved with a quarterly sd of 1.073 against
    log GDP's own 0.947, so POTENTIAL WAS MORE VOLATILE THAN OUTPUT, and the
    implied gap had sd 1.92 with a range of -3.0 to +7.6. That is not a
    trend-and-cycle decomposition, and every coefficient measured against it
    (a_r, sigma_IS) was measured against a gap that is not a gap.

    This is the classic unconstrained-trend pile-up, and the original HLW has a
    device to prevent it that this implementation dropped: Holston-Laubach-
    Williams fix the signal-to-noise ratios lambda_g and lambda_z by
    Stock-Watson median-unbiased estimation rather than estimating the
    variances freely. See the note at the top of `z_star.py` ("no lambda_z").

    The default 0.12 puts the prior median at 0.081, which is what this repo's
    own potential-output model uses: `ystar` fixes sigma_ystar at
    ratio_ystar x sigma_c = 0.13 x 0.60 = 0.078. So the number is not invented
    here, it is the view `ystar` already takes of how fast potential can move.
    Pass 0.55 to reproduce the old behaviour.

    AND TIGHTENING THE PRIOR WAS NOT ENOUGH. Under HalfNormal(0.12) the
    posterior still landed at 0.862, about seven standard deviations into that
    prior's tail: the likelihood would rather let potential chase GDP than
    explain it, and a prior cannot win an argument the likelihood insists on.
    Potential still moved with a quarterly sd of 0.923 against GDP's 0.947.

    `sigma_ystar_fixed` is the actual HLW device, and it is now the default.
    Holston-Laubach-Williams do not estimate this variance at all: they fix the
    signal-to-noise ratio lambda_g by Stock-Watson median-unbiased estimation
    and impose it. Fixing sigma_ystar at 0.078 is the same move with the ratio
    taken from `ystar` rather than re-estimated, and it is an IMPOSED setting,
    recorded as such in the run log and on the charts. Set it to None to go
    back to estimating sigma_ystar under `sigma_ystar_prior`, which is what
    the eight resolutions in MODEL_NOTES.md were run on.
    """
    if constant is None:
        constant = {}

    # The imposed value is passed through the same `constant` channel a caller
    # would use, so it lands in model._fixed_constants and shows up wherever
    # imposed settings are reported. An explicit caller-supplied constant wins.
    if sigma_ystar_fixed is not None and "sigma_ystar" not in constant:
        constant = {**constant, "sigma_ystar": sigma_ystar_fixed}

    with model:
        settings = {
            "sigma_ystar": {"sigma": sigma_ystar_prior},
            "initial_potential": {
                "mu": float(obs["log_gdp"][0]),
                "sigma": 2.0,
            },
        }
        mc = set_model_coefficients(model, settings, constant)

        g = latents["trend_growth"]
        n_periods = len(obs["log_gdp"])

        innovations = pm.Normal(
            "potential_innovations",
            mu=0,
            sigma=mc["sigma_ystar"],
            shape=n_periods - 1,
        )

        # Quarterly drift from lagged trend growth (g_{t-1}/4 in log x 100 units)
        drift = g[:-1] / 4

        init_value = mc["initial_potential"]
        cumulative_growth = pt.cumsum(drift + innovations)
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[init_value], init_value + cumulative_growth]),
        )

    latents["potential_output"] = potential_output
    latents["sigma_ystar"] = mc["sigma_ystar"]
    sigma_desc = (
        f"sigma_ystar = {constant['sigma_ystar']:.3f} imposed"
        if "sigma_ystar" in constant
        else f"sigma_ystar ~ HalfNormal({sigma_ystar_prior})"
    )
    return f"y*_t = y*_{{t-1}} + g_{{t-1}}/4 + e_ystar,  e ~ N(0, sigma_ystar);  {sigma_desc}"
