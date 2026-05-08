"""z-star state equation for textbook canonical HLW.

Adds latent `z_star` as a Gaussian random walk, plus the deterministic
`r_star = trend_growth + z_star`.

This is the **textbook canonical HLW (2017)** form:
- z is a simple Gaussian random walk
- sigma_z is a free parameter with a HalfNormal prior (no lambda_z
  reparameterisation, no AR(1) damping)

Used as **Resolution A** in the CLI toggle. The AR(1)-reparameterised
Lewis-Vazquez-Grande variant lives in git history (it was the form we
explored mid-project before settling on Resolution C). This canonical
version is the form used in iteration 1 of the sampler progression — the
"very first model" in MODEL_NOTES.md.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def z_star_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Gaussian random walk in the latent r* component beyond trend growth.

    Model:
        z_t  = z_{t-1} + e_z,  e_z ~ N(0, sigma_z)
        r*_t = g_t + z_t
    """
    if constant is None:
        constant = {}

    if "trend_growth" not in latents:
        raise RuntimeError(
            "z_star_equation requires trend_growth in latents — "
            "ensure trend_growth_equation runs first.",
        )

    with model:
        settings = {
            "sigma_z": {"sigma": 0.10},
        }
        mc = set_model_coefficients(model, settings, constant)

        z_star = (
            pm.GaussianRandomWalk(
                "z_star",
                mu=0,
                sigma=mc["sigma_z"],
                init_dist=pm.Normal.dist(mu=0.0, sigma=1.5),
                steps=len(obs["log_gdp"]) - 1,
            )
            if "z_star" not in constant
            else constant["z_star"]
        )

        r_star = pm.Deterministic("r_star", latents["trend_growth"] + z_star)

    latents["z_star"] = z_star
    latents["r_star"] = r_star
    return "z_t = z_{t-1} + e_z,  e_z ~ N(0, sigma_z);  r* = g + z"
