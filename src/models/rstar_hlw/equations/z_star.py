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
from src.models.rstar_hlw.equations.states import walk_or_level

# Prior on z's opening level, annualised %. Centred on zero because z is
# defined as r*'s departure from trend growth, so it has no reason to open
# away from it.
_INIT_Z_MU = 0.0
_INIT_Z_SIGMA = 1.5


def z_star_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    *,
    constant: dict[str, Any] | None = None,
    sigma_z_prior: float = 0.10,
) -> str:
    """Gaussian random walk in the latent r* component beyond trend growth.

    Model:
        z_t  = z_{t-1} + e_z,  e_z ~ N(0, sigma_z)
        r*_t = g_t + z_t

    `sigma_z_prior` is the HalfNormal scale on sigma_z, which governs how fast
    r* is allowed to wander. It is exposed so it can be swept: sigma_z is the
    one quantity here the data may have nothing to say about, and the only way
    to find out is to vary its prior and watch what the r* posterior does. See
    `sigma_z_prior_sweep.py`.
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
            "sigma_z": {"sigma": sigma_z_prior},
        }
        mc = set_model_coefficients(model, settings, constant)

        z_star = (
            walk_or_level(
                model,
                "z_star",
                sigma=mc["sigma_z"],
                init_mu=_INIT_Z_MU,
                init_sigma=_INIT_Z_SIGMA,
                steps=len(obs["log_gdp"]) - 1,
                # Nothing in this model observes z, so its states are pinned
                # only by their own scale. That is the funnel in its sharpest
                # form, and the case non-centring is for.
                non_centred=True,
            )
            if "z_star" not in constant
            else constant["z_star"]
        )

        r_star = pm.Deterministic("r_star", latents["trend_growth"] + z_star)

    latents["z_star"] = z_star
    latents["r_star"] = r_star
    return "z_t = z_{t-1} + e_z,  e_z ~ N(0, sigma_z);  r* = g + z"
