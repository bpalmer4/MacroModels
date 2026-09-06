"""Trend growth state: a Gaussian random walk drift.

Just the `g` path. `potential.py` builds `g` *and* `y*` as states, which is what
a conventional unobserved-components model needs. The inflation specification
does not: there `y*` is a deterministic residual, `log_gdp - c·(pi - anchor)`,
so only the drift is a free path. Splitting the drift out lets that
specification impose the random walk on the *implied* potential rather than
generate potential from it.
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.ystar.base import set_model_coefficients


def trend_growth_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Trend growth as a Gaussian random walk. Adds latent `trend_growth`.

    Units are log x 100 per quarter, so `g` is quarterly trend growth in per
    cent. `latents` must already carry `sigma_g` (see `scale.py`).
    """
    if constant is None:
        constant = {}

    with model:
        settings = {"initial_trend_growth": {"mu": 0.90, "sigma": 0.40}}
        mc = set_model_coefficients(model, settings, constant)

        n_periods = len(obs["log_gdp"])
        innovations = pm.Normal(
            "trend_growth_innovations",
            mu=0,
            sigma=latents["sigma_g"],
            shape=n_periods - 1,
        )
        g_init = mc["initial_trend_growth"]
        trend_growth = pm.Deterministic(
            "trend_growth",
            pt.concatenate([[g_init], g_init + pt.cumsum(innovations)]),
        )

    latents["trend_growth"] = trend_growth
    return "g_t = g_{t-1} + e_g,  e_g ~ N(0, sigma_g)"
