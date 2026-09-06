"""Trend productivity state equations.

Adds latents `trend_prod_growth` (g_lp, quarterly %) and `trend_productivity`
(lp*, log x 100).

This is where the model's flexibility is spent. Trend productivity growth is a
random walk, so the level of trend productivity is a random walk with a
time-varying drift — an I(2) trend. That is the whole point of the model: the
productivity growth slowdown is the object of interest, and giving g_lp a
random walk is what lets the estimate say whether trend growth fell rather
than assuming it did or did not.

Because it is a second-difference channel, sigma_g_lp is far smaller than the
level innovations (see `config.py` on the HP-1600 correspondence).
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.ystar.base import set_model_coefficients


def trend_productivity_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Trend productivity with a random-walk drift.

    Model: g_lp_t = g_lp_{t-1} + e_g,       e_g  ~ N(0, sigma_g_lp)
           lp*_t  = lp*_{t-1} + g_lp_{t-1} + e_lp, e_lp ~ N(0, sigma_lp*)

    The drift uses g_lp_{t-1} rather than g_lp_t to avoid simultaneity between
    the level and growth innovations.
    """
    if constant is None:
        constant = {}

    initial_lp = float(obs["log_gdp"][0] - obs["log_hours"][0])

    with model:
        settings = {
            "initial_prod_growth": {"mu": 0.25, "sigma": 0.25},
            "initial_trend_productivity": {"mu": initial_lp, "sigma": 2.0},
        }
        mc = set_model_coefficients(model, settings, constant)

        n_periods = len(obs["log_gdp"])

        # --- Trend productivity growth: Gaussian random walk ---
        growth_innovations = pm.Normal(
            "prod_growth_innovations",
            mu=0,
            sigma=latents["sigma_g_lp"],
            shape=n_periods - 1,
        )
        g_init = mc["initial_prod_growth"]
        trend_prod_growth = pm.Deterministic(
            "trend_prod_growth",
            pt.concatenate([[g_init], g_init + pt.cumsum(growth_innovations)]),
        )

        # --- Trend productivity level: random walk with that drift ---
        level_innovations = pm.Normal(
            "trend_prod_innovations",
            mu=0,
            sigma=latents["sigma_lp_star"],
            shape=n_periods - 1,
        )
        lp_init = mc["initial_trend_productivity"]
        cumulative = pt.cumsum(trend_prod_growth[:-1] + level_innovations)
        trend_productivity = pm.Deterministic(
            "trend_productivity",
            pt.concatenate([[lp_init], lp_init + cumulative]),
        )

    latents["trend_prod_growth"] = trend_prod_growth
    latents["trend_productivity"] = trend_productivity
    return (
        "g_lp_t = g_lp_{t-1} + e_g;  "
        "lp*_t = lp*_{t-1} + g_lp_{t-1} + e_lp"
    )
