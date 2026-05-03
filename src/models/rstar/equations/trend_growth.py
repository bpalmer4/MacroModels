"""Trend growth state equation for the HLW r-star model.

Adds latent `trend_growth` (g_t, annualised %) to the latents dict, plus a
soft observation equation that gently anchors g to a **linear regression**
of year-on-year GDP growth over the model sample.

Why a linear trend rather than a Henderson MA:
- Linear is immune to cyclical contamination (COVID, GFC etc) — these single
  shocks barely move a regression line
- HMA(13) had the COVID dip baked into the anchor, which then bled into g
- The linear trend captures the secular slowdown narrative cleanly
  (~-0.07 pp/year over 1993Q1-2025Q4)

`sigma_trend_obs` is **fixed** at a large value (2.0) so the anchor stays
genuinely soft. When this sigma was a free parameter, the data collapsed it
to ~0.02, turning the "soft" anchor into a hard constraint that forced g to
follow the smoothed data.

Kept in centred form (`pm.GaussianRandomWalk`) because non-centring this
random walk produced catastrophic divergences when combined with the y*
equation, which already uses `g[:-1]/4` as a drift inside another cumsum.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def trend_growth_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Random walk in trend growth.

    Model: g_t = g_{t-1} + e_g,  e_g ~ N(0, sigma_g)
    """
    if constant is None:
        constant = {}

    # Fixed (very-soft) measurement sigma on the linear-trend anchor.
    # Kept fixed rather than estimated because the previous run with a free
    # HalfNormal(1.5) prior collapsed the posterior to 0.022, hardening the
    # "soft" anchor into a constraint.
    SIGMA_TREND_OBS = 2.0

    with model:
        settings = {
            "sigma_g": {"sigma": 0.04},
        }
        mc = set_model_coefficients(model, settings, constant)

        trend_growth = (
            pm.GaussianRandomWalk(
                "trend_growth",
                mu=0,
                sigma=mc["sigma_g"],
                init_dist=pm.Normal.dist(mu=3.5, sigma=1.5),
                steps=len(obs["log_gdp"]) - 1,
            )
            if "trend_growth" not in constant
            else constant["trend_growth"]
        )

        # Soft observation: linear-regression trend of YoY growth ~ N(g, fixed sigma)
        soft_anchor_active = "trend_growth_obs" in obs
        if soft_anchor_active:
            if not hasattr(model, "_fixed_constants"):
                model._fixed_constants = {}  # noqa: SLF001
            model._fixed_constants["sigma_trend_obs"] = SIGMA_TREND_OBS  # noqa: SLF001
            pm.Normal(
                "observed_trend_growth",
                mu=trend_growth,
                sigma=SIGMA_TREND_OBS,
                observed=obs["trend_growth_obs"],
            )

    latents["trend_growth"] = trend_growth
    desc = "g_t = g_{t-1} + e_g,  e_g ~ N(0, sigma_g)"
    if soft_anchor_active:
        desc += f";  linear_trend ~ N(g, {SIGMA_TREND_OBS:.1f})"
    return desc
