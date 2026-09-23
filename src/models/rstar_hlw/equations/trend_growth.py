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
import pytensor.tensor as pt

from src.models.nairu.base import set_model_coefficients
from src.models.rstar_hlw.equations.states import walk_or_level

# Prior on g's opening level, annualised %. Australia's trend growth at the
# start of the inflation-targeting era; the sd is wide enough that the data
# move it, and the posterior does.
_INIT_G_MU = 3.5
_INIT_G_SIGMA = 1.5


def trend_growth_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    *,
    constant: dict[str, Any] | None = None,
    sigma_g_value: float | pt.TensorVariable | None = None,
) -> str:
    """Random walk in trend growth.

    Model: g_t = g_{t-1} + e_g,  e_g ~ N(0, sigma_g)

    `sigma_g_value` replaces sigma_g's own prior, and is how the lambda_g
    ratio reaches this equation: the caller works out
    lambda_g x sigma_ystar (see `estimate.py:_sigma_g_from_lambda`, which also
    handles the annualisation) and passes the result here. A float arrives
    when sigma_ystar was imposed, in which case both variances are pinned. A
    tensor arrives when sigma_ystar is free, which is HLW's device: only the
    ratio is imposed, and sigma_g is recorded as a derived quantity rather
    than sampled as a parameter.

    WHY IT MATTERS. With sigma_ystar imposed but sigma_g free, the posterior
    on sigma_g comes back around 0.12 against this prior's scale of 0.04, and
    g carries a pandemic-shaped trough: the volatility that used to pile into
    potential piles into trend growth instead. HLW prevent that by fixing the
    RATIO of the two, not either one alone. Leave it None to sample sigma_g
    under the prior below.
    """
    if constant is None:
        constant = {}

    # A float goes through the `constant` channel so it lands in
    # model._fixed_constants and shows up wherever imposed settings are
    # reported. A tensor cannot: it is an expression in another parameter, so
    # it is registered below as a Deterministic instead, which is also how it
    # reaches the trace.
    if isinstance(sigma_g_value, float) and "sigma_g" not in constant:
        constant = {**constant, "sigma_g": sigma_g_value}

    # Fixed (very-soft) measurement sigma on the linear-trend anchor.
    # Kept fixed rather than estimated because the previous run with a free
    # HalfNormal(1.5) prior collapsed the posterior to 0.022, hardening the
    # "soft" anchor into a constraint.
    SIGMA_TREND_OBS = 2.0

    derived = sigma_g_value is not None and not isinstance(sigma_g_value, float)

    with model:
        # No prior on sigma_g when it is derived from sigma_ystar: asking for
        # one would build a free parameter the likelihood never sees, which
        # then samples its own prior and reports a number that is not the
        # sigma_g the model used.
        settings = {} if derived else {"sigma_g": {"sigma": 0.04}}
        mc = set_model_coefficients(model, settings, constant)
        if derived:
            mc["sigma_g"] = pm.Deterministic("sigma_g", sigma_g_value)

        trend_growth = (
            walk_or_level(
                model,
                "trend_growth",
                sigma=mc["sigma_g"],
                init_mu=_INIT_G_MU,
                init_sigma=_INIT_G_SIGMA,
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
    if "sigma_g" in constant:
        sigma_desc = f";  sigma_g = {constant['sigma_g']:.4f} imposed"
    elif sigma_g_value is not None:
        sigma_desc = ";  sigma_g = lambda_g x sigma_ystar, ratio imposed, level free"
    else:
        sigma_desc = ""
    desc = f"g_t = g_{{t-1}} + e_g,  e_g ~ N(0, sigma_g){sigma_desc}"
    if soft_anchor_active:
        desc += f";  linear_trend ~ N(g, {SIGMA_TREND_OBS:.1f})"
    return desc
