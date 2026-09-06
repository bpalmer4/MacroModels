"""Trend hours state equation, decomposed into its demographic parts.

Adds latents `trend_participation` (pr*), `trend_hours_per_participant` (hpp*)
and `trend_hours` (h*), all in log x 100.

The decomposition is an exact identity:

    log hours = log pop + log participation + log (hours per participant)

with population *observed* rather than latent. That is the whole point. Trend
hours growth is dominated by demography, demography is measured, and it is not
cyclical — so feeding it in as data removes a large chunk of low-frequency
variation from the set of things a weakly-identified random walk would
otherwise have to discover.

`hpp*` is hours per member of the *labour force*, not per employed person. That
choice absorbs the unemployment margin into hpp*, which keeps the identity
closed in three terms without introducing a NAIRU. The cost is that hpp* is not
"average hours worked" in the usual sense: it is average hours per participant,
and trend unemployment moves it.

Both trends carry a constant drift as well as a level innovation, because both
have a genuine secular slope over the sample: participation climbed 6.74 log
points 1993Q1-2026Q2 (0.051/quarter) and hours per participant fell 1.42 log
points (-0.011/quarter). A driftless random walk with innovations small enough
to still count as a trend cannot climb that far.

This replaces an earlier specification in which h* had a single constant drift.
That version could not see the border closure or the migration rebound at all:
it held trend hours growth at 1.58%/yr while measured working-age population
growth ran from 0.20% (2021Q2) to 2.95% (2023Q3).
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.ystar.base import set_model_coefficients


def trend_hours_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Trend hours as observed population plus two latent trends.

    Model: pr*_t  = pr*_{t-1}  + g_pr  + e_pr    e_pr  ~ N(0, sigma_pr*)
           hpp*_t = hpp*_{t-1} + g_hpp + e_hpp   e_hpp ~ N(0, sigma_hpp*)
           h*_t   = pop_t + pr*_t + hpp*_t

    Units are log x 100 throughout, so the drifts are quarterly per cent.
    """
    if constant is None:
        constant = {}

    n_periods = len(obs["log_pop"])

    with model:
        settings = {
            "g_pr": {"mu": 0.05, "sigma": 0.03},
            "g_hpp": {"mu": -0.01, "sigma": 0.03},
            "initial_trend_participation": {
                "mu": float(obs["log_pr"][0]),
                "sigma": 1.0,
            },
            "initial_trend_hpp": {
                "mu": float(obs["log_hours"][0] - obs["log_pop"][0] - obs["log_pr"][0]),
                "sigma": 1.0,
            },
        }
        mc = set_model_coefficients(model, settings, constant)

        # --- Trend participation ---
        pr_innovations = pm.Normal(
            "trend_participation_innovations",
            mu=0,
            sigma=latents["sigma_pr_star"],
            shape=n_periods - 1,
        )
        pr_init = mc["initial_trend_participation"]
        trend_participation = pm.Deterministic(
            "trend_participation",
            pt.concatenate([
                [pr_init],
                pr_init + pt.cumsum(mc["g_pr"] + pr_innovations),
            ]),
        )

        # --- Trend hours per labour-force participant ---
        hpp_innovations = pm.Normal(
            "trend_hpp_innovations",
            mu=0,
            sigma=latents["sigma_hpp_star"],
            shape=n_periods - 1,
        )
        hpp_init = mc["initial_trend_hpp"]
        trend_hpp = pm.Deterministic(
            "trend_hours_per_participant",
            pt.concatenate([
                [hpp_init],
                hpp_init + pt.cumsum(mc["g_hpp"] + hpp_innovations),
            ]),
        )

        # --- Trend hours, by the identity ---
        trend_hours = pm.Deterministic(
            "trend_hours",
            obs["log_pop"] + trend_participation + trend_hpp,
        )

    latents["trend_participation"] = trend_participation
    latents["trend_hours_per_participant"] = trend_hpp
    latents["trend_hours"] = trend_hours
    return "h*_t = pop_t (observed) + pr*_t + hpp*_t"
