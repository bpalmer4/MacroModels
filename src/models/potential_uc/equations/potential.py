"""Potential output state equations for the core (Y, pi) specification.

Adds latents `trend_growth` (g, quarterly %) and `potential_output` (y*,
log x 100).

This is the model the project set out to build: potential output as a Gaussian
random walk, identified against output and inflation alone. It is Kuttner
(1994) with an anchored Phillips curve in place of the accelerationist one.

    g_t  = g_{t-1} + e_g                 e_g ~ N(0, sigma_g)
    y*_t = y*_{t-1} + g_{t-1} + e_y      e_y ~ N(0, sigma_ystar)

The drift is itself a random walk, so y* is an I(2) trend: the level can bend
rather than merely wander. That is what lets trend growth fall over the sample
instead of being pinned to a constant. The drift enters lagged (g_{t-1}) to
avoid simultaneity between the level and growth innovations.

Two states and two observation equations. Everything the model knows about
potential comes from two questions: does the gap behave like a cycle
(`output.py`), and does it move inflation (`phillips.py`).

The `labour` specification decomposes y* into trend hours and trend
productivity instead. It was built and then set aside: the decomposition
proved close to circular. Observed population cancels algebraically out of the
hours equation, and the resulting trend hours path reproduces a plain HP(1600)
trend of hours with a correlation of 0.9972 — an elaborate apparatus returning
what a filter gives for free. The gap itself was never the problem: it
correlates only 0.66 with an HP(1600) GDP cycle, so inflation is doing real
identifying work. This equation keeps that part and drops the rest.
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.potential_uc.base import set_model_coefficients


def potential_output_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Potential output as a Gaussian random walk with a random-walk drift.

    Units are log x 100, so g is quarterly trend growth in per cent.
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "initial_trend_growth": {"mu": 0.90, "sigma": 0.40},
            "initial_potential": {
                "mu": float(obs["log_gdp"][0]),
                "sigma": 2.0,
            },
        }
        mc = set_model_coefficients(model, settings, constant)

        n_periods = len(obs["log_gdp"])

        # --- Trend growth: Gaussian random walk ---
        growth_innovations = pm.Normal(
            "trend_growth_innovations",
            mu=0,
            sigma=latents["sigma_g"],
            shape=n_periods - 1,
        )
        g_init = mc["initial_trend_growth"]
        trend_growth = pm.Deterministic(
            "trend_growth",
            pt.concatenate([[g_init], g_init + pt.cumsum(growth_innovations)]),
        )

        # --- Potential output: random walk with that drift ---
        # sigma_ystar = 0 is a meaningful setting, not a degenerate one: it
        # removes the level innovation entirely and leaves an integrated random
        # walk, which is exactly the HP(1600) state space. With it, the trend
        # has a smoothing channel HP does not have, so `ratio_ystar = 0` is the
        # test of how much of the answer comes from that extra channel. A
        # Normal with sigma=0 is not a valid PyMC distribution, so the term is
        # dropped rather than zeroed.
        y_init = mc["initial_potential"]
        drift = trend_growth[:-1]
        if latents["sigma_ystar"] > 0:
            level_innovations = pm.Normal(
                "potential_innovations",
                mu=0,
                sigma=latents["sigma_ystar"],
                shape=n_periods - 1,
            )
            drift = drift + level_innovations
        cumulative = pt.cumsum(drift)
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[y_init], y_init + cumulative]),
        )

    latents["trend_growth"] = trend_growth
    latents["potential_output"] = potential_output
    level_term = " + e_y" if latents["sigma_ystar"] > 0 else " (no level innovation)"
    return f"g_t = g_{{t-1}} + e_g;  y*_t = y*_{{t-1}} + g_{{t-1}}{level_term}"
