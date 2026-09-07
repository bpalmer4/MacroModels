"""Potential output state equations for the core (Y, pi) specification.

Adds latents `trend_growth` (g, quarterly %) and `potential_output` (y*,
log x 100).

This is the model the project set out to build: potential output as a Gaussian
random walk, identified against output and inflation alone. It is Kuttner
(1994) with an anchored Phillips curve in place of the accelerationist one.

    g_t  = g_{t-1} + e_g                 e_g ~ N(0, sigma_g)
    y*_t = y*_{t-1} + g_{t-1} + e_y      e_y ~ N(0, sigma_ystar)

`ModelConfig.level_break` optionally adds a free one-off step `delta` to the
level recursion at a nominated quarter. See that field for what a step can and
cannot do here, which is less than it first appears: the gap is defined off
inflation, so a step moves variation between y* and the GDP residual and
touches the gap only through `c`.

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

from src.models.ystar.base import set_model_coefficients


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

        # --- Optional one-off level break ---
        # `drift[i]` is the increment carrying y* from period i to period i+1,
        # so a break *at* period k is added to drift[k-1]. Because the drift is
        # cumulated, the step is permanent: every quarter from k onward shifts
        # by delta.
        #
        # The prior is wide on purpose. sigma_ystar is 0.078 at the default
        # settings, so a Normal(0, 5) admits a step some sixty times a single
        # quarterly innovation and lets GDP decide. Nothing here asserts the
        # sign.
        break_index = constant.get("break_index")
        if break_index is not None:
            if not isinstance(break_index, tuple):
                raise TypeError(
                    f"break_index must be a tuple of ints, got {type(break_index).__name__}",
                )
            for position in break_index:
                if not isinstance(position, int):
                    raise TypeError(f"break_index entries must be ints, got {position!r}")
                if not 1 <= position < n_periods:
                    raise ValueError(
                        f"break index {position} must lie in 1..{n_periods - 1}; a break at "
                        f"the first observation is absorbed by initial_potential and is not "
                        f"identified",
                    )
            labels = constant.get("break_labels")
            if not isinstance(labels, tuple) or len(labels) != len(break_index):
                raise ValueError("break_labels must be a tuple naming each break quarter")

            # One free step per break, labelled by quarter so the trace reads
            # `level_break[2020Q2]` rather than `level_break[0]`.
            model.add_coord("level_break_quarter", labels)
            step = pm.Normal("level_break", mu=0.0, sigma=5.0, dims="level_break_quarter")
            indicator = np.zeros((n_periods - 1, len(break_index)))
            for column, position in enumerate(break_index):
                indicator[position - 1, column] = 1.0
            drift = drift + pt.dot(indicator, step)

        cumulative = pt.cumsum(drift)
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[y_init], y_init + cumulative]),
        )

    latents["trend_growth"] = trend_growth
    latents["potential_output"] = potential_output
    level_term = " + e_y" if latents["sigma_ystar"] > 0 else " (no level innovation)"
    breaks = constant.get("break_labels")
    break_term = "" if not breaks else "".join(f" + delta[{q}]·1{{t = {q}}}" for q in breaks)
    return f"g_t = g_{{t-1}} + e_g;  y*_t = y*_{{t-1}} + g_{{t-1}}{level_term}{break_term}"
