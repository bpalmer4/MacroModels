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

from src.models.common.spline import basis
from src.models.ystar.base import set_model_coefficients


def potential_spline_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Potential output as a natural cubic spline in time.

        y*_t = sum_j c_j B_j(t)

    Deterministic given the coefficients, so `sigma_ystar` and `sigma_g` both
    disappear rather than being chosen.

    What it buys over the random walk is not smoothness of the innovations,
    which was never the binding problem: the walk's realised innovations sit
    well inside their imposed sd. It is that a random walk penalises the SIZE
    of each step and says nothing about a run of same-signed steps, so a
    sustained pull over a business cycle moves the trend even when every
    individual step is tiny. Three coefficients over forty years cannot dip
    and recover inside two, whatever the likelihood asks for, which is the
    restriction a random walk cannot express.

    The knot count bounds how fast trend growth may turn while leaving its
    amplitude free, so the long decline in potential growth is still
    estimable; a two-year recession is not.
    """
    if constant is None:
        constant = {}
    if "obs_index" not in constant:
        raise ValueError("potential_spline_equation requires 'obs_index' to place its knots")

    knots = tuple(constant.get("spline_knots", ()))
    # Naturality forces zero curvature at both outer knots. On a NAIRU, which
    # is roughly flat at the ends, that is harmless. On log potential output,
    # which must keep rising, it drags the slope toward zero at BOTH ends: the
    # level spline opened at 0.3 per cent growth in 1984 and closed at 0.8,
    # with a narrow band around both. Off, and with no interior knots, the
    # basis is a global cubic in the level, so growth is a quadratic in time
    # that rises, peaks and slows, with no end condition to distort it.
    natural = bool(constant.get("natural", True))
    degree = int(constant.get("degree", 3))
    design = basis(constant["obs_index"], knots, natural=natural, degree=degree)
    mu0, sd0 = constant.get("coef_prior", (float(obs["log_gdp"].mean()), 20.0))

    # Optional slow-moving adjustment: y*_t = poly_t + z_t, with z a driftless
    # random walk starting at zero.
    #
    # The polynomial cannot express a departure that is neither cyclical nor
    # part of the long arc, and a global form has no local freedom at all. The
    # walk supplies it. Its INNOVATION SD IS THE WHOLE CONTENT OF THE IDEA: a
    # random walk can mimic any polynomial, so the two are separated only by
    # how far z is allowed to move per quarter. Loose, and z absorbs the cycle
    # and the trend chases output again, which is the defect the polynomial was
    # brought in to fix. Imposed, never estimated, for the Stock-Watson reason
    # that applies to every variance in this model.
    #
    # z starts at zero because the polynomial already carries a constant, and
    # a free level in both is not identified.
    sigma_z = float(constant.get("sigma_z", 0.0))
    n_periods = len(obs["log_gdp"])

    with model:
        coef = pm.Normal("ystar_coef", mu=mu0, sigma=sd0, shape=design.shape[1])
        smooth = pt.dot(pt.as_tensor_variable(design), coef)
        if sigma_z > 0:
            # Non-centred: the data are only weakly informative about any one
            # quarter's innovation, which is the case this parameterisation is
            # for.
            raw = pm.Normal("z_ystar", mu=0.0, sigma=1.0, shape=n_periods - 1)
            adjustment = pm.Deterministic(
                "ystar_adjustment",
                pt.concatenate([[0.0], pt.cumsum(raw * sigma_z)]),
            )
            smooth = smooth + adjustment
        potential_output = pm.Deterministic("potential_output", smooth)
        # Reported so the growth charts and the summary read the same name
        # they do under the walk. Quarterly, in log x 100, as `trend_growth`
        # is there.
        trend_growth = pm.Deterministic(
            "trend_growth",
            pt.concatenate([[potential_output[1] - potential_output[0]],
                            potential_output[1:] - potential_output[:-1]]),
        )

    latents["trend_growth"] = trend_growth
    latents["potential_output"] = potential_output
    where = ", ".join(knots) if knots else "none"
    ends = "natural ends" if natural else "free ends"
    # Growth is one degree below the level, which is the number the shape of
    # the growth chart is set by: a global cubic in y* is a parabola in g*.
    growth = f"degree {degree - 1}" + ("" if not knots else ", piecewise")
    walk = f" + z_t  (GRW, sigma_z = {sigma_z:g} imposed)" if sigma_z > 0 else ""
    return (
        f"y*_t = sum_j c_j B_j(t){walk}   (degree {degree}, {ends}, "
        f"{design.shape[1]} coefficients, knots: {where}; g* is {growth} in time)"
    )


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
