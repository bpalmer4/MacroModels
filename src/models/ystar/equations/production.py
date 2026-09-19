"""Potential growth from a Cobb-Douglas production function.

An alternative to `potential.py`. There, potential growth is a free drift
state smoothed by an imposed `sigma_g`. Here it is built from factor trends:

    g_K*_t = g_K*_{t-1} + e_K       trend capital growth
    g_L*_t = g_L*_{t-1} + e_L       trend hours growth
    g_M*_t = g_M*_{t-1} + e_M       trend MFP growth
    g_K_t  ~ N(g_K*_t, sigma_K)     observed capital growth
    g_L_t  ~ N(g_L*_t, sigma_L)     observed hours growth
    mfp_t  ~ N(g_M*_t, sigma_M)     Solow residual, given alpha_t
    g_Y*_t = alpha_t·g_K*_t + (1-alpha_t)·g_L*_t + g_M*_t
    y*_t   = y*_{t-1} + g_Y*_{t-1}

The level and the gap are untouched: `inflation_gap.py` still supplies
`gap = c·(pi - anchor)` and fits GDP around it, so the inflation fulcrum still
positions potential. Only the source of its *growth* changes.

**Why the smoothing ratios must differ.** Each trend is a random walk observed
with noise, and the imposed ratio of trend innovation sd to observation sd is
what sets the smoothing.

These ratios are not HP lambdas, and an earlier version of this docstring
claimed they were. HP(lambda) is the local *linear trend* model, in which
lambda is a variance ratio against the innovation to the slope of an I(2)
trend; these are local *level* models, so the ratio is against the innovation
to the level and smooths far harder for the same nominal lambda. The
interpretable quantity is how far a trend can wander across the sample,
`r x sigma_obs x sqrt(T)`.

If those ratios were equal and alpha were constant, this specification would
be pointless, and provably so. HP is a linear filter, so

    alpha·HP(g_K) + (1-alpha)·HP(g_L) + HP(g_Y - alpha·g_K - (1-alpha)·g_L)
      = HP(g_Y)

exactly: the factor terms cancel and potential growth collapses to a smoothed
GDP growth rate, with capital, hours, MFP and alpha contributing nothing. That
is not a hypothetical. It is what the deterministic `cobb_douglas` package
does, verified to 0.000 at every quarter.

What breaks the cancellation is applying *different* smoothing to series with
different cyclicality, and the data say they differ sharply: over 1978Q4
onward the HP(1600) cycle is 37% of the variation in capital growth and 97% of
it in hours growth. Putting both through one filter under-smooths labour
badly. A time-varying alpha breaks the cancellation too, but only just, since
a smoothed capital share drifts slowly; on this sample it is worth about
0.02pp. The smoothing difference is doing essentially all the work.

**What this does not buy.** The ratios are imposed, not estimated, and the
answer moves with them: trend growth at the sample end runs from about 1.8 to
2.2 across defensible settings for the hours ratio alone. That is the
Stock-Watson pile-up problem in its usual form and freeing the ratios does not
escape it. So this specification is a way of writing the smoothing choice down
explicitly, not a way of learning it from data. The compensating gain is that
trend MFP becomes a state with a credible interval, where `decompose.py` can
only give it as a residual absorbing every error in the hours trend.

**A note on `scale.py`.** It derives a `sigma_<name>` float from every
`ratio_<name>` it is given, so it will produce `sigma_gk = ratio_gk · sigma_c`
and report that in the run log. This specification does not use those: its
smoothing is a ratio to an *estimated* observation sd, not an absolute sd, and
that is what makes it equivalent to an HP lambda rather than to a fixed
innovation size. The estimated observation sds are therefore named
`sigma_obs_gk`, `sigma_obs_gl` and `sigma_obs_gm`, so nothing shares a name
with the unused derived constants.

**alpha.** The capital share is a fourth latent trend, `a*`, observed by the
published ABS series and smoothed on the same footing as the factor trends. It
is not estimated in the sense of being inferred from fit: the Solow residual is
an accounting identity, so alpha is not identified against it and any value
would be absorbed by the residual. It is *conditioned on the published series*,
with the model doing the smoothing rather than a filter applied beforehand.

It is smoothed almost to a constant, 0.335 to 0.338 against a published range
of 0.299 to 0.406, because **most of the published movement is artefact**.
alpha is GOS / (GOS + COE) and correlates +0.852 with the terms of trade in
levels, while correlating −0.077 with potential growth in changes. Rising
commodity prices put mining revenue into GOS without a matching rise in COE,
since the wage bill does not scale with the ore price, so the share moves with
no change in what the economy can produce. Cobb-Douglas also assumes an
elasticity of substitution of one, which implies constant shares outright, so a
drifting alpha would be inconsistent with the functional form wrapped around
it. See `ModelConfig.ratio_a`.

The Solow residual is built with the *raw* share, so the identity
`g_Y = a·g_K + (1-a)·g_L + mfp` holds exactly in the data and potential is
assembled from the trend of each component, rather than from an identity that
is part-smoothed and part-not.
"""

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.common.spline import basis
from src.models.ystar.base import set_model_coefficients


def _factor_trend(
    name: str,
    observed: np.ndarray,
    ratio: float,
    n_periods: int,
) -> pt.TensorVariable:
    """One factor: a random-walk trend observed with noise, smoothing imposed.

    `ratio` is the trend innovation sd as a fraction of the observation sd, so
    the smoothing is scale-free and the estimated `sigma_obs_{name}` only sets the
    overall noise level. See the module docstring for the HP equivalence.
    """
    sigma = pm.HalfNormal(f"sigma_obs_{name}", sigma=1.0)

    innovations = pm.Normal(
        f"{name}_innovations",
        mu=0.0,
        sigma=ratio * sigma,
        shape=n_periods - 1,
    )
    initial = pm.Normal(f"initial_{name}", mu=float(np.mean(observed)), sigma=2.0)
    trend = pm.Deterministic(
        f"trend_{name}",
        pt.concatenate([[initial], initial + pt.cumsum(innovations)]),
    )

    pm.Normal(f"observed_{name}", mu=trend, sigma=sigma, observed=observed)

    return trend


def _polynomial_trend(
    name: str,
    observed: np.ndarray,
    obs_index: pd.PeriodIndex,
    degree: int,
) -> pt.TensorVariable:
    """Build a factor trend as a global polynomial in time, with no innovations.

    For `g_M*`, where a random walk is the wrong object. MFP is the Solow
    residual, so a smoothed random walk through it is a smoothed residual of
    GDP: the cycle enters potential through this component and nothing else.
    Capital and hours trends do not have the problem, being smooth series in
    their own right, so only this one is replaced.

    Free ends and no interior knots, for the reason `potential.py` records:
    end conditions on a trend that must keep moving drag its slope at both
    boundaries, while a global polynomial has no local segment to distort.
    """
    design = basis(obs_index, (), natural=False, degree=degree)
    coef = pm.Normal(
        f"{name}_coef", mu=float(np.mean(observed)), sigma=2.0, shape=design.shape[1],
    )
    return pm.Deterministic(f"trend_{name}", pt.dot(pt.as_tensor_variable(design), coef))


def _imposed_trend(
    name: str,
    start: float,
    sigma: float,
    n_periods: int,
) -> pt.TensorVariable:
    """Build a random-walk trend with an imposed innovation sd and no observation.

    Used for `g_M*` when the MFP observation is dropped: with no observation
    equation there is no `sigma_obs` to take a ratio against, so the smoothing
    has to be an absolute sd, in the same style as `sigma_g` in `potential.py`.
    The trend is then identified only through the level equation, by way of the
    cumulation into `y*`.
    """
    innovations = pm.Normal(f"{name}_innovations", mu=0.0, sigma=sigma, shape=n_periods - 1)
    initial = pm.Normal(f"initial_{name}", mu=start, sigma=2.0)
    return pm.Deterministic(
        f"trend_{name}",
        pt.concatenate([[initial], initial + pt.cumsum(innovations)]),
    )


def production_potential_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Potential output whose growth comes from trend capital, hours and MFP.

    `constant` must carry `ratio_gk`, `ratio_gl` and `ratio_gm`. `obs` must
    carry `g_k`, `g_l`, `mfp` and `alpha` alongside `log_gdp`.
    """
    if constant is None:
        constant = {}

    mfp_observed = bool(constant.get("mfp_observed", True))
    if constant.get("mfp_degree"):
        required = ("ratio_gk", "ratio_gl", "ratio_a")
    else:
        required = ("ratio_gk", "ratio_gl", "ratio_a", "ratio_gm" if mfp_observed else "sigma_gm")
    missing = [key for key in required if key not in constant]
    if missing:
        raise ValueError(f"production_potential_equation requires {missing}")
    for key in ("g_k", "g_l", "mfp", "alpha"):
        if key not in obs:
            raise ValueError(f"production_potential_equation requires obs[{key!r}]")

    n_periods = len(obs["log_gdp"])

    with model:
        trend_k = _factor_trend("gk", np.asarray(obs["g_k"], float), float(constant["ratio_gk"]), n_periods)
        trend_l = _factor_trend("gl", np.asarray(obs["g_l"], float), float(constant["ratio_gl"]), n_periods)
        mfp = np.asarray(obs["mfp"], dtype=float)
        mfp_degree = int(constant.get("mfp_degree", 0))
        if mfp_degree:
            if constant.get("obs_index") is None:
                raise ValueError("a polynomial MFP trend needs 'obs_index'")
            trend_m = _polynomial_trend("gm", mfp, constant["obs_index"], mfp_degree)
        elif mfp_observed:
            trend_m = _factor_trend("gm", mfp, float(constant["ratio_gm"]), n_periods)
        else:
            trend_m = _imposed_trend("gm", float(mfp.mean()), float(constant["sigma_gm"]), n_periods)
        # The capital share is smoothed here rather than before the data
        # reaches the model, and conditioned on the published series. Same
        # machinery as the factor trends, so the smoothing is declared in one
        # place and carries uncertainty into potential growth.
        trend_a = _factor_trend("a", np.asarray(obs["alpha"], float), float(constant["ratio_a"]), n_periods)

        # The production identity, applied to the trends rather than the data.
        trend_growth = pm.Deterministic(
            "trend_growth",
            trend_a * trend_k + (1.0 - trend_a) * trend_l + trend_m,
        )

        # Potential is the cumulation. The drift enters lagged, matching
        # `potential.py`, so there is no simultaneity between level and growth.
        # No separate level innovation: everything potential does comes from
        # the factor trends, which is the point of the specification.
        settings = {"initial_potential": {"mu": float(obs["log_gdp"][0]), "sigma": 2.0}}
        mc = set_model_coefficients(model, settings, constant)
        y_init = mc["initial_potential"]
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[y_init], y_init + pt.cumsum(trend_growth[:-1])]),
        )

    latents["trend_growth"] = trend_growth
    latents["potential_output"] = potential_output
    latents["trend_gk"] = trend_k
    latents["trend_gl"] = trend_l
    latents["trend_gm"] = trend_m
    latents["trend_a"] = trend_a

    mfp_term = (
        f"r_M={constant['ratio_gm']:g}"
        if mfp_observed
        else f"sigma_gm={constant['sigma_gm']:g}, MFP not observed"
    )
    return (
        f"g_Y* = a*·g_K* + (1-a*)·g_L* + g_M*  "
        f"(r_K={constant['ratio_gk']:g}, r_L={constant['ratio_gl']:g}, "
        f"{mfp_term}, r_a={constant['ratio_a']:g});"
        f"  y*_t = y*_{{t-1}} + g_Y*_{{t-1}}"
    )
