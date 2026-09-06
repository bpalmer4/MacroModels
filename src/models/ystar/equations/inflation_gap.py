"""Potential output with the gap defined by inflation, and GDP fitted.

Three lines:

    y*_t      = y*_{t-1} + g_{t-1} + e_y      potential: a random walk (potential.py)
    gap_t     = c · (pi_t - anchor)           gap: defined by inflation
    log_gdp_t = y*_t + gap_t + e_c            output: fitted, with a residual

The gap is proportional to inflation's deviation from target. Inflation supplies
the sign, the timing and the relative magnitude: six per cent inflation means a
bigger gap than three. What it cannot supply is `c`, the conversion from
percentage points of inflation into per cent of output, because nothing in a
price index is denominated in units of GDP. `c` is estimated here.

**The third line is what makes this a model rather than a definition.** An
earlier version set `y* = log_gdp - c·d` outright. That reproduced GDP exactly
by construction, so there was no residual, no fit, and nothing the model could
get wrong: every wiggle in output that inflation did not account for was forced
into potential, and potential duly tracked GDP through the pandemic. With `e_c`
present, trend and inflation-implied gap no longer have to exhaust output, and
what they miss is measured instead of absorbed.

`sigma_e` is therefore the diagnostic the exercise needs. It says how much
Australian output variation the inflation-defined gap fails to explain, which is
the amplitude question answered with a number rather than an argument.

Note what is *not* here. No Phillips curve: `c` is not estimated by regressing
inflation on the gap. No IS curve and no policy rule. And no cycle dynamics:
`e_c` is white noise, so nothing is imposed about how the unexplained part of
output behaves over time, which is the opposite of assuming an AR(2) cycle.

White noise is a choice rather than an absence, though, and `e_c` is serially
correlated in fact: lag-1 autocorrelation 0.51. That is expected rather than
troubling. Policy acts with lags and expectations feed inflation, so the part
of output that contemporaneous inflation does not account for is bound to
persist. What follows is that `e_c` is where the omitted IS curve and
expectations block show through, and modelling it would mean reintroducing, in
reduced form, the structure this package exists to do without.

The residual is therefore left unmodelled deliberately, and its persistence is
not a defect to be patched. `ModelConfig.ar1_residual` fits an AR(1) to it
anyway, as a diagnostic rather than a candidate specification; see iteration
log item 15 for what it shows and why it is not adopted. Under that option
`sigma_e` becomes the innovation sd and `sigma_e_total` the stationary
residual sd.

The second line is contemporaneous, which asserts that inflation accompanies
the gap rather than following it. Letting inflation lead was tried and did not
earn its keep; see "Explored but did not work" in MODEL_NOTES.md.

One identification point. `sigma_ystar` and `sigma_e` compete for the same
variation, since a quarterly wiggle in GDP can be a shift in potential or a
residual. That is the Stock-Watson pile-up problem in its proper form, and one
of the two must be pinned. `sigma_ystar` is the one imposed (see `scale.py`),
which is what "potential is smooth" means operationally; `sigma_e` is free so
that it can report what is left over.
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.ystar.base import set_model_coefficients


def inflation_gap_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Gap as a scaled inflation deviation, with GDP fitted around it.

    Model: gap_t = c · (pi_t - anchor)
           log_gdp_t = y*_t + gap_t + e_c,   e_c ~ N(0, sigma_e)

    `anchor` must be supplied via `constant`. `latents` must already carry
    `potential_output` (see `potential.py`).
    """
    if constant is None:
        constant = {}

    if "anchor" not in constant:
        raise ValueError("inflation_gap_equation requires a fixed 'anchor'")
    if "potential_output" not in latents:
        raise ValueError("inflation_gap_equation requires potential_output — run potential.py first")

    anchor = float(constant["anchor"])
    ar1 = bool(constant.get("ar1_residual", False))
    two_sided_c = bool(constant.get("two_sided_c", False))
    deviation = np.asarray(obs["pi"], dtype=float) - anchor

    with model:
        # c is per cent of output per percentage point of inflation deviation.
        # The default prior is HalfNormal, so the sign is *imposed*: above-target
        # inflation means output above potential. That is the model's premise
        # rather than a finding, and it means "c's interval is clear of zero"
        # cannot be read as evidence for the premise, because the prior forbids
        # the alternative. `two_sided_c` swaps in Normal(0, 2) so the data can
        # place mass below zero, which is the only form in which the premise is
        # actually tested. Weak either way, since c is what the exercise is about.
        settings = {
            "c": {"mu": 0.0, "sigma": 2.0} if two_sided_c else {"sigma": 2.0},
            "sigma_e": {"sigma": 1.0},
        }
        if ar1:
            # Weak and roughly flat over the stationary region: the point of
            # freeing rho is to let the data say how persistent the residual
            # is, so the prior must not answer that question. Two-sided,
            # because a negative rho is a real (if unlikely) answer.
            settings["rho_e"] = {"mu": 0.0, "sigma": 0.5, "lower": -0.99, "upper": 0.99}
        mc = set_model_coefficients(model, settings, constant)

        potential_output = latents["potential_output"]
        output_gap = pm.Deterministic("output_gap", mc["c"] * deviation)

        if not ar1:
            pm.Normal(
                "observed_gdp",
                mu=potential_output + output_gap,
                sigma=mc["sigma_e"],
                observed=obs["log_gdp"],
            )
        else:
            # e_c,t = rho·e_c,{t-1} + eps_t. The lagged residual is observable
            # given the states, since log_gdp is data, so the likelihood is
            # written directly on GDP with the AR term in the mean rather than
            # as a latent process: no extra states, no filtering.
            #
            # `sigma_e` is the *innovation* sd here, not the residual sd. The
            # comparable quantity to the white-noise run's 0.98 is the
            # stationary sd, recorded as `sigma_e_total`.
            log_gdp = np.asarray(obs["log_gdp"], dtype=float)
            fitted = potential_output + output_gap
            rho = mc["rho_e"]
            stationary_sigma = mc["sigma_e"] / pt.sqrt(1.0 - rho**2)

            # The first observation carries the stationary distribution rather
            # than being conditioned away. Dropping it would be the standard
            # conditional likelihood; keeping it matters here because rho is
            # the parameter under test and the exact likelihood is what makes
            # the two runs' `sigma_e` comparable at rho = 0.
            pm.Normal(
                "observed_gdp_initial",
                mu=fitted[0],
                sigma=stationary_sigma,
                observed=log_gdp[0],
            )
            pm.Normal(
                "observed_gdp",
                mu=fitted[1:] + rho * (log_gdp[:-1] - fitted[:-1]),
                sigma=mc["sigma_e"],
                observed=log_gdp[1:],
            )
            pm.Deterministic("sigma_e_total", stationary_sigma)

        pm.Deterministic("inflation_deviation", pt.as_tensor_variable(deviation))

    latents["output_gap"] = output_gap

    residual = "e_c ~ AR(1)" if ar1 else "e_c ~ N(0, sigma_e)"
    return f"gap_t = c · (pi_t - {anchor:g});  log_gdp_t = y*_t + gap_t + e_c,  {residual}"
