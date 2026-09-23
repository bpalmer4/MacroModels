"""HLW IS curve with latent r* and an opt-in SOE block of external regressors.

Observation equation indexed from t=2 onwards (needs two output-gap and two
real-rate-gap lags).

The fiscal impulse helps the IS curve do real work — without it, sigma_IS
absorbs both demand shocks and fiscal effects, leaving little explanatory
power for the real rate gap. With it, a_r can plausibly identify away from
zero.

The SOE-block regressors (active when their obs key is present, used by
Resolution D) target three external channels for the Australian
small-open-economy context:

- ``tot_change_1``: terms-of-trade growth (lag 1q). Captures the
  income/price effect of commodity-price swings (mining booms, post-2014
  correction) that would otherwise be absorbed into sigma_IS.
- ``twi_change_1``: trade-weighted index change (lag 1q). Captures the
  exchange-rate / competitiveness channel — AUD depreciation supports
  net exports and demand independently of the rate gap. Sign is negative
  (AUD appreciation suppresses demand).
- ``icp_change_1``: RBA Index of Commodity Prices (A$) growth (lag 1q).
  Upstream price signal for Asian commodity demand (China iron-ore, Japan
  LNG, Korea coal) that's more exogenous than ToT (no import-price
  denominator). Sign positive.

The hypothesis being tested: in canonical HLW (Resolution A), the IS curve
is too weak to identify r* because sigma_IS absorbs SOE shocks attributed to
the wrong channel. Adding the SOE block should shrink sigma_IS and let a_r
firm up, which in turn lets the latent z in canonical HLW identify.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients
from src.models.rstar_hlw.equations.exclusion import drop_excluded

# The optional open-economy and fiscal regressors, each included only when
# `build_observations` supplied it: resolutions D and F carry the SOE block,
# A and B carry none of this, C/E/G/H carry the fiscal impulse alone.
#
# Held as a table rather than as eight separate branches so adding one does
# not push this function past the complexity limit again.
_OPTIONAL_REGRESSORS: tuple[tuple[str, str, dict[str, float]], ...] = (
    # Fiscal impulse. Positive: a fiscal expansion adds to demand.
    ("fiscal_impulse_1", "gamma_fi", {"mu": 0.05, "sigma": 0.20, "lower": 0.0}),
    # ToT growth is in % per quarter; gamma_tot translates 1pp of ToT change
    # into log-points of output gap. Positive sign expected.
    ("tot_change_1", "gamma_tot", {"mu": 0.05, "sigma": 0.10, "lower": 0.0}),
    # AUD appreciation suppresses demand (net exports + competitiveness).
    # Sign expected negative.
    ("twi_change_1", "gamma_twi", {"mu": -0.05, "sigma": 0.10, "upper": 0.0}),
    # RBA ICP (AUD) growth: an upstream commodity-price demand signal for
    # Asian buyers. Positive sign expected.
    ("icp_change_1", "gamma_icp", {"mu": 0.05, "sigma": 0.10, "lower": 0.0}),
)


# The two priors on the IS slope, side by side so the difference between
# them is visible in one place.
#
# HLW2017 impose only that a_r is negative, calling it a "minimal prior" that
# facilitates convergence of the numerical optimisation. This repo's default
# also asserts a magnitude, centring the slope at -0.15 with an sd of 0.08,
# against a posterior that comes back near -0.04. That is an informative
# prior pulling in a direction the data do not support, so a run claiming to
# be canonical must not carry it.
#
# The sign-only sd of 0.5 is not an estimate of anything. It is wide enough
# that the truncated normal is close to flat across the range the slope could
# plausibly occupy, which is what "sign only" has to mean in a prior.
_A_R_PRIOR = {"mu": -0.15, "sigma": 0.08, "upper": 0.0}
_A_R_PRIOR_SIGN_ONLY = {"mu": 0.0, "sigma": 0.5, "upper": 0.0}


def is_curve_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    *,
    constant: dict[str, Any] | None = None,
    rate_lag: int | None = None,
    keep: np.ndarray | None = None,
    sign_prior_only: bool = False,
) -> str:
    """HLW (2017) IS curve in level form, with fiscal impulse.

    Model:
        log_gdp_t = y*_t
                  + a_y1 * y_gap_{t-1}
                  + a_y2 * y_gap_{t-2}
                  + (a_r/2) * (r_gap_{t-1} + r_gap_{t-2})
                  + gamma_fi * fiscal_impulse_{t-1}
                  + e_IS

    where r_gap = (cash_rate - pi_exp) - r*, all annualised %.
    Output gap is in log x 100 units.

    `rate_lag` replaces the averaged (t-1, t-2) rate gap with a SINGLE lag, so
    the rate term becomes `a_r * r_gap_{t-rate_lag}`. HLW's own shape is the
    default (None). A longer single lag is what the `is_curve` bench and
    `rstar_invert` both point at: the slope there strengthens monotonically
    from lag 1 to 5 and the bench's negative-slope sample peaks near 6, because
    a regressor further from t carries less of the RBA's reaction to the
    economy. Setting it also makes `a_r` directly comparable with
    `rstar_invert`'s single-lag runs, ONCE persistence is accounted for:
    `a_r` here is an IMPACT coefficient and the comparable level response is
    `a_r / (1 - a_y1 - a_y2)`.

    UNITS WARNING. Both forms take the coefficient on a rate gap in annualised
    percentage points, but the averaged form applies `a_r/2` to each of two
    lags while the single-lag form applies the whole of `a_r` to one. `a_r`
    means the same thing in both (the response to a SUSTAINED rate gap), so
    the two are comparable with each other; what is not comparable is `a_r`
    against a level slope from a model without persistence.

    `keep` drops a window of quarters from this likelihood (see
    `exclusion.py`). This equation is the one that makes potential track GDP,
    so it is where the lockdown quarters did their damage.
    """
    if constant is None:
        constant = {}

    r_star = latents["r_star"]
    potential_output = latents["potential_output"]

    with model:
        settings = {
            "a_y1": {"mu": 0.90, "sigma": 0.10, "lower": 0.0, "upper": 1.0},
            "a_y2": {"mu": -0.10, "sigma": 0.10, "upper": 0.0},
            "a_r": dict(_A_R_PRIOR_SIGN_ONLY if sign_prior_only else _A_R_PRIOR),
            "sigma_IS": {"sigma": 0.4},
        }
        present = [entry for entry in _OPTIONAL_REGRESSORS if entry[0] in obs]
        settings.update({name: prior for _, name, prior in present})
        mc = set_model_coefficients(model, settings, constant)

        real_rate = obs["cash_rate"] - obs["pi_exp"]
        r_gap = real_rate - r_star

        output_gap = obs["log_gdp"] - potential_output

        # The first quarter that has every lag the equation needs. HLW's own
        # shape needs two; a longer single rate lag needs that many.
        start = 2 if rate_lag is None else max(2, rate_lag)
        end = len(obs["log_gdp"])

        if rate_lag is None:
            rate_term = (mc["a_r"] / 2) * (
                r_gap[start - 1:end - 1] + r_gap[start - 2:end - 2]
            )
        else:
            rate_term = mc["a_r"] * r_gap[start - rate_lag:end - rate_lag]

        predicted_log_gdp = (
            potential_output[start:]
            + mc["a_y1"] * output_gap[start - 1:end - 1]
            + mc["a_y2"] * output_gap[start - 2:end - 2]
            + rate_term
        )

        for key, name, _ in present:
            predicted_log_gdp = predicted_log_gdp + mc[name] * obs[key][start:]

        # This is the equation that ties potential to GDP, so it is the one
        # that made potential absorb the lockdown collapse. See exclusion.py.
        fitted, observed = drop_excluded(
            keep, start, predicted_log_gdp, np.asarray(obs["log_gdp"][start:], dtype=float),
        )

        pm.Normal(
            "observed_IS",
            mu=fitted,
            sigma=mc["sigma_IS"],
            observed=observed,
        )

    rate_desc = (
        "(a_r/2)*(r_gap_{t-1}+r_gap_{t-2})" if rate_lag is None
        else f"a_r*r_gap_{{t-{rate_lag}}}"
    )
    parts = [f"y_gap_t = a_y1*y_gap_{{t-1}} + a_y2*y_gap_{{t-2}} + {rate_desc}"]
    parts.extend(
        f"{name}*{key.removesuffix('_1')}_{{t-1}}" for key, name, _ in present
    )
    parts.append("e_IS")
    return " + ".join(parts)
