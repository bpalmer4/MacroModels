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


def is_curve_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
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
    """
    if constant is None:
        constant = {}

    r_star = latents["r_star"]
    potential_output = latents["potential_output"]

    with model:
        settings = {
            "a_y1": {"mu": 0.90, "sigma": 0.10, "lower": 0.0, "upper": 1.0},
            "a_y2": {"mu": -0.10, "sigma": 0.10, "upper": 0.0},
            "a_r": {"mu": -0.15, "sigma": 0.08, "upper": 0.0},
            "sigma_IS": {"sigma": 0.4},
        }
        if "fiscal_impulse_1" in obs:
            settings["gamma_fi"] = {"mu": 0.05, "sigma": 0.20, "lower": 0.0}
        if "tot_change_1" in obs:
            # ToT growth is in % per quarter; gamma_tot translates 1pp of ToT
            # change into log-points of output gap. Positive sign expected.
            settings["gamma_tot"] = {"mu": 0.05, "sigma": 0.10, "lower": 0.0}
        if "twi_change_1" in obs:
            # AUD appreciation suppresses demand (net exports + competitiveness).
            # Sign expected negative.
            settings["gamma_twi"] = {"mu": -0.05, "sigma": 0.10, "upper": 0.0}
        if "icp_change_1" in obs:
            # RBA ICP (AUD) growth — upstream commodity-price demand signal
            # for Asian buyers. Positive sign expected.
            settings["gamma_icp"] = {"mu": 0.05, "sigma": 0.10, "lower": 0.0}
        mc = set_model_coefficients(model, settings, constant)

        real_rate = obs["cash_rate"] - obs["pi_exp"]
        r_gap = real_rate - r_star

        output_gap = obs["log_gdp"] - potential_output

        # t = 2 .. T-1 (we predict log_gdp from index 2 onwards)
        predicted_log_gdp = (
            potential_output[2:]
            + mc["a_y1"] * output_gap[1:-1]
            + mc["a_y2"] * output_gap[:-2]
            + (mc["a_r"] / 2) * (r_gap[1:-1] + r_gap[:-2])
        )

        if "fiscal_impulse_1" in obs:
            predicted_log_gdp = predicted_log_gdp + mc["gamma_fi"] * obs["fiscal_impulse_1"][2:]
        if "tot_change_1" in obs:
            predicted_log_gdp = predicted_log_gdp + mc["gamma_tot"] * obs["tot_change_1"][2:]
        if "twi_change_1" in obs:
            predicted_log_gdp = predicted_log_gdp + mc["gamma_twi"] * obs["twi_change_1"][2:]
        if "icp_change_1" in obs:
            predicted_log_gdp = predicted_log_gdp + mc["gamma_icp"] * obs["icp_change_1"][2:]

        pm.Normal(
            "observed_IS",
            mu=predicted_log_gdp,
            sigma=mc["sigma_IS"],
            observed=obs["log_gdp"][2:],
        )

    parts = [
        "y_gap_t = a_y1*y_gap_{t-1} + a_y2*y_gap_{t-2} + (a_r/2)*(r_gap_{t-1}+r_gap_{t-2})",
    ]
    if "fiscal_impulse_1" in obs:
        parts.append("gamma_fi*fiscal_{t-1}")
    if "tot_change_1" in obs:
        parts.append("gamma_tot*tot_change_{t-1}")
    if "twi_change_1" in obs:
        parts.append("gamma_twi*twi_change_{t-1}")
    if "icp_change_1" in obs:
        parts.append("gamma_icp*icp_change_{t-1}")
    parts.append("e_IS")
    return " + ".join(parts)
