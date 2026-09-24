"""IS curve equation linking output gap to interest rates and fiscal impulse."""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.common.model_constants import record_constant
from src.models.nairu.base import set_model_coefficients


def is_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
    *,
    rstar_blend: bool = False,
    rstar_blend_alpha_prior: tuple[float, float] = (1.0, 1.0),
    rstar_blend_alpha_fixed: float | None = None,
    rstar_blend_k: float = 0.0,
) -> str:
    """IS curve linking output gap to interest rates and fiscal impulse.

    Model: y_gap = rho x y_gap_{-1} - beta x r_gap_{-2} + gamma x fiscal_{-1} + e

    When ``rstar_blend`` is True, the rate gap is built against a blended r*
    estimated by the model rather than the fixed Cobb-Douglas growth r*:

        r* = alpha_rstar x growth_anchor + (1 - alpha_rstar) x (yield_anchor - k)

    alpha_rstar ~ Beta(a, b) (flat Beta(1, 1) by default). alpha=1 recovers the
    growth anchor (obs["det_r_star"]); alpha=0 the real bond yield. The data
    identifies alpha only through this single (weak) rate-gap channel, so a flat
    prior makes the posterior a clean read on how much signal there is.
    """
    if constant is None:
        constant = {}

    potential_output = latents["potential_output"]

    with model:
        settings = {
            "rho_is": {"mu": 0.85, "sigma": 0.1},
            "beta_is": {"mu": 0.20, "sigma": 0.10, "lower": 0},
            "gamma_fi": {"mu": 0.05, "sigma": 0.2, "lower": 0},
            "epsilon_is": {"sigma": 0.4},
        }
        mc = set_model_coefficients(model, settings, constant)

        real_rate = obs["cash_rate"] - obs["π_exp"]
        if rstar_blend:
            # Override the default fixed-blend det_r_star with an in-model blend of
            # the pure growth anchor and the yield anchor, so α can be re-estimated
            # (free Beta) or imposed at a different value (rstar_blend_alpha_fixed).
            if "rstar_growth" not in obs or "yield_anchor" not in obs:
                raise RuntimeError(
                    "rstar_blend=True requires obs['rstar_growth'] and obs['yield_anchor'] — "
                    "ensure observations.py loads the growth and bond-yield anchors.",
                )
            if rstar_blend_alpha_fixed is not None:
                alpha_rstar = float(rstar_blend_alpha_fixed)
                record_constant(model, "alpha_rstar", alpha_rstar)
            else:
                a_param, b_param = rstar_blend_alpha_prior
                alpha_rstar = pm.Beta("alpha_rstar", alpha=float(a_param), beta=float(b_param))
            blend = (
                alpha_rstar * obs["rstar_growth"]
                + (1.0 - alpha_rstar) * (obs["yield_anchor"] - rstar_blend_k)
            )
            # pt.as_tensor_variable so a fixed (float) alpha — which yields a plain
            # numpy array — can still be wrapped as a Deterministic for the trace.
            r_star = pm.Deterministic("r_star_blend", pt.as_tensor_variable(blend))
            rate_gap = real_rate - r_star
        else:
            rate_gap = real_rate - obs["det_r_star"]
        rate_gap_lag2 = rate_gap[:-2]

        output_gap = obs["log_gdp"] - potential_output
        output_gap_lag1 = output_gap[1:-1]
        potential_t = potential_output[2:]
        fiscal_impulse_lag1 = obs["fiscal_impulse_1"][2:]

        predicted_log_gdp = (
            potential_t
            + mc["rho_is"] * output_gap_lag1
            - mc["beta_is"] * rate_gap_lag2
            + mc["gamma_fi"] * fiscal_impulse_lag1
        )

        pm.Normal(
            "observed_is",
            mu=predicted_log_gdp,
            sigma=mc["epsilon_is"],
            observed=obs["log_gdp"][2:],
        )

    return "y_gap = rho x y_gap_{-1} - beta x r_gap_{-2} + gamma x fiscal_{-1} + e"
