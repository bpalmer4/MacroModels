"""r-star identity: convex blend of trend growth and bond-implied real rate.

    r*_t = alpha * g_t + (1 - alpha) * (indexed_10y_t - k) + eps_t
           eps_t ~ N(0, sigma_r)

Replaces the canonical HLW z_star random walk with a single scalar weight
alpha that the data can identify through the IS curve. The bond yield
appears in the *definition* of r*, not as a separate observation, so there
is no separate indexed_bond observation equation.

- alpha = 1 reduces to r* = g (Resolution A in MODEL_NOTES.md)
- alpha = 0 reduces to r* = indexed_10y - k (Resolution B-like)
- The data picks where to land via the alpha posterior

The eps_t term is i.i.d. (not a random walk) — it gives r* a small posterior
spread around the blend without re-introducing the unidentified random walk
that previously bedevilled identification. Because the IS curve already has
its own sigma_IS absorbing rate-gap noise (and a_r is small), sigma_r is
likely to be largely prior-dominated. eps_t is non-centred (sample raw N(0,1),
scale by sigma_r) to avoid a Neal's funnel when sigma_r is small.

Extensions tested and reverted (see MODEL_NOTES.md):
- Regime-switching alpha (alpha_pre / alpha_post around 2008Q4): regimes
  not separated, sampling much worse.
- Regime-switching alpha (alpha_normal / alpha_divergence, 2011Q3–2021Q4
  "great divergence" period): regimes again not separated (0.54 vs 0.50),
  267 divergences, rhat warnings. Reverted.
- Slope-based time-varying k_t = k0 + k_slope * (10y_nominal - cash):
  k_slope was identifiable but sampling collapsed (10,655 divergences,
  r_star ESS 17). Substantive answer barely changed.
- Intercept c replacing k: r* = alpha*g + (1-alpha)*indexed + c + eps.
  g still 2.66%, 359 divergences. c did not release g.
- Intercept c alongside k: r* = alpha*g + (1-alpha)*(indexed-k) + c + eps.
  2,104 divergences — c and (1-alpha)*k collinear, sampler collapsed.

Requires obs["indexed_10y"]. Must run after trend_growth_equation.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def r_star_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """r* as a weighted blend of trend growth and indexed bond yield."""
    if constant is None:
        constant = {}

    if "trend_growth" not in latents:
        raise RuntimeError(
            "r_star_equation requires trend_growth in latents — "
            "ensure trend_growth_equation runs first.",
        )
    if "indexed_10y" not in obs:
        raise RuntimeError(
            "r_star_equation requires obs['indexed_10y'] — "
            "ensure observations.py loads the indexed bond yield.",
        )

    n_periods = len(obs["log_gdp"])
    g = latents["trend_growth"]
    indexed = obs["indexed_10y"]

    with model:
        # alpha uses Beta — set_model_coefficients only handles Normal/HalfNormal/Truncated
        if "alpha_rstar" in constant:
            alpha = constant["alpha_rstar"]
            if not hasattr(model, "_fixed_constants"):
                model._fixed_constants = {}  # noqa: SLF001
            model._fixed_constants["alpha_rstar"] = alpha  # noqa: SLF001
        else:
            alpha = pm.Beta("alpha_rstar", alpha=2.0, beta=2.0)

        settings = {
            "k": {"mu": 0.5, "sigma": 0.5, "lower": 0.0},
            "sigma_r": {"sigma": 0.3},
        }
        mc = set_model_coefficients(model, settings, constant)

        blend = alpha * g + (1 - alpha) * (indexed - mc["k"])

        if "r_innovation" in constant:
            r_innovation = constant["r_innovation"]
        else:
            # Non-centred parameterization: sample raw N(0,1) and scale by
            # sigma_r. Eliminates the Neal's funnel when sigma_r is small.
            r_innovation_raw = pm.Normal(
                "r_innovation_raw", mu=0, sigma=1, shape=n_periods,
            )
            r_innovation = pm.Deterministic(
                "r_innovation", r_innovation_raw * mc["sigma_r"],
            )

        r_star = pm.Deterministic("r_star", blend + r_innovation)

    latents["r_star"] = r_star
    return "r*_t = alpha*g_t + (1-alpha)*(indexed_10y_t - k) + eps_t,  eps ~ N(0, sigma_r)"
