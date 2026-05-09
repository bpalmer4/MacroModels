"""r-star with time-varying alpha (Resolution H).

    r*_t  =  alpha_t * g_t + (1 - alpha_t) * (indexed_10y_t - k) + eps_t,  eps ~ N(0, sigma_r)

    logit_alpha_0       ~  N(0, 2)                       [initial value, broad]
    logit_alpha_t       =  logit_alpha_{t-1} + sigma_a * eps_a_t,  eps_a ~ N(0, 1)
    alpha_t             =  sigmoid(logit_alpha_t)

Generalises C and G by letting the blend weight alpha drift over time on the
logit scale. This addresses the structural limitation of constant-alpha
specifications that produce wide r* posterior CIs at every t — the wide CI
under constant alpha represents *sample-wide* uncertainty about a single
alpha projected period-by-period, not period-specific uncertainty about r*.

A time-varying alpha can express "alpha was high in the 1990s and lower
post-GFC" if the data supports it. Tested in iterations 7 (alpha_pre vs
alpha_post around 2008Q4) and 13 (alpha_normal vs alpha_divergence around
2011Q3-2021Q4) failed because the data couldn't pick a hard regime date. A
continuous logit-RW allows smooth gradients rather than forcing a breakpoint
— the data can drift alpha without committing to a specific transition date.

Design choices:

- **Logit scale RW** (not direct RW on alpha). Keeps alpha_t in (0, 1) without
  truncation pile-up at the boundaries; the chain explores freely on the
  unbounded logit scale.
- **sigma_a fixed at 0.05** (default; overridable via constant). This is the
  RW innovation scale on the logit scale — corresponds to roughly 1pp of
  alpha movement per quarter at alpha = 0.5 (less near 0 or 1). Slow drift
  consistent with structural shifts (mining boom, post-GFC, COVID).
- **Non-centred parameterisation** (eps_a_t ~ N(0,1) scaled by sigma_a) for
  sampling robustness even though sigma_a is fixed (not strictly needed but
  cheap and consistent with E's z reparameterisation).
- **No hyperprior on alpha shape**: time-varying alpha already allows the
  data to express its preference period-by-period; an additional Beta shape
  hyperprior would be redundant complexity.

Risks:
- sigma_a too small: alpha_t looks constant, no advantage over G
- sigma_a too large: alpha_t fits noise, r* bands explode
- 0.05 is a guess; revisit if alpha_t is suspiciously flat or noisy

Requires obs['indexed_10y']. Must run after trend_growth_equation.
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.nairu.base import set_model_coefficients

SIGMA_A_DEFAULT = 0.05  # logit-RW innovation scale, default


def r_star_tv_alpha_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Blend r* with time-varying alpha on logit-scale RW."""
    if constant is None:
        constant = {}

    if "trend_growth" not in latents:
        raise RuntimeError(
            "r_star_tv_alpha_equation requires trend_growth in latents — "
            "ensure trend_growth_equation runs first.",
        )
    if "indexed_10y" not in obs:
        raise RuntimeError(
            "r_star_tv_alpha_equation requires obs['indexed_10y'].",
        )

    sigma_a_value = float(constant.get("sigma_a", SIGMA_A_DEFAULT))

    n_periods = len(obs["log_gdp"])
    g = latents["trend_growth"]
    indexed = obs["indexed_10y"]

    with model:
        # k (term-premium offset) and sigma_r (i.i.d. noise on r*)
        settings = {
            "k": {"mu": 0.5, "sigma": 0.5, "lower": 0.0},
            "sigma_r": {"sigma": 0.3},
        }
        mc = set_model_coefficients(model, settings, constant)

        if not hasattr(model, "_fixed_constants"):
            model._fixed_constants = {}  # noqa: SLF001
        model._fixed_constants["sigma_a"] = sigma_a_value  # noqa: SLF001

        sigma_a = pt.constant(sigma_a_value)

        # Logit-scale RW on alpha
        if "logit_alpha" in constant:
            logit_alpha = constant["logit_alpha"]
        else:
            logit_alpha_0 = pm.Normal("logit_alpha_0", mu=0.0, sigma=2.0)
            logit_alpha_innov = pm.Normal(
                "logit_alpha_innov", mu=0, sigma=1, shape=n_periods - 1,
            )
            logit_alpha_path = logit_alpha_0 + pt.cumsum(sigma_a * logit_alpha_innov)
            logit_alpha = pm.Deterministic(
                "logit_alpha",
                pt.concatenate([logit_alpha_0.reshape((1,)), logit_alpha_path]),
            )

        alpha_t = pm.Deterministic("alpha_rstar", pm.math.sigmoid(logit_alpha))

        # r* with time-varying alpha
        blend = alpha_t * g + (1 - alpha_t) * (indexed - mc["k"])

        if "r_innovation" in constant:
            r_innovation = constant["r_innovation"]
        else:
            r_innovation_raw = pm.Normal(
                "r_innovation_raw", mu=0, sigma=1, shape=n_periods,
            )
            r_innovation = pm.Deterministic(
                "r_innovation", r_innovation_raw * mc["sigma_r"],
            )

        r_star = pm.Deterministic("r_star", blend + r_innovation)

    latents["alpha_rstar"] = alpha_t
    latents["r_star"] = r_star
    return (
        "r*_t = alpha_t*g_t + (1-alpha_t)*(indexed_10y_t - k) + eps_t,  "
        f"logit_alpha RW with sigma_a = {sigma_a_value:.2f}"
    )
