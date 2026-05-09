"""r-star as the blend plus a persistent z deviation (Resolution E).

    r*_t   = alpha * g_t + (1 - alpha) * (indexed_10y_t - k) + z_t
    z_t    = rho_z * z_{t-1} + sigma_z * eps_z_raw,   eps_z_raw ~ N(0, 1)
    z_0    ~ N(0, sigma_z / sqrt(1 - rho_z^2))                  [stationary init]

Generalises both Resolution A and Resolution C:
- alpha = 1, sigma_z free  -> r* = g + z (canonical HLW = Resolution A)
- z = i.i.d. eps with sigma_z small -> r* = blend + eps (Resolution C)

The point: keep C's blend as the "structural anchor" half of HLW (replacing
g), and add z back as the "what the IS curve adds" half. The z posterior is
the directly interpretable readout of how much the IS curve is moving r*
above and beyond the blend.

Design decisions:

- ``sigma_z`` is **fixed** at 0.15 pp/quarter (not estimated). Rationale
  identical to iteration 9's lesson and the soft-anchor sigma in Resolution
  E (proposed alternative): when sigma_z is the only constraint keeping z
  alive against an underconstrained IS curve, the data collapses it to ~0
  and r* reverts to the blend. Fixing keeps z genuinely alive. Value is in
  the canonical HLW lambda_z neighbourhood (Lewis-Vazquez-Grande 2019, RBA
  estimates).

- ``rho_z`` is **estimated** with a tight prior centred at 0.95
  (TruncatedNormal(0.95, 0.03, [0, 1])). Mean-reversion to 0 over ~20
  quarters; long-run r* ~= blend. The tight prior keeps the AR away from
  the unit root (where geometry gets ugly) without committing to a specific
  persistence.

- z is **non-centred** (sample raw N(0, 1) and scale by sigma_z, then apply
  AR recursion). With sigma_z fixed this is less critical than for free
  sigma — there's no Neal's funnel — but it costs nothing and is a robust
  default.

- Initial z_0 is set from the **stationary distribution** rather than 0,
  so the AR has a sensible variance at t = 0 rather than being artificially
  pinned.

- alpha ~ Beta(2, 2) and k ~ TruncatedNormal(0.5, 0.5, lower=0): identical
  to Resolution C — same structural-vs-market anchor weighting.

Requires obs['indexed_10y']. Must run after trend_growth_equation.
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from src.models.nairu.base import set_model_coefficients

SIGMA_Z_DEFAULT = 0.15  # AR(1) innovation scale, pp/quarter (Buncic-tested via sweep)


def r_star_blended_z_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """r* = blend + z, with blend = alpha*g + (1-alpha)*(indexed - k) and z AR(1).

    σ_z is fixed (not estimated) for the reasons in the module docstring.
    Override the default via ``constant['sigma_z']`` for prior-sensitivity
    sweeps.
    """
    if constant is None:
        constant = {}
    sigma_z_value = float(constant.get("sigma_z", SIGMA_Z_DEFAULT))

    if "trend_growth" not in latents:
        raise RuntimeError(
            "r_star_blended_z_equation requires trend_growth in latents — "
            "ensure trend_growth_equation runs first.",
        )
    if "indexed_10y" not in obs:
        raise RuntimeError(
            "r_star_blended_z_equation requires obs['indexed_10y'].",
        )

    n_periods = len(obs["log_gdp"])
    g = latents["trend_growth"]
    indexed = obs["indexed_10y"]

    with model:
        # alpha: weight on structural anchor. Default Beta(1, 1) = Uniform —
        # uninformative; matches the new C default. Override with
        # constant['alpha_prior'] = (a, b) for prior-sensitivity sweeps.
        if "alpha_rstar" in constant:
            alpha = constant["alpha_rstar"]
            if not hasattr(model, "_fixed_constants"):
                model._fixed_constants = {}  # noqa: SLF001
            model._fixed_constants["alpha_rstar"] = alpha  # noqa: SLF001
        else:
            a_param, b_param = constant.get("alpha_prior", (1.0, 1.0))
            alpha = pm.Beta(
                "alpha_rstar", alpha=float(a_param), beta=float(b_param),
            )
            if not hasattr(model, "_fixed_constants"):
                model._fixed_constants = {}  # noqa: SLF001
            model._fixed_constants["alpha_prior_a"] = float(a_param)  # noqa: SLF001
            model._fixed_constants["alpha_prior_b"] = float(b_param)  # noqa: SLF001

        # k (term-premium offset) and rho_z (z's AR persistence)
        settings = {
            "k": {"mu": 0.5, "sigma": 0.5, "lower": 0.0},
            "rho_z": {"mu": 0.95, "sigma": 0.03, "lower": 0.0, "upper": 1.0},
        }
        mc = set_model_coefficients(model, settings, constant)

        # Track sigma_z as a fixed constant (not estimated).
        if not hasattr(model, "_fixed_constants"):
            model._fixed_constants = {}  # noqa: SLF001
        model._fixed_constants["sigma_z"] = sigma_z_value  # noqa: SLF001

        rho_z = mc["rho_z"]
        sigma_z = pt.constant(sigma_z_value)

        if "z_star" in constant:
            z_star = constant["z_star"]
        else:
            # Non-centred AR(1): z_raw ~ N(0,1), then scale & recurse.
            z_raw = pm.Normal("z_raw", mu=0, sigma=1, shape=n_periods)

            # Stationary initial value: z_0 ~ N(0, sigma_z / sqrt(1 - rho^2))
            z_0 = z_raw[0] * sigma_z / pt.sqrt(1 - rho_z ** 2)

            def _ar1_step(eps_t, z_prev, rho, sig):
                return rho * z_prev + sig * eps_t

            z_rest, _ = pytensor.scan(
                fn=_ar1_step,
                sequences=z_raw[1:],
                outputs_info=z_0,
                non_sequences=[rho_z, sigma_z],
            )

            z_star = pm.Deterministic(
                "z_star",
                pt.concatenate([z_0.reshape((1,)), z_rest]),
            )

        blend = alpha * g + (1 - alpha) * (indexed - mc["k"])
        r_star = pm.Deterministic("r_star", blend + z_star)

    latents["z_star"] = z_star
    latents["r_star"] = r_star
    return (
        "r*_t = alpha*g_t + (1-alpha)*(indexed_10y_t - k) + z_t,  "
        f"z AR(1): rho ~ TN(0.95,0.03,[0,1]), sigma_z = {sigma_z_value:.2f}"
    )
