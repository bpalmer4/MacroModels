"""Production function equations for potential output."""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.nairu.base import set_model_coefficients


def potential_output_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    constant: dict[str, Any] | None = None,
) -> pm.Distribution:
    """Potential output with Cobb-Douglas production function.

    Potential growth equals Cobb-Douglas drift plus small innovation:
        g_potential_t = drift_t + ε_t

    Where drift is from Cobb-Douglas:
        drift_t = α_t × capital_growth_t + (1-α_t) × lf_growth_t + mfp_growth_t

    Capital share (α) is time-varying, loaded from ABS national accounts:
        α = GOS / (GOS + COE)

    With tight innovation variance, potential is driven almost entirely by
    supply-side fundamentals (capital, labor, MFP).

    The innovation uses a zero-mean SkewNormal distribution with positive skew (α=1).
    The location is shifted negative to ensure E[ε]=0 while ~75% of draws are positive.
    This makes potential "sticky downwards" — growth is more likely than decline,
    reflecting that capital accumulation and productivity gains are hard to reverse.

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP (for initial value)
            - "capital_growth": Quarterly capital stock growth
            - "lf_growth": Quarterly labor force growth
            - "mfp_growth": Multi-factor productivity growth
            - "alpha_capital": Time-varying capital share from national accounts
        model: PyMC model context
        constant: Optional fixed values. Keys:
            - "potential_innovation": Fix innovation std dev
            - "initial_potential": Fix initial potential output

    Returns:
        pm.Distribution: Potential output latent variable (log scale)

    Example:
        potential = potential_output_equation(inputs, model)

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            # Innovation allows potential to absorb some high-frequency noise
            # while staying smooth through recessions
            "potential_innovation": {"mu": 0.05, "sigma": 0.02},
            "initial_potential": {
                "mu": inputs["log_gdp"][0],
                "sigma": inputs["log_gdp"][0] * 0.1,
            },
        }
        mc = set_model_coefficients(model, settings, constant)

        # Time-varying capital share from national accounts
        alpha = inputs["alpha_capital"]

        # Cobb-Douglas drift: α_t×g_K + (1-α_t)×g_L + g_MFP
        drift = (
            alpha * inputs["capital_growth"]
            + (1 - alpha) * inputs["lf_growth"]
            + inputs["mfp_growth"]
        )

        init_value = mc["initial_potential"]

        # Build potential output: growth = drift + innovation
        # SkewNormal with alpha=+1: ~75% of draws are positive (growth more common)
        # Zero-mean adjustment: shift location so E[ε] = 0 despite positive skew
        # E[SkewNormal(μ, σ, α)] = μ + σ × δ × sqrt(2/π) where δ = α/sqrt(1+α²)
        # For α=1: mean_shift = σ × 0.5642, so set μ = -σ × 0.5642
        n_periods = len(inputs["log_gdp"])
        skew_alpha = 1.0
        mean_offset = np.sqrt(2 / np.pi) * skew_alpha / np.sqrt(1 + skew_alpha**2)  # ≈ 0.5642
        innovations = pm.SkewNormal(
            "potential_innovations",
            mu=-mc["potential_innovation"] * mean_offset,  # shift to zero mean
            sigma=mc["potential_innovation"],
            alpha=skew_alpha,  # positive skew: ~75% positive draws
            shape=n_periods,
        )

        # Growth rates: g_t = drift_t + ε_t (all periods have innovation)
        growth_rates = drift + innovations

        # Cumulative: Y*_t = Y*_0 + Σ(g_i) for i=1..t
        # Note: growth_rates[0] is for t=0 (not used in potential_output, but exposed)
        # growth_rates[1:] drives potential from t=1 onwards
        cumulative_growth = pt.cumsum(growth_rates[1:])
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[init_value], init_value + cumulative_growth]),
        )

        # Expose growth rates for diagnostics (all periods now have variance)
        pm.Deterministic(
            "potential_growth",
            growth_rates,
        )

    return potential_output
