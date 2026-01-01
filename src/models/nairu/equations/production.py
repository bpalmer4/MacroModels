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

    The innovation uses a SkewNormal distribution with positive skew (alpha=2),
    making potential output "sticky downwards" - it's easier to move up than down.
    This reflects that potential output typically grows (capital accumulation,
    labor force growth, productivity) and shouldn't easily decline permanently.

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
        # SkewNormal with alpha=2: positive skew makes downward moves harder
        # This makes potential "sticky downwards" - more resistant to decline
        n_periods = len(inputs["log_gdp"])
        innovations = pm.SkewNormal(
            "potential_innovations",
            mu=0,
            sigma=mc["potential_innovation"],
            alpha=-1,  # milder negative skew: ~60% positive draws, more flexibility
            shape=n_periods - 1,
        )

        # Growth rates: g_t = drift_t + ε_t
        growth_rates = drift[1:] + innovations

        # Cumulative: Y*_t = Y*_0 + Σ(g_i)
        cumulative_growth = pt.cumsum(growth_rates)
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[init_value], init_value + cumulative_growth]),
        )

        # Expose growth rates for diagnostics
        pm.Deterministic(
            "potential_growth",
            pt.concatenate([[drift[0]], growth_rates]),
        )

    return potential_output
