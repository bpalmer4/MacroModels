"""Production function equations for potential output.

Two variants:
- Normal innovations with estimated capital share (standard central bank approach)
- SkewNormal innovations with time-varying observed capital share
"""

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
    """Potential output with Cobb-Douglas production function and estimated alpha.

    Model: Y*_t = Y*_{t-1} + α×ΔK + (1-α)×ΔL + ΔMFP + ε

    Capital share (α) is estimated with prior N(0.3, 0.05).
    Innovations are Normal (symmetric).

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP (for initial value)
            - "capital_growth": Quarterly capital stock growth
            - "lf_growth": Quarterly labor force growth
            - "mfp_growth": Multi-factor productivity growth
        model: PyMC model context
        constant: Optional fixed values. Keys:
            - "alpha_capital": Fix capital share
            - "potential_innovation": Fix innovation std dev
            - "initial_potential": Fix initial potential output

    Returns:
        pm.Distribution: Potential output latent variable (log scale)

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "alpha_capital": {"mu": 0.3, "sigma": 0.05},
            "potential_innovation": {"mu": 0.1, "sigma": 0.05},
            "initial_potential": {
                "mu": inputs["log_gdp"][0],
                "sigma": inputs["log_gdp"][0] * 0.1,
            },
        }
        mc = set_model_coefficients(model, settings, constant)

        alpha = mc["alpha_capital"]

        # Cobb-Douglas drift: α×g_K + (1-α)×g_L + g_MFP
        drift = (
            alpha * inputs["capital_growth"]
            + (1 - alpha) * inputs["lf_growth"]
            + inputs["mfp_growth"]
        )

        init_value = mc["initial_potential"]

        # Build potential output as cumulative sum with innovations
        n_periods = len(inputs["log_gdp"])
        innovations = pm.Normal(
            "potential_innovations",
            mu=0,
            sigma=mc["potential_innovation"],
            shape=n_periods - 1,
        )

        # Cumulative: Y*_t = Y*_0 + Σ(drift_i + ε_i)
        cumulative_growth = pt.cumsum(drift[1:] + innovations)
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[init_value], init_value + cumulative_growth]),
        )

        # Expose growth rates for diagnostics
        growth_rates = pt.concatenate([[drift[0]], drift[1:] + innovations])
        pm.Deterministic("potential_growth", growth_rates)

    return potential_output


def potential_output_skewnormal_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    constant: dict[str, Any] | None = None,
) -> pm.Distribution:
    """Potential output with Cobb-Douglas and SkewNormal innovations.

    Model: Y*_t = Y*_{t-1} + α_t×ΔK + (1-α_t)×ΔL + ΔMFP + ε,  ε ~ SkewNormal(α=1)

    Capital share (α) is time-varying, loaded from ABS national accounts.
    SkewNormal innovations with positive skew make potential "sticky downwards".

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

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
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

        # SkewNormal with alpha=+1: ~75% of draws are positive
        # Zero-mean adjustment: shift location so E[ε] = 0
        n_periods = len(inputs["log_gdp"])
        skew_alpha = 1.0
        mean_offset = np.sqrt(2 / np.pi) * skew_alpha / np.sqrt(1 + skew_alpha**2)
        innovations = pm.SkewNormal(
            "potential_innovations",
            mu=-mc["potential_innovation"] * mean_offset,
            sigma=mc["potential_innovation"],
            alpha=skew_alpha,
            shape=n_periods,
        )

        # Growth rates: g_t = drift_t + ε_t
        growth_rates = drift + innovations

        # Cumulative: Y*_t = Y*_0 + Σ(g_i) for i=1..t
        cumulative_growth = pt.cumsum(growth_rates[1:])
        potential_output = pm.Deterministic(
            "potential_output",
            pt.concatenate([[init_value], init_value + cumulative_growth]),
        )

        pm.Deterministic("potential_growth", growth_rates)

    return potential_output
