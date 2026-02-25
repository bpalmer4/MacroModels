"""Okun's Law equations linking output gap to unemployment.

Two forms available:
- Simple change form: ΔU = β × output_gap + ε
- Gap-to-gap with persistence: U_gap = τ₂ × U_gap_{-1} + τ₁ × Y_gap + ε
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.nairu.base import set_model_coefficients


def okun_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    potential_output: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Okun's Law (simple change form).

    Model: ΔU = β × output_gap + ε

    Where:
        - ΔU is the observed quarterly change in unemployment rate (pp)
        - output_gap = (log_gdp - potential_output)
        - β is negative: when Y > Y* (positive gap), unemployment falls

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP
            - "ΔU": Change in unemployment rate
        model: PyMC model context
        nairu: NAIRU latent variable (not directly used, keeps interface consistent)
        potential_output: Potential output latent variable
        constant: Optional fixed values for coefficients

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "beta_okun": {"mu": -0.2, "sigma": 0.15},
            "epsilon_okun": {"sigma": 0.5},
        }
        mc = set_model_coefficients(model, settings, constant)

        # Output gap: (Y - Y*) in log terms
        output_gap = inputs["log_gdp"] - potential_output

        # Okun's Law: ΔU = β × output_gap
        pm.Normal(
            "okun_law",
            mu=mc["beta_okun"] * output_gap,
            sigma=mc["epsilon_okun"],
            observed=inputs["ΔU"],
        )


def okun_gap_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    potential_output: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Okun's Law (gap-to-gap with persistence).

    Model: U_gap = τ₂ × U_gap_{-1} + τ₁ × Y_gap + ε

    Where:
        - U_gap = U - NAIRU (unemployment gap, pp)
        - Y_gap = log_gdp - potential_output (output gap, log %)
        - τ₁ < 0: positive output gap → negative unemployment gap (tight labour market)
        - τ₂ ∈ (0,1): persistence in unemployment gap

    This ties the unemployment gap level to the output gap level,
    providing stronger identification of NAIRU than the simple ΔU form.

    Implemented on observed U[t] (since U_gap involves latent NAIRU):
        U[t] = NAIRU[t] + τ₂ × (U[t-1] - NAIRU[t-1]) + τ₁ × Y_gap[t] + ε

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP
            - "U": Unemployment rate
        model: PyMC model context
        nairu: NAIRU latent variable
        potential_output: Potential output latent variable
        constant: Optional fixed values for coefficients

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "tau1_okun": {"mu": -0.18, "sigma": 0.1, "upper": 0},
            "tau2_okun": {"mu": 0.75, "sigma": 0.15, "lower": 0, "upper": 1},
            "epsilon_okun": {"sigma": 0.3},
        }
        mc = set_model_coefficients(model, settings, constant)

        # Output gap: (Y - Y*) in log terms
        output_gap = inputs["log_gdp"] - potential_output

        # Lagged NAIRU (pad first value)
        nairu_lag1 = pt.concatenate([[nairu[0]], nairu[:-1]])

        # Lagged unemployment gap: U[t-1] - NAIRU[t-1]
        U_lag1 = np.concatenate([[inputs["U"][0]], inputs["U"][:-1]])
        u_gap_lag1 = U_lag1 - nairu_lag1

        # Predicted U[t] = NAIRU[t] + τ₂ × u_gap[t-1] + τ₁ × y_gap[t]
        predicted_U = nairu + mc["tau2_okun"] * u_gap_lag1 + mc["tau1_okun"] * output_gap

        pm.Normal(
            "okun",
            mu=predicted_U,
            sigma=mc["epsilon_okun"],
            observed=inputs["U"],
        )
