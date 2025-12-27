"""Net exports equation linking trade balance to domestic demand and exchange rates."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def net_exports_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    potential_output: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Net exports equation - trade balance dynamics.

    Models the trade balance response to:
    - Domestic demand (output gap) - imports rise when economy is strong
    - Exchange rate - appreciation worsens trade balance

    Equation:
        Δ(NX/Y) = β₁×output_gap + β₂×Δtwi + ε

    Where:
        - Δ(NX/Y) = change in net exports ratio to GDP (pp)
        - output_gap = log_gdp - potential_output (domestic demand pressure)
        - Δtwi = change in TWI (appreciation is positive)
        - β₁ < 0: positive output gap → more imports → lower NX
        - β₂ < 0: appreciation (positive Δtwi) → worse NX

    This links the exchange rate equation to real trade outcomes.

    Note:
        An intercept term (α) was tested but found to be statistically
        indistinguishable from zero. This is economically sensible: net exports
        changes should not exhibit persistent drift unrelated to the output gap
        and exchange rate. The zero-intercept specification is preferred.

        A persistence term (ρ×Δ(NX/Y)_{t-1}) was also tested but found to be
        near-zero, indicating dynamics are already captured by the persistent
        output gap. The simplified specification without persistence is preferred.

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP
            - "Δnx_ratio": Change in net exports / GDP ratio (pp)
            - "Δtwi": Change in TWI (quarterly)
        model: PyMC model context
        potential_output: Potential output latent variable
        constant: Optional fixed values for coefficients

    Example:
        net_exports_equation(inputs, model, potential_output)

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            # Domestic demand effect: positive gap → more imports → worse NX
            "beta_nx_ygap": {"mu": -0.05, "sigma": 0.05, "upper": 0},
            # Exchange rate effect: appreciation → worse NX
            "beta_nx_twi": {"mu": -0.02, "sigma": 0.02, "upper": 0},
            # Error term
            "epsilon_nx": {"sigma": 0.3},
        }
        mc = set_model_coefficients(model, settings, constant)

        # Output gap: (Y - Y*) in log terms (full length, aligned with obs_index)
        output_gap = inputs["log_gdp"] - potential_output

        # All inputs are pre-aligned
        dnx_ratio = inputs["Δnx_ratio"]
        dtwi = inputs["Δtwi"]

        # Net exports equation
        predicted_dnx = mc["beta_nx_ygap"] * output_gap + mc["beta_nx_twi"] * dtwi

        pm.Normal(
            "observed_net_exports",
            mu=predicted_dnx,
            sigma=mc["epsilon_nx"],
            observed=dnx_ratio,
        )
