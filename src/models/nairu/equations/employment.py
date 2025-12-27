"""Employment equation linking labour demand to output and real wages."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def employment_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    potential_output: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Employment equation - labour demand based on output and real wages.

    Firms hire based on output and real labour costs relative to productivity.
    This provides a structural explanation for labour market dynamics, complementing
    Okun's Law with a wage channel.

    Equation:
        Δle = α + β₁×output_gap + β₂×(Δulc - Δmfp) + ε

    Where:
        - Δle = employment growth (log difference, %)
        - output_gap = log_gdp - potential_output (positive when Y > Y*)
        - (Δulc - Δmfp) = real wage gap: unit labour costs growing faster than productivity
        - β₁ > 0: positive output gap raises employment
        - β₂ < 0: higher real wages reduce employment (firms cut hiring)

    This connects wage Phillips curves back to labour demand:
    If wages rise faster than productivity → firms hire less → labour market loosens

    Note:
        A persistence term (ρ×Δle_{t-1}) was tested but found to be negative,
        indicating dynamics already captured by the persistent output gap.
        The simplified specification without persistence is preferred.

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP
            - "emp_growth": Employment growth (quarterly, %)
            - "real_wage_gap": Δulc - Δmfp (unit labour cost growth minus MFP growth)
        model: PyMC model context
        potential_output: Potential output latent variable
        constant: Optional fixed values for coefficients

    Example:
        employment_equation(inputs, model, potential_output)

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            # Intercept (trend employment growth)
            "alpha_emp": {"mu": 0.3, "sigma": 0.2},
            # Output gap effect: positive gap → more hiring
            "beta_emp_ygap": {"mu": 0.15, "sigma": 0.1, "lower": 0},
            # Real wage gap effect: higher real wages → less hiring
            "beta_emp_wage": {"mu": -0.1, "sigma": 0.1, "upper": 0},
            # Error term
            "epsilon_emp": {"sigma": 0.4},
        }
        mc = set_model_coefficients(model, settings, constant)

        # Output gap: (Y - Y*) in log terms (full length, aligned with obs_index)
        output_gap = inputs["log_gdp"] - potential_output

        # All inputs are pre-aligned
        emp_growth = inputs["emp_growth"]
        real_wage_gap = inputs["real_wage_gap"]

        # Employment equation
        predicted_emp_growth = (
            mc["alpha_emp"]
            + mc["beta_emp_ygap"] * output_gap
            + mc["beta_emp_wage"] * real_wage_gap
        )

        pm.Normal(
            "observed_employment",
            mu=predicted_emp_growth,
            sigma=mc["epsilon_emp"],
            observed=emp_growth,
        )
