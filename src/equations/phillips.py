"""Phillips curve equations for inflation and wage growth."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.base import set_model_coefficients


def price_inflation_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Anchor-augmented Phillips Curve for price inflation.

    Model: π = π_anchor/4 + γ × u_gap + controls + ε

    Uses inflation anchor (expectations pre-1993, phased to target 1993-1998,
    target thereafter). This means NAIRU is interpreted as the unemployment
    rate needed to achieve the inflation target (post-1998).

    The inflation anchor (π_anchor) is annual and converted to quarterly.

    Args:
        inputs: Must contain:
            - "U": Unemployment rate
            - "π": Observed quarterly inflation
            - "π_anchor": Annual inflation anchor (expectations → target blend)
            - "Δ4ρm_1": Lagged import price growth (optional control)
            - "ξ_2": COVID indicator (optional control)
        model: PyMC model context
        nairu: NAIRU latent variable
        constant: Optional fixed values for coefficients

    Example:
        price_inflation_equation(inputs, model, nairu)
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "rho_pi": {"mu": 0.0, "sigma": 0.1},      # import prices pass-through
            "gamma_pi": {"mu": -0.5, "sigma": 0.3},   # unemployment gap
            "xi_2sq_pi": {"mu": 0.0, "sigma": 0.1},   # COVID disruptions
            "epsilon_pi": {"sigma": 0.25},            # error term
        }
        mc = set_model_coefficients(model, settings, constant)

        # Unemployment gap (relative to NAIRU)
        u_gap = (inputs["U"] - nairu) / inputs["U"]

        # Expected quarterly inflation from anchor
        pi_anchor_quarterly = inputs["π_anchor"] / 4

        # Phillips curve: π = π_anchor/4 + γ × u_gap + controls
        mu = pi_anchor_quarterly + mc["gamma_pi"] * u_gap

        # Add optional controls if present
        if "Δ4ρm_1" in inputs:
            mu = mu + mc["rho_pi"] * inputs["Δ4ρm_1"]
        if "ξ_2" in inputs:
            mu = mu + mc["xi_2sq_pi"] * inputs["ξ_2"] ** 2 * np.sign(inputs["ξ_2"])

        pm.Normal(
            "observed_price_inflation",
            mu=mu,
            sigma=mc["epsilon_pi"],
            observed=inputs["π"],
        )


def wage_growth_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Wage Phillips curve equation.

    Models wage growth as a function of:
    - Unemployment gap (level effect)
    - Change in unemployment (speed limit effect)

    The unemployment gap coefficient (gamma_wg) is constrained to be negative
    using a truncated normal prior.

    Args:
        inputs: Must contain:
            - "U": Unemployment rate
            - "Δulc": Unit labor cost growth (observed wage variable)
            - "ΔU_1_over_U": Lagged unemployment change over unemployment level
        model: PyMC model context
        nairu: NAIRU latent variable
        constant: Optional fixed values for coefficients

    Example:
        wage_growth_equation(inputs, model, nairu)
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "alpha_wg": {"mu": 0, "sigma": 1.0},       # intercept
            "gamma_wg": {"mu": -1.5, "sigma": 1.0, "upper": 0},  # U-gap slope (negative)
            "lambda_wg": {"mu": -4.0, "sigma": 2.0},   # UE rate change (speed limit)
            "epsilon_wg": {"sigma": 1.0},             # error term
        }
        mc = set_model_coefficients(model, settings, constant)

        # Unemployment gap
        u_gap = (inputs["U"] - nairu) / inputs["U"]

        pm.Normal(
            "observed_wage_growth",
            mu=(
                mc["alpha_wg"]
                + mc["gamma_wg"] * u_gap
                + mc["lambda_wg"] * inputs["ΔU_1_over_U"]
            ),
            sigma=mc["epsilon_wg"],
            observed=inputs["Δulc"],
        )
