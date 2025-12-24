"""Phillips curve equations for inflation and wage growth."""

from typing import Any

import numpy as np
import pymc as pm

from src.analysis.rate_conversion import quarterly
from src.models.base import set_model_coefficients


def price_inflation_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Anchor-augmented Phillips Curve for price inflation.

    Model: π = quarterly(π_anchor) + γ × u_gap + controls + ε

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
            - "ξ_2": GSCPI (COVID supply chain pressure, optional control)
        model: PyMC model context
        nairu: NAIRU latent variable
        constant: Optional fixed values for coefficients

    Note:
        Oil and coal price effects were tested but found to be statistically
        indistinguishable from zero. Their effects on Australian inflation are
        already captured through the import price channel (Δ4ρm).
        GSCPI captures COVID-era supply chain disruptions (applied non-linearly).

    Example:
        price_inflation_equation(inputs, model, nairu)

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "rho_pi": {"mu": 0.0, "sigma": 0.1},      # import prices pass-through
            "gamma_pi": {"mu": -0.5, "sigma": 0.3},   # unemployment gap
            "xi_gscpi": {"mu": 0.0, "sigma": 0.1},    # GSCPI supply chain effect
            "epsilon_pi": {"sigma": 0.25},            # error term
        }
        mc = set_model_coefficients(model, settings, constant)

        # Unemployment gap (relative to NAIRU)
        u_gap = (inputs["U"] - nairu) / inputs["U"]

        # Expected quarterly inflation from anchor (compound conversion)
        pi_anchor_quarterly = quarterly(inputs["π_anchor"])

        # Phillips curve: π = quarterly(π_anchor) + γ × u_gap + controls
        mu = pi_anchor_quarterly + mc["gamma_pi"] * u_gap

        # Add optional controls if present
        if "Δ4ρm_1" in inputs:
            mu = mu + mc["rho_pi"] * inputs["Δ4ρm_1"]
        if "ξ_2" in inputs:
            # GSCPI: non-linear effect (squared with sign preservation)
            # Captures asymmetric supply chain pressure during COVID
            mu = mu + mc["xi_gscpi"] * inputs["ξ_2"] ** 2 * np.sign(inputs["ξ_2"])

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
