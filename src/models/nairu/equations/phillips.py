"""Phillips curve equations for inflation and wage growth."""

from typing import Any

import numpy as np
import pymc as pm

from src.utilities.rate_conversion import quarterly
from src.models.nairu.base import set_model_coefficients


def price_inflation_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Anchor-augmented Phillips Curve for price inflation with regime-switching.

    Model: π = quarterly(π_anchor) + γ_regime × u_gap + controls + ε

    Uses regime-specific Phillips curve slopes (3 regimes):
    - gamma_pi_pre_gfc: Pre-GFC (up to 2007Q4) - moderate slope
    - gamma_pi_gfc: Post-GFC (2008Q1 - 2020Q4) - flat
    - gamma_pi_covid: Post-COVID (2021Q1+) - steep

    Uses inflation anchor (expectations pre-1993, phased to target 1993-1998,
    target thereafter). This means NAIRU is interpreted as the unemployment
    rate needed to achieve the inflation target (post-1998).

    Args:
        inputs: Must contain:
            - "U": Unemployment rate
            - "π": Observed quarterly inflation
            - "π_anchor": Annual inflation anchor (expectations → target blend)
            - "regime_pre_gfc", "regime_gfc", "regime_covid": Regime indicators
            - "Δ4ρm_1": Lagged import price growth (optional control)
            - "ξ_2": GSCPI (COVID supply chain pressure, optional control)
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
            # Regime-specific unemployment gap coefficients (3 regimes)
            "gamma_pi_pre_gfc": {"mu": -1.5, "sigma": 1.0},  # pre-GFC (moderate)
            "gamma_pi_gfc": {"mu": -0.5, "sigma": 0.5},      # post-GFC (flat)
            "gamma_pi_covid": {"mu": -2.5, "sigma": 1.0},    # post-COVID (steep)
            "xi_gscpi": {"mu": 0.0, "sigma": 0.1},           # GSCPI supply chain effect
            "epsilon_pi": {"sigma": 0.25},                   # error term
        }
        mc = set_model_coefficients(model, settings, constant)

        # Unemployment gap (relative to NAIRU)
        u_gap = (inputs["U"] - nairu) / inputs["U"]

        # Expected quarterly inflation from anchor (compound conversion)
        pi_anchor_quarterly = quarterly(inputs["π_anchor"])

        # Phillips curve with regime-specific slopes
        # γ_effective = Σ γ_regime × regime_indicator
        gamma_effective = (
            mc["gamma_pi_pre_gfc"] * inputs["regime_pre_gfc"]
            + mc["gamma_pi_gfc"] * inputs["regime_gfc"]
            + mc["gamma_pi_covid"] * inputs["regime_covid"]
        )

        mu = pi_anchor_quarterly + gamma_effective * u_gap

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
    """Wage Phillips curve equation with regime-switching (RBA-style specification).

    Models wage growth as a function of:
    - Unemployment gap (level effect) with regime-specific slopes
    - Change in unemployment (speed limit effect)
    - Demand deflator (price→wage channel)
    - Trend inflation expectations (anchoring)

    Note: Persistence term (rho_wg) was tested but posterior not different from zero;
    removed for parsimony.

    Uses regime-specific Phillips curve slopes (3 regimes, parallel to price equation):
    - gamma_wg_pre_gfc: Pre-GFC (up to 2007Q4) - moderate slope
    - gamma_wg_gfc: Post-GFC (2008Q1 - 2020Q4) - flat
    - gamma_wg_covid: Post-COVID (2021Q1+) - steep

    Args:
        inputs: Must contain:
            - "U": Unemployment rate
            - "Δulc": Unit labor cost growth (observed wage variable)
            - "ΔU_1_over_U": Lagged unemployment change over unemployment level
            - "Δ4dfd": DFD deflator growth (year-ended)
            - "π_anchor": Trend inflation expectations
            - "regime_pre_gfc", "regime_gfc", "regime_covid": Regime indicators
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
            # Regime-specific unemployment gap coefficients (3 regimes, all negative)
            "gamma_wg_pre_gfc": {"mu": -1.0, "sigma": 0.75, "upper": 0},  # pre-GFC (moderate)
            "gamma_wg_gfc": {"mu": -0.5, "sigma": 0.5, "upper": 0},       # post-GFC (flat)
            "gamma_wg_covid": {"mu": -1.5, "sigma": 0.75, "upper": 0},    # post-COVID (steep)
            "lambda_wg": {"mu": -4.0, "sigma": 2.0},   # UE rate change (speed limit)
            "phi_wg": {"mu": 0.1, "sigma": 0.1},       # demand deflator (price→wage)
            "theta_wg": {"mu": 0.1, "sigma": 0.1, "lower": 0},  # trend expectations (positive by theory)
            "epsilon_wg": {"sigma": 1.0},             # error term
        }
        mc = set_model_coefficients(model, settings, constant)

        # Unemployment gap
        u_gap = (inputs["U"] - nairu) / inputs["U"]

        # Wage Phillips curve with regime-specific slopes
        # γ_effective = Σ γ_regime × regime_indicator
        gamma_effective = (
            mc["gamma_wg_pre_gfc"] * inputs["regime_pre_gfc"]
            + mc["gamma_wg_gfc"] * inputs["regime_gfc"]
            + mc["gamma_wg_covid"] * inputs["regime_covid"]
        )

        # Convert annual π_anchor to quarterly for consistency with ULC growth
        pi_anchor_quarterly = quarterly(inputs["π_anchor"])

        pm.Normal(
            "observed_wage_growth",
            mu=(
                mc["alpha_wg"]
                + gamma_effective * u_gap                   # unemployment gap
                + mc["lambda_wg"] * inputs["ΔU_1_over_U"]   # speed limit
                + mc["phi_wg"] * inputs["Δ4dfd"]            # demand deflator
                + mc["theta_wg"] * pi_anchor_quarterly      # trend expectations
            ),
            sigma=mc["epsilon_wg"],
            observed=inputs["Δulc"],
        )


def hourly_coe_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Hourly compensation Phillips curve equation with regime-switching.

    Models hourly compensation growth (COE/hours) as a function of:
    - Unemployment gap (level effect) with regime-specific slopes
    - Change in unemployment (speed limit effect)
    - Demand deflator (price→wage channel)
    - Trend inflation expectations (anchoring)
    - MFP growth (productivity → wage channel)

    Unlike ULC (which divides by output), HCOE divides by hours, so productivity
    gains should flow through to higher hourly pay. The MFP term captures this:
    higher productivity growth → higher hourly compensation (positive coefficient).

    Note: Persistence term (rho_hcoe) was tested but posterior not different from zero;
    removed for parsimony.

    Uses regime-specific Phillips curve slopes (3 regimes, parallel to ULC equation):
    - gamma_hcoe_pre_gfc: Pre-GFC (up to 2007Q4) - moderate slope
    - gamma_hcoe_gfc: Post-GFC (2008Q1 - 2020Q4) - flat
    - gamma_hcoe_covid: Post-COVID (2021Q1+) - steep

    Args:
        inputs: Must contain:
            - "U": Unemployment rate
            - "Δhcoe": Hourly COE growth (observed wage variable)
            - "ΔU_1_over_U": Lagged unemployment change over unemployment level
            - "Δ4dfd": DFD deflator growth (year-ended)
            - "π_anchor": Trend inflation expectations
            - "mfp_growth": MFP trend growth (HP-filtered)
            - "regime_pre_gfc", "regime_gfc", "regime_covid": Regime indicators
        model: PyMC model context
        nairu: NAIRU latent variable
        constant: Optional fixed values for coefficients

    Example:
        hourly_coe_equation(inputs, model, nairu)

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "alpha_hcoe": {"mu": 0, "sigma": 0.5},       # intercept
            # Regime-specific unemployment gap coefficients (3 regimes, all negative)
            "gamma_hcoe_pre_gfc": {"mu": -1.0, "sigma": 0.75, "upper": 0},  # pre-GFC
            "gamma_hcoe_gfc": {"mu": -0.5, "sigma": 0.5, "upper": 0},       # post-GFC (flat)
            "gamma_hcoe_covid": {"mu": -1.5, "sigma": 0.75, "upper": 0},    # post-COVID
            "lambda_hcoe": {"mu": -4.0, "sigma": 2.0},   # UE rate change (speed limit)
            "phi_hcoe": {"mu": 0.1, "sigma": 0.1},       # demand deflator (price→wage)
            "theta_hcoe": {"mu": 0.1, "sigma": 0.1, "lower": 0},  # trend expectations (positive by theory)
            "psi_hcoe": {"mu": 1.0, "sigma": 0.5, "lower": 0},    # MFP → wages (positive: productivity gains shared)
            "epsilon_hcoe": {"sigma": 0.75},            # error term (tighter than ULC)
        }
        mc = set_model_coefficients(model, settings, constant)

        # Unemployment gap
        u_gap = (inputs["U"] - nairu) / inputs["U"]

        # Regime-specific slopes
        gamma_effective = (
            mc["gamma_hcoe_pre_gfc"] * inputs["regime_pre_gfc"]
            + mc["gamma_hcoe_gfc"] * inputs["regime_gfc"]
            + mc["gamma_hcoe_covid"] * inputs["regime_covid"]
        )

        # Convert annual π_anchor to quarterly for consistency
        pi_anchor_quarterly = quarterly(inputs["π_anchor"])

        pm.Normal(
            "observed_hourly_coe",
            mu=(
                mc["alpha_hcoe"]
                + gamma_effective * u_gap                      # unemployment gap
                + mc["lambda_hcoe"] * inputs["ΔU_1_over_U"]    # speed limit
                + mc["phi_hcoe"] * inputs["Δ4dfd"]             # demand deflator
                + mc["theta_hcoe"] * pi_anchor_quarterly       # trend expectations
                + mc["psi_hcoe"] * inputs["mfp_growth"]        # productivity → wages
            ),
            sigma=mc["epsilon_hcoe"],
            observed=inputs["Δhcoe"],
        )
