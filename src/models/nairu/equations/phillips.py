"""Phillips curve equations for inflation and wage growth."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients
from src.utilities.rate_conversion import quarterly


def _price_phillips_likelihood(
    inputs: dict[str, np.ndarray],
    nairu: pm.Distribution,
    mc: dict[str, Any],
    gamma_effective: Any,
) -> None:
    """Shared likelihood for price Phillips curve (regime-switching or single slope)."""
    # Unemployment gap (relative to NAIRU)
    u_gap = (inputs["U"] - nairu) / inputs["U"]

    # Expected quarterly inflation from expectations (compound conversion)
    pi_exp_quarterly = quarterly(inputs["π_exp"])

    mu = pi_exp_quarterly + gamma_effective * u_gap

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


def price_inflation_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Anchor-augmented Phillips Curve for price inflation (single slope).

    Model: π = quarterly(π_exp) + γ × u_gap + controls + ε

    Uses a single Phillips curve slope across all periods.

    Uses inflation anchor (expectations pre-1993, phased to target 1993-1998,
    target thereafter). This means NAIRU is interpreted as the unemployment
    rate needed to achieve the inflation target (post-1998).

    Args:
        inputs: Must contain:
            - "U": Unemployment rate
            - "π": Observed quarterly inflation
            - "π_exp": Annual inflation expectations (from signal extraction model)
            - "Δ4ρm_1": Lagged import price growth (optional control)
            - "ξ_2": GSCPI (COVID supply chain pressure, optional control)
        model: PyMC model context
        nairu: NAIRU latent variable
        constant: Optional fixed values for coefficients

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "gamma_pi": {"mu": -1.5, "sigma": 1.0},
            "epsilon_pi": {"sigma": 0.25},
        }
        if "Δ4ρm_1" in inputs:
            settings["rho_pi"] = {"mu": 0.0, "sigma": 0.1}
        if "ξ_2" in inputs:
            settings["xi_gscpi"] = {"mu": 0.0, "sigma": 0.1}
        mc = set_model_coefficients(model, settings, constant)
        _price_phillips_likelihood(inputs, nairu, mc, gamma_effective=mc["gamma_pi"])


def price_inflation_regime_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Anchor-augmented Phillips Curve for price inflation (regime-switching).

    Model: π = quarterly(π_exp) + γ_regime × u_gap + controls + ε

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
            - "π_exp": Annual inflation expectations (from signal extraction model)
            - "regime_pre_gfc", "regime_gfc", "regime_covid": Regime indicators
            - "Δ4ρm_1": Lagged import price growth (optional control)
            - "ξ_2": GSCPI (COVID supply chain pressure, optional control)
        model: PyMC model context
        nairu: NAIRU latent variable
        constant: Optional fixed values for coefficients

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "gamma_pi_pre_gfc": {"mu": -1.5, "sigma": 1.0},
            "gamma_pi_gfc": {"mu": -0.5, "sigma": 0.5},
            "gamma_pi_covid": {"mu": -2.5, "sigma": 1.0},
            "epsilon_pi": {"sigma": 0.25},
        }
        if "Δ4ρm_1" in inputs:
            settings["rho_pi"] = {"mu": 0.0, "sigma": 0.1}
        if "ξ_2" in inputs:
            settings["xi_gscpi"] = {"mu": 0.0, "sigma": 0.1}
        mc = set_model_coefficients(model, settings, constant)
        gamma_effective = (
            mc["gamma_pi_pre_gfc"] * inputs["regime_pre_gfc"]
            + mc["gamma_pi_gfc"] * inputs["regime_gfc"]
            + mc["gamma_pi_covid"] * inputs["regime_covid"]
        )
        _price_phillips_likelihood(inputs, nairu, mc, gamma_effective=gamma_effective)


def wage_growth_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Wage Phillips curve equation (single slope).

    Model: Δulc = α + γ × u_gap + λ × ΔU/U + ε

    Models wage growth as a function of:
    - Unemployment gap (level effect, single slope across all periods)
    - Change in unemployment (speed limit effect)

    Args:
        inputs: Must contain:
            - "U": Unemployment rate
            - "Δulc": Unit labor cost growth (observed wage variable)
            - "ΔU_1_over_U": Lagged unemployment change over unemployment level
        model: PyMC model context
        nairu: NAIRU latent variable
        constant: Optional fixed values for coefficients

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "alpha_wg": {"mu": 0, "sigma": 1.0},
            "gamma_wg": {"mu": -1.5, "sigma": 1.0, "upper": 0},
            "lambda_wg": {"mu": -4.0, "sigma": 2.0},
            "epsilon_wg": {"sigma": 1.0},
        }
        mc = set_model_coefficients(model, settings, constant)

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


def wage_growth_regime_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Wage Phillips curve equation with regime-switching.

    Model: Δulc = α + γ_regime × u_gap + λ × ΔU/U + ε

    Models wage growth as a function of:
    - Unemployment gap (level effect) with regime-specific slopes
    - Change in unemployment (speed limit effect)

    Uses regime-specific Phillips curve slopes (3 regimes, parallel to price equation):
    - gamma_wg_pre_gfc: Pre-GFC (up to 2007Q4) - moderate slope
    - gamma_wg_gfc: Post-GFC (2008Q1 - 2020Q4) - flat
    - gamma_wg_covid: Post-COVID (2021Q1+) - steep

    Args:
        inputs: Must contain:
            - "U": Unemployment rate
            - "Δulc": Unit labor cost growth (observed wage variable)
            - "ΔU_1_over_U": Lagged unemployment change over unemployment level
            - "regime_pre_gfc", "regime_gfc", "regime_covid": Regime indicators
        model: PyMC model context
        nairu: NAIRU latent variable
        constant: Optional fixed values for coefficients

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "alpha_wg": {"mu": 0, "sigma": 1.0},
            "gamma_wg_pre_gfc": {"mu": -1.5, "sigma": 1.0, "upper": 0},
            "gamma_wg_gfc": {"mu": -0.5, "sigma": 0.5, "upper": 0},
            "gamma_wg_covid": {"mu": -1.5, "sigma": 0.75, "upper": 0},
            "lambda_wg": {"mu": -4.0, "sigma": 2.0},
            "epsilon_wg": {"sigma": 1.0},
        }
        mc = set_model_coefficients(model, settings, constant)

        u_gap = (inputs["U"] - nairu) / inputs["U"]

        gamma_effective = (
            mc["gamma_wg_pre_gfc"] * inputs["regime_pre_gfc"]
            + mc["gamma_wg_gfc"] * inputs["regime_gfc"]
            + mc["gamma_wg_covid"] * inputs["regime_covid"]
        )

        pm.Normal(
            "observed_wage_growth",
            mu=(
                mc["alpha_wg"]
                + gamma_effective * u_gap
                + mc["lambda_wg"] * inputs["ΔU_1_over_U"]
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
    """Hourly compensation Phillips curve equation (single slope).

    Model: Δhcoe = α + γ × u_gap + λ × ΔU/U + ψ × MFP + ε

    Models hourly compensation growth (COE/hours) as a function of:
    - Unemployment gap (level effect, single slope across all periods)
    - Change in unemployment (speed limit effect)
    - MFP growth (productivity → wage channel)

    Args:
        inputs: Must contain:
            - "U": Unemployment rate
            - "Δhcoe": Hourly COE growth (observed wage variable)
            - "ΔU_1_over_U": Lagged unemployment change over unemployment level
            - "mfp_growth": MFP trend growth
        model: PyMC model context
        nairu: NAIRU latent variable
        constant: Optional fixed values for coefficients

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "alpha_hcoe": {"mu": 0, "sigma": 0.5},
            "gamma_hcoe": {"mu": -1.0, "sigma": 0.75, "upper": 0},
            "lambda_hcoe": {"mu": -4.0, "sigma": 2.0},
            "psi_hcoe": {"mu": 1.0, "sigma": 0.5, "lower": 0},
            "epsilon_hcoe": {"sigma": 0.75},
        }
        mc = set_model_coefficients(model, settings, constant)

        u_gap = (inputs["U"] - nairu) / inputs["U"]

        pm.Normal(
            "observed_hourly_coe",
            mu=(
                mc["alpha_hcoe"]
                + mc["gamma_hcoe"] * u_gap
                + mc["lambda_hcoe"] * inputs["ΔU_1_over_U"]
                + mc["psi_hcoe"] * inputs["mfp_growth"]
            ),
            sigma=mc["epsilon_hcoe"],
            observed=inputs["Δhcoe"],
        )


def hourly_coe_regime_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Hourly compensation Phillips curve equation with regime-switching.

    Model: Δhcoe = α + γ_regime × u_gap + λ × ΔU/U + ψ × MFP + ε

    Models hourly compensation growth (COE/hours) as a function of:
    - Unemployment gap (level effect) with regime-specific slopes
    - Change in unemployment (speed limit effect)
    - MFP growth (productivity → wage channel)

    Uses regime-specific Phillips curve slopes (3 regimes, parallel to ULC equation):
    - gamma_hcoe_pre_gfc: Pre-GFC (up to 2007Q4) - moderate slope
    - gamma_hcoe_gfc: Post-GFC (2008Q1 - 2020Q4) - flat
    - gamma_hcoe_covid: Post-COVID (2021Q1+) - steep

    Args:
        inputs: Must contain:
            - "U": Unemployment rate
            - "Δhcoe": Hourly COE growth (observed wage variable)
            - "ΔU_1_over_U": Lagged unemployment change over unemployment level
            - "mfp_growth": MFP trend growth
            - "regime_pre_gfc", "regime_gfc", "regime_covid": Regime indicators
        model: PyMC model context
        nairu: NAIRU latent variable
        constant: Optional fixed values for coefficients

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "alpha_hcoe": {"mu": 0, "sigma": 0.5},
            "gamma_hcoe_pre_gfc": {"mu": -1.0, "sigma": 0.75, "upper": 0},
            "gamma_hcoe_gfc": {"mu": -0.5, "sigma": 0.5, "upper": 0},
            "gamma_hcoe_covid": {"mu": -1.5, "sigma": 0.75, "upper": 0},
            "lambda_hcoe": {"mu": -4.0, "sigma": 2.0},
            "psi_hcoe": {"mu": 1.0, "sigma": 0.5, "lower": 0},
            "epsilon_hcoe": {"sigma": 0.75},
        }
        mc = set_model_coefficients(model, settings, constant)

        u_gap = (inputs["U"] - nairu) / inputs["U"]

        gamma_effective = (
            mc["gamma_hcoe_pre_gfc"] * inputs["regime_pre_gfc"]
            + mc["gamma_hcoe_gfc"] * inputs["regime_gfc"]
            + mc["gamma_hcoe_covid"] * inputs["regime_covid"]
        )

        pm.Normal(
            "observed_hourly_coe",
            mu=(
                mc["alpha_hcoe"]
                + gamma_effective * u_gap
                + mc["lambda_hcoe"] * inputs["ΔU_1_over_U"]
                + mc["psi_hcoe"] * inputs["mfp_growth"]
            ),
            sigma=mc["epsilon_hcoe"],
            observed=inputs["Δhcoe"],
        )
