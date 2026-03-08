"""Price Phillips curve equations (single slope and regime-switching)."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients
from src.utilities.rate_conversion import quarterly


def _price_phillips_likelihood(
    obs: dict[str, np.ndarray],
    nairu: Any,
    mc: dict[str, Any],
    gamma_effective: Any,
) -> None:
    """Shared likelihood for price Phillips curve."""
    u_gap = (obs["U"] - nairu) / obs["U"]
    pi_exp_quarterly = quarterly(obs["π_exp"])

    mu = pi_exp_quarterly + gamma_effective * u_gap

    if "Δ4ρm_1" in obs:
        mu = mu + mc["rho_pi"] * obs["Δ4ρm_1"]
    if "ξ_2" in obs:
        mu = mu + mc["xi_gscpi"] * obs["ξ_2"] ** 2 * np.sign(obs["ξ_2"])

    pm.Normal(
        "observed_price_inflation",
        mu=mu,
        sigma=mc["epsilon_pi"],
        observed=obs["π"],
    )


def price_inflation_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Anchor-augmented Phillips Curve for price inflation (single slope).

    Model: pi = quarterly(pi_exp) + gamma x u_gap + controls + e
    """
    if constant is None:
        constant = {}

    nairu = latents["nairu"]

    with model:
        settings = {
            "gamma_pi": {"mu": -1.5, "sigma": 1.0},
            "epsilon_pi": {"sigma": 0.25},
        }
        if "Δ4ρm_1" in obs:
            settings["rho_pi"] = {"mu": 0.0, "sigma": 0.1}
        if "ξ_2" in obs:
            settings["xi_gscpi"] = {"mu": 0.0, "sigma": 0.1}
        mc = set_model_coefficients(model, settings, constant)
        _price_phillips_likelihood(obs, nairu, mc, gamma_effective=mc["gamma_pi"])

    parts = ["pi = quarterly(pi_exp) + gamma x u_gap"]
    if "Δ4ρm_1" in obs:
        parts.append("rho x d4pm")
    if "ξ_2" in obs:
        parts.append("xi x GSCPI^2")
    parts.append("e")
    return " + ".join(parts)


def price_inflation_regime_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Anchor-augmented Phillips Curve for price inflation (regime-switching).

    Model: pi = quarterly(pi_exp) + gamma_regime x u_gap + controls + e
    """
    if constant is None:
        constant = {}

    nairu = latents["nairu"]

    with model:
        settings = {
            "gamma_pi_pre_gfc": {"mu": -1.5, "sigma": 1.0},
            "gamma_pi_gfc": {"mu": -0.5, "sigma": 0.5},
            "gamma_pi_covid": {"mu": -2.5, "sigma": 1.0},
            "epsilon_pi": {"sigma": 0.25},
        }
        if "Δ4ρm_1" in obs:
            settings["rho_pi"] = {"mu": 0.0, "sigma": 0.1}
        if "ξ_2" in obs:
            settings["xi_gscpi"] = {"mu": 0.0, "sigma": 0.1}
        mc = set_model_coefficients(model, settings, constant)
        gamma_effective = (
            mc["gamma_pi_pre_gfc"] * obs["regime_pre_gfc"]
            + mc["gamma_pi_gfc"] * obs["regime_gfc"]
            + mc["gamma_pi_covid"] * obs["regime_covid"]
        )
        _price_phillips_likelihood(obs, nairu, mc, gamma_effective=gamma_effective)

    parts = ["pi = quarterly(pi_exp) + gamma_regime x u_gap"]
    if "Δ4ρm_1" in obs:
        parts.append("rho x d4pm")
    if "ξ_2" in obs:
        parts.append("xi x GSCPI^2")
    parts.append("e")
    return " + ".join(parts)
