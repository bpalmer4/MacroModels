"""Hourly compensation Phillips curve equations (single slope and regime-switching)."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients
from src.utilities.rate_conversion import quarterly


def hourly_coe_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
    wage_expectations: bool = True,
    wage_price_passthrough: bool = False,
) -> str:
    """Hourly compensation Phillips curve (single slope).

    Model: dhcoe = alpha [+ pi_exp] + gamma x u_gap + lambda x dU/U [+ phi x d4dfd] + psi x MFP + e
    """
    if constant is None:
        constant = {}

    nairu = latents["nairu"]

    with model:
        settings = {
            "alpha_hcoe": {"mu": 0, "sigma": 0.5},
            "gamma_hcoe": {"mu": -1.0, "sigma": 0.75, "upper": 0},
            "lambda_hcoe": {"mu": -4.0, "sigma": 2.0},
            "psi_hcoe": {"mu": 1.0, "sigma": 0.5, "lower": 0},
            "epsilon_hcoe": {"sigma": 0.75},
        }
        if wage_price_passthrough:
            settings["phi_hcoe"] = {"mu": 0.0, "sigma": 0.2}
        mc = set_model_coefficients(model, settings, constant)

        u_gap = (obs["U"] - nairu) / obs["U"]
        mu = (
            mc["alpha_hcoe"]
            + mc["gamma_hcoe"] * u_gap
            + mc["lambda_hcoe"] * obs["ΔU_1_over_U"]
            + mc["psi_hcoe"] * obs["mfp_growth"]
        )
        if wage_expectations:
            mu = mu + quarterly(obs["π_exp"])
        if wage_price_passthrough:
            mu = mu + mc["phi_hcoe"] * obs["Δ4dfd"]

        pm.Normal(
            "observed_hourly_coe",
            mu=mu,
            sigma=mc["epsilon_hcoe"],
            observed=obs["Δhcoe"],
        )

    parts = ["dhcoe = alpha"]
    if wage_expectations:
        parts.append("pi_exp")
    parts.append("gamma x u_gap + lambda x dU/U")
    if wage_price_passthrough:
        parts.append("phi x d4dfd")
    parts.append("psi x MFP + e")
    return " + ".join(parts)


def hourly_coe_regime_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
    wage_expectations: bool = True,
    wage_price_passthrough: bool = False,
) -> str:
    """Hourly compensation Phillips curve (regime-switching).

    Model: dhcoe = alpha [+ pi_exp] + gamma_regime x u_gap + lambda x dU/U [+ phi x d4dfd] + psi x MFP + e
    """
    if constant is None:
        constant = {}

    nairu = latents["nairu"]

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
        if wage_price_passthrough:
            settings["phi_hcoe"] = {"mu": 0.0, "sigma": 0.2}
        mc = set_model_coefficients(model, settings, constant)

        u_gap = (obs["U"] - nairu) / obs["U"]
        gamma_effective = (
            mc["gamma_hcoe_pre_gfc"] * obs["regime_pre_gfc"]
            + mc["gamma_hcoe_gfc"] * obs["regime_gfc"]
            + mc["gamma_hcoe_covid"] * obs["regime_covid"]
        )

        mu = (
            mc["alpha_hcoe"]
            + gamma_effective * u_gap
            + mc["lambda_hcoe"] * obs["ΔU_1_over_U"]
            + mc["psi_hcoe"] * obs["mfp_growth"]
        )
        if wage_expectations:
            mu = mu + quarterly(obs["π_exp"])
        if wage_price_passthrough:
            mu = mu + mc["phi_hcoe"] * obs["Δ4dfd"]

        pm.Normal(
            "observed_hourly_coe",
            mu=mu,
            sigma=mc["epsilon_hcoe"],
            observed=obs["Δhcoe"],
        )

    parts = ["dhcoe = alpha"]
    if wage_expectations:
        parts.append("pi_exp")
    parts.append("gamma_regime x u_gap + lambda x dU/U")
    if wage_price_passthrough:
        parts.append("phi x d4dfd")
    parts.append("psi x MFP + e")
    return " + ".join(parts)
