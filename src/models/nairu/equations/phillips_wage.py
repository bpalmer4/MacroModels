"""Wage Phillips curve equations (ULC-based, single slope and regime-switching)."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients
from src.utilities.rate_conversion import quarterly


def wage_growth_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
    wage_expectations: bool = True,
    wage_price_passthrough: bool = False,
) -> str:
    """Wage Phillips curve (ULC, single slope).

    Model: dulc = alpha [+ pi_exp] + gamma x u_gap + lambda x dU/U [+ phi x d4dfd] + e
    """
    if constant is None:
        constant = {}

    nairu = latents["nairu"]

    with model:
        settings = {
            "alpha_wg": {"mu": 0, "sigma": 1.0},
            "gamma_wg": {"mu": -1.5, "sigma": 1.0, "upper": 0},
            "lambda_wg": {"mu": -4.0, "sigma": 2.0},
            "epsilon_wg": {"sigma": 1.0},
        }
        if wage_price_passthrough:
            settings["phi_wg"] = {"mu": 0.0, "sigma": 0.2}
        mc = set_model_coefficients(model, settings, constant)

        u_gap = (obs["U"] - nairu) / obs["U"]
        mu = mc["alpha_wg"] + mc["gamma_wg"] * u_gap + mc["lambda_wg"] * obs["ΔU_1_over_U"]
        if wage_expectations:
            mu = mu + quarterly(obs["π_exp"])
        if wage_price_passthrough:
            mu = mu + mc["phi_wg"] * obs["Δ4dfd"]

        pm.Normal(
            "observed_wage_growth",
            mu=mu,
            sigma=mc["epsilon_wg"],
            observed=obs["Δulc"],
        )

    parts = ["dulc = alpha"]
    if wage_expectations:
        parts.append("pi_exp")
    parts.append("gamma x u_gap + lambda x dU/U")
    if wage_price_passthrough:
        parts.append("phi x d4dfd")
    parts.append("e")
    return " + ".join(parts)


def wage_growth_regime_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
    wage_expectations: bool = True,
    wage_price_passthrough: bool = False,
) -> str:
    """Wage Phillips curve (ULC, regime-switching).

    Model: dulc = alpha [+ pi_exp] + gamma_regime x u_gap + lambda x dU/U [+ phi x d4dfd] + e
    """
    if constant is None:
        constant = {}

    nairu = latents["nairu"]

    with model:
        settings = {
            "alpha_wg": {"mu": 0, "sigma": 1.0},
            "gamma_wg_pre_gfc": {"mu": -1.5, "sigma": 1.0, "upper": 0},
            "gamma_wg_gfc": {"mu": -0.5, "sigma": 0.5, "upper": 0},
            "gamma_wg_covid": {"mu": -1.5, "sigma": 0.75, "upper": 0},
            "lambda_wg": {"mu": -4.0, "sigma": 2.0},
            "epsilon_wg": {"sigma": 1.0},
        }
        if wage_price_passthrough:
            settings["phi_wg"] = {"mu": 0.0, "sigma": 0.2}
        mc = set_model_coefficients(model, settings, constant)

        u_gap = (obs["U"] - nairu) / obs["U"]
        gamma_effective = (
            mc["gamma_wg_pre_gfc"] * obs["regime_pre_gfc"]
            + mc["gamma_wg_gfc"] * obs["regime_gfc"]
            + mc["gamma_wg_covid"] * obs["regime_covid"]
        )

        mu = mc["alpha_wg"] + gamma_effective * u_gap + mc["lambda_wg"] * obs["ΔU_1_over_U"]
        if wage_expectations:
            mu = mu + quarterly(obs["π_exp"])
        if wage_price_passthrough:
            mu = mu + mc["phi_wg"] * obs["Δ4dfd"]

        pm.Normal(
            "observed_wage_growth",
            mu=mu,
            sigma=mc["epsilon_wg"],
            observed=obs["Δulc"],
        )

    parts = ["dulc = alpha"]
    if wage_expectations:
        parts.append("pi_exp")
    parts.append("gamma_regime x u_gap + lambda x dU/U")
    if wage_price_passthrough:
        parts.append("phi x d4dfd")
    parts.append("e")
    return " + ".join(parts)
