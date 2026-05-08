"""Indexed-bond observation equation.

Observation: indexed_10y_t = r*_t + tp + e_tp

The 10-year inflation-linked Australian government bond yield is a market
estimate of the average expected real short rate over 10 years plus a term
premium. Treating the term premium as a constant parameter, the indexed yield
is a noisy direct observation of r*. This is the simplest form of a macro-
finance r* model (Christensen-Mouabbi 2024 in spirit), and serves as a
substitute for the convenience-yield observation equation in the FEDS note
(Szoke-Vazquez-Grande-Xavier 2024) — Australia lacks long-history AA corporate
yield data, so we use the indexed bond directly instead of a corporate spread.

The constant term premium is a strong simplification — it ignores the well-
documented decline in term premia post-GFC. But the alternative (modelling
term premium as another latent random walk) re-introduces the very
identification problem this equation is supposed to fix.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def indexed_bond_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """10y indexed bond yield as a noisy observation of r*.

    Model: indexed_10y_t = r*_t + tp + e_tp,  e_tp ~ N(0, sigma_tp)

    tp is a constant term premium (positive prior). Indexed bond yield data
    starts in 1986; for periods before observation, the equation is simply
    not informative (handled by NaN dropping in observations.py).
    """
    if constant is None:
        constant = {}

    r_star = latents["r_star"]

    with model:
        settings = {
            "tp": {"mu": 0.5, "sigma": 1.0, "lower": 0.0},
            "sigma_tp": {"sigma": 0.5},
        }
        mc = set_model_coefficients(model, settings, constant)

        predicted = r_star + mc["tp"]

        pm.Normal(
            "observed_indexed",
            mu=predicted,
            sigma=mc["sigma_tp"],
            observed=obs["indexed_10y"],
        )

    return "indexed_10y_t = r*_t + tp + e_tp,  e_tp ~ N(0, sigma_tp)"
