"""Okun's Law equation linking output gap to unemployment.

Error correction form combining dynamics and equilibrium:
    ΔU_t = β × OG_t + α × (U_{t-1} - NAIRU_{t-1} - γ × OG_{t-1}) + ε

This captures:
- Direct output gap effect (β): How output gap affects unemployment change
- Equilibrium relationship (γ): Where unemployment should settle given output gap
- Adjustment speed (α): How fast unemployment corrects toward equilibrium
"""

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.models.nairu.base import set_model_coefficients


def okun_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pm.Distribution,
    potential_output: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """Error correction form of Okun's Law.

    ΔU_t = β × OG_t + α × (U_{t-1} - NAIRU_{t-1} - γ × OG_{t-1}) + ε

    Where:
        - ΔU is the observed quarterly change in unemployment rate (pp)
        - OG = output_gap = (log_gdp - potential_output)
        - β is the direct output gap effect (negative: positive gap → falling U)
        - γ is the equilibrium coefficient (negative: negative gap → U above NAIRU)
        - α is adjustment speed (negative: U moves toward equilibrium)

    The error correction term (U - NAIRU - γ×OG) measures disequilibrium:
        - When U > equilibrium, error > 0, α×error < 0, so ΔU < 0 (U falls)
        - When U < equilibrium, error < 0, α×error > 0, so ΔU > 0 (U rises)

    This provides both dynamics and equilibrium anchoring in a single equation,
    avoiding the over-identification issues of separate level/change equations.

    Interpretation:
        - β ≈ -0.15: 1% output gap → -0.15pp direct effect on ΔU
        - γ ≈ -0.4: Equilibrium U = NAIRU - 0.4 × OG
        - α ≈ -0.3: 30% of disequilibrium corrected each quarter

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP
            - "ΔU": Change in unemployment rate
            - "U": Unemployment rate (level, for lagged values)
        model: PyMC model context
        nairu: NAIRU latent variable
        potential_output: Potential output latent variable
        constant: Optional fixed values for coefficients

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "beta_okun": {"mu": -0.15, "sigma": 0.1, "upper": 0},
            "alpha_okun": {"mu": -0.3, "sigma": 0.15, "upper": 0},
            "gamma_okun": {"mu": -0.4, "sigma": 0.2, "upper": 0},
            "epsilon_okun": {"sigma": 0.3},
        }
        mc = set_model_coefficients(model, settings, constant)

        # Output gap: (Y - Y*) in log terms
        output_gap = inputs["log_gdp"] - potential_output

        # Lagged values for error correction term (pad with first value to keep same length)
        U_lag = pt.concatenate([[inputs["U"][0]], inputs["U"][:-1]])
        nairu_lag = pt.concatenate([[nairu[0]], nairu[:-1]])
        og_lag = pt.concatenate([[output_gap[0]], output_gap[:-1]])

        # Error correction term: (U_{t-1} - NAIRU_{t-1} - γ × OG_{t-1})
        # Positive when U is above equilibrium
        equilibrium_error = U_lag - nairu_lag - mc["gamma_okun"] * og_lag

        # Okun's Law (error correction form):
        # ΔU_t = β × OG_t + α × error_{t-1}
        pm.Normal(
            "okun_ec",
            mu=mc["beta_okun"] * output_gap + mc["alpha_okun"] * equilibrium_error,
            sigma=mc["epsilon_okun"],
            observed=inputs["ΔU"],
        )
