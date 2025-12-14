"""IS curve equation linking output gap to interest rates."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.base import set_model_coefficients


def is_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    potential_output: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """IS curve linking output gap to real interest rate gap.

    IS equation: y_gap_t = ρ·y_gap_{t-1} - β·(r_{t-2} - r*) + ε

    Implemented as likelihood on log_gdp:
        log_gdp_t = potential_t + ρ·(log_gdp_{t-1} - potential_{t-1}) - β·(r_{t-2} - r*) + ε

    Where:
        - y_gap = log_gdp - potential_output (output gap)
        - r = cash_rate - π_anchor (real interest rate)
        - r* = det_r_star (equilibrium real rate)
        - ρ = output gap persistence (typically 0.8-0.9)
        - β = interest rate sensitivity (typically 0.1-0.2)

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP
            - "cash_rate": Nominal interest rate (annual)
            - "π_anchor": Inflation anchor (annual)
            - "det_r_star": Deterministic equilibrium real rate
        model: PyMC model context
        potential_output: Potential output latent variable
        constant: Optional fixed values for coefficients

    Example:
        is_equation(inputs, model, potential_output)
    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "rho_is": {"mu": 0.85, "sigma": 0.1},    # output gap persistence
            "beta_is": {"mu": 0.15, "sigma": 0.1},   # interest rate sensitivity
            "epsilon_is": {"sigma": 0.4},            # error term
        }
        mc = set_model_coefficients(model, settings, constant)

        # Real interest rate: i - π_anchor
        real_rate = inputs["cash_rate"] - inputs["π_anchor"]

        # Real rate gap: (r - r*)
        rate_gap = real_rate - inputs["det_r_star"]

        # Lagged rate gap (lag=2)
        rate_gap_lag2 = rate_gap[:-2]

        # Output gap = log_gdp - potential_output
        output_gap = inputs["log_gdp"] - potential_output

        # Lagged output gap
        output_gap_lag1 = output_gap[1:-1]

        # Potential at time t (for t=2,...,T-1)
        potential_t = potential_output[2:]

        # IS equation predicts log_gdp_t:
        # log_gdp_t = potential_t + ρ·y_gap_{t-1} - β·rate_gap_{t-2}
        predicted_log_gdp = (
            potential_t
            + mc["rho_is"] * output_gap_lag1
            - mc["beta_is"] * rate_gap_lag2
        )

        pm.Normal(
            "observed_is",
            mu=predicted_log_gdp,
            sigma=mc["epsilon_is"],
            observed=inputs["log_gdp"][2:],
        )
