"""IS curve equation linking output gap to interest rates and wealth."""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def is_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    potential_output: pm.Distribution,
    constant: dict[str, Any] | None = None,
) -> None:
    """IS curve linking output gap to interest rates and fiscal impulse.

    Model: y_gap = ρ × y_gap_{-1} - β_is × r_gap_{-2} + γ_fi × fiscal_{-1} + ε

    Implemented as likelihood on log_gdp:
        log_gdp_t = potential_t + ρ·(log_gdp_{t-1} - potential_{t-1})
                    - β·(r_{t-2} - r*) + γ·fiscal_{t-1} + ε

    Where:
        - y_gap = log_gdp - potential_output (output gap)
        - r = cash_rate - π_exp (real interest rate)
        - r* = det_r_star (equilibrium real rate)
        - fiscal_impulse = Δlog(G) - Δlog(GDP) (expansionary when positive)
        - ρ = output gap persistence (typically 0.8-0.9)
        - β = interest rate sensitivity (typically 0.1-0.2)
        - γ = fiscal impulse effect (expected positive)

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP
            - "cash_rate": Nominal interest rate (annual)
            - "π_exp": Inflation expectations (annual)
            - "det_r_star": Deterministic equilibrium real rate
            - "fiscal_impulse_1": Lagged fiscal impulse (G growth - GDP growth)
        model: PyMC model context
        potential_output: Potential output latent variable
        constant: Optional fixed values for coefficients

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            "rho_is": {"mu": 0.85, "sigma": 0.1},    # output gap persistence
            "beta_is": {"mu": 0.20, "sigma": 0.10, "lower": 0},  # interest rate sensitivity (positive)
            "gamma_fi": {"mu": 0.05, "sigma": 0.2, "lower": 0},  # fiscal impulse effect (positive)
            "epsilon_is": {"sigma": 0.4},            # error term
        }
        mc = set_model_coefficients(model, settings, constant)

        # Real interest rate: i - π_exp
        real_rate = inputs["cash_rate"] - inputs["π_exp"]

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

        # Fiscal impulse (lagged 1 quarter, aligned with t=2,...,T-1)
        fiscal_impulse_lag1 = inputs["fiscal_impulse_1"][2:]

        # IS equation predicts log_gdp_t:
        # log_gdp_t = potential_t + ρ·y_gap_{t-1} - β·rate_gap_{t-2} + γ·fiscal_{t-1}
        predicted_log_gdp = (
            potential_t
            + mc["rho_is"] * output_gap_lag1
            - mc["beta_is"] * rate_gap_lag2
            + mc["gamma_fi"] * fiscal_impulse_lag1
        )

        pm.Normal(
            "observed_is",
            mu=predicted_log_gdp,
            sigma=mc["epsilon_is"],
            observed=inputs["log_gdp"][2:],
        )
