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
    """IS curve linking output gap to interest rates, fiscal, DSR, and housing wealth.

    IS equation:
        y_gap_t = ρ·y_gap_{t-1} - β·(r_{t-2} - r*) + γ·fiscal_{t-1} - δ·Δdsr_{t-1} + η·Δhw_{t-1} + ε

    Implemented as likelihood on log_gdp:
        log_gdp_t = potential_t + ρ·(log_gdp_{t-1} - potential_{t-1})
                    - β·(r_{t-2} - r*) + γ·fiscal_{t-1} - δ·Δdsr_{t-1} + η·Δhw_{t-1} + ε

    Where:
        - y_gap = log_gdp - potential_output (output gap)
        - r = cash_rate - π_anchor (real interest rate)
        - r* = det_r_star (equilibrium real rate)
        - fiscal_impulse = Δlog(G) - Δlog(GDP) (expansionary when positive)
        - Δdsr = change in debt servicing ratio (interest payments / income)
        - Δhw = housing wealth growth (quarterly log change × 100)
        - ρ = output gap persistence (typically 0.8-0.9)
        - β = interest rate sensitivity (typically 0.1-0.2)
        - γ = fiscal impulse effect (expected positive)
        - δ = debt servicing sensitivity (expected positive: rising DSR reduces demand)
        - η = housing wealth effect (expected positive: rising wealth boosts consumption)

    Transmission channels:
        1. Direct rate channel: rates ↑ → borrowing cost ↑ → investment/consumption ↓
        2. DSR channel: rates ↑ → mortgage payments ↑ → disposable income ↓ → spending ↓
        3. Wealth channel: rates ↑ → house prices ↓ → wealth ↓ → consumption ↓

    The wealth channel captures how rate rises affect consumption through housing wealth:
        - RBA estimates MPC out of housing wealth ~2-3 cents per dollar
        - Housing wealth ~$11T, GDP ~$2.5T/yr
        - 1% housing wealth decline → ~$110B → ~$2-3B less consumption → ~0.1% GDP

    Note:
        Terms of trade (ToT) and household saving ratio change were tested
        but found to be statistically indistinguishable from zero.

    Args:
        inputs: Must contain:
            - "log_gdp": Log of real GDP
            - "cash_rate": Nominal interest rate (annual)
            - "π_anchor": Inflation anchor (annual)
            - "det_r_star": Deterministic equilibrium real rate
            - "fiscal_impulse_1": Lagged fiscal impulse (G growth - GDP growth)
            - "Δdsr_1": Lagged change in debt servicing ratio (pp)
            - "Δhw_1": Lagged housing wealth growth (%)
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
            "beta_is": {"mu": 0.20, "sigma": 0.10, "lower": 0},  # interest rate sensitivity (positive)
            "gamma_fi": {"mu": 0.05, "sigma": 0.2, "lower": 0},  # fiscal impulse effect (positive)
            "delta_dsr": {"mu": 0.10, "sigma": 0.05, "lower": 0},  # DSR sensitivity (positive: rising DSR hurts)
            "eta_hw": {"mu": 0.02, "sigma": 0.01, "lower": 0},  # housing wealth effect (positive: rising wealth helps)
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

        # Fiscal impulse (lagged 1 quarter, aligned with t=2,...,T-1)
        fiscal_impulse_lag1 = inputs["fiscal_impulse_1"][2:]

        # Debt servicing ratio change (lagged 1 quarter, aligned with t=2,...,T-1)
        delta_dsr_lag1 = inputs["Δdsr_1"][2:]

        # Housing wealth growth (lagged 1 quarter, aligned with t=2,...,T-1)
        delta_hw_lag1 = inputs["Δhw_1"][2:]

        # IS equation predicts log_gdp_t:
        # log_gdp_t = potential_t + ρ·y_gap_{t-1} - β·rate_gap_{t-2} + γ·fiscal_{t-1} - δ·Δdsr_{t-1} + η·Δhw_{t-1}
        predicted_log_gdp = (
            potential_t
            + mc["rho_is"] * output_gap_lag1
            - mc["beta_is"] * rate_gap_lag2
            + mc["gamma_fi"] * fiscal_impulse_lag1
            - mc["delta_dsr"] * delta_dsr_lag1
            + mc["eta_hw"] * delta_hw_lag1
        )

        pm.Normal(
            "observed_is",
            mu=predicted_log_gdp,
            sigma=mc["epsilon_is"],
            observed=inputs["log_gdp"][2:],
        )
