"""Hourly COE growth decomposition."""

import arviz as az
import numpy as np
import pandas as pd

from src.models.common.extraction import get_scalar_var, get_vector_var
from src.models.nairu.analysis._decomposition_helpers import get_regime_gamma
from src.models.nairu.analysis.decomposition_types import HCOEInflationDecomposition
from src.utilities.rate_conversion import quarterly


def decompose_hcoe_inflation(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    wage_expectations: bool = False,
) -> HCOEInflationDecomposition:
    """Decompose hourly COE growth into demand, price, and productivity components."""
    alpha_hcoe = get_scalar_var("alpha_hcoe", trace).median()
    lambda_hcoe = get_scalar_var("lambda_hcoe", trace).median()
    has_phi = "phi_hcoe" in trace.posterior
    phi_hcoe = get_scalar_var("phi_hcoe", trace).median() if has_phi else 0.0
    psi_hcoe = get_scalar_var("psi_hcoe", trace).median()

    has_exp = "theta_hcoe" in trace.posterior or wage_expectations
    if "theta_hcoe" in trace.posterior:
        theta_hcoe = get_scalar_var("theta_hcoe", trace).median()
    elif wage_expectations:
        theta_hcoe = 1.0
    else:
        theta_hcoe = 0.0

    gamma_hcoe = get_regime_gamma(trace, obs_index, "gamma_hcoe")
    nairu = pd.Series(get_vector_var("nairu", trace).median(axis=1).values, index=obs_index)

    U = pd.Series(obs["U"], index=obs_index)
    pi_exp = pd.Series(obs["π_exp"], index=obs_index)
    hcoe_observed = pd.Series(obs["Δhcoe"], index=obs_index)
    delta_u_over_u = pd.Series(obs["ΔU_1_over_U"], index=obs_index)
    dfd_growth = pd.Series(obs["Δ4dfd"], index=obs_index)
    mfp_growth = pd.Series(obs["mfp_growth"], index=obs_index)

    anchor = alpha_hcoe + theta_hcoe * quarterly(pi_exp)
    demand = gamma_hcoe * (U - nairu) / U + lambda_hcoe * delta_u_over_u
    price_passthrough = phi_hcoe * dfd_growth
    productivity = psi_hcoe * mfp_growth
    fitted = anchor + demand + price_passthrough + productivity
    residual = hcoe_observed - fitted

    return HCOEInflationDecomposition(
        observed=hcoe_observed, anchor=anchor, demand=demand,
        price_passthrough=price_passthrough, productivity=productivity,
        residual=residual, fitted=fitted, index=obs_index,
        has_price_passthrough=has_phi, has_expectations=has_exp,
    )
