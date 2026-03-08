"""Wage inflation (ULC growth) decomposition."""

import pandas as pd

from src.models.common.extraction import get_scalar_var, get_vector_var
from src.models.nairu.analysis._decomposition_helpers import get_regime_gamma
from src.models.nairu.analysis.decomposition_types import WageInflationDecomposition
from src.utilities.rate_conversion import quarterly


def decompose_wage_inflation(trace, obs, obs_index, wage_expectations=False):
    """Decompose wage inflation (ULC growth) into demand and price components."""
    alpha_wg = get_scalar_var("alpha_wg", trace).median()
    lambda_wg = get_scalar_var("lambda_wg", trace).median()
    has_phi = "phi_wg" in trace.posterior
    phi_wg = get_scalar_var("phi_wg", trace).median() if has_phi else 0.0

    has_exp = "theta_wg" in trace.posterior or wage_expectations
    if "theta_wg" in trace.posterior:
        theta_wg = get_scalar_var("theta_wg", trace).median()
    elif wage_expectations:
        theta_wg = 1.0
    else:
        theta_wg = 0.0

    gamma_wg = get_regime_gamma(trace, obs_index, "gamma_wg")
    nairu = pd.Series(get_vector_var("nairu", trace).median(axis=1).values, index=obs_index)

    U = pd.Series(obs["U"], index=obs_index)
    pi_exp = pd.Series(obs["π_exp"], index=obs_index)
    ulc_observed = pd.Series(obs["Δulc"], index=obs_index)
    delta_u_over_u = pd.Series(obs["ΔU_1_over_U"], index=obs_index)
    dfd_growth = pd.Series(obs["Δ4dfd"], index=obs_index)

    anchor = alpha_wg + theta_wg * quarterly(pi_exp)
    demand = gamma_wg * (U - nairu) / U + lambda_wg * delta_u_over_u
    price_passthrough = phi_wg * dfd_growth
    fitted = anchor + demand + price_passthrough
    residual = ulc_observed - fitted

    return WageInflationDecomposition(
        observed=ulc_observed, anchor=anchor, demand=demand,
        price_passthrough=price_passthrough, residual=residual,
        fitted=fitted, index=obs_index,
        has_price_passthrough=has_phi, has_expectations=has_exp,
    )
