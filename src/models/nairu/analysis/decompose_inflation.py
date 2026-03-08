"""Price inflation decomposition into demand and supply components."""

import numpy as np
import pandas as pd

from src.models.common.extraction import get_scalar_var, get_vector_var
from src.models.nairu.analysis._decomposition_helpers import get_regime_gamma
from src.models.nairu.analysis.decomposition_types import InflationDecomposition
from src.utilities.rate_conversion import quarterly


def decompose_inflation(trace, obs, obs_index):
    """Decompose price inflation into demand and supply components."""
    gamma_pi = get_regime_gamma(trace, obs_index, "gamma_pi")
    rho_pi = get_scalar_var("rho_pi", trace).median() if "rho_pi" in trace.posterior else 0.0
    xi_gscpi = get_scalar_var("xi_gscpi", trace).median() if "xi_gscpi" in trace.posterior else 0.0

    nairu = pd.Series(get_vector_var("nairu", trace).median(axis=1).values, index=obs_index)
    U = pd.Series(obs["U"], index=obs_index)
    pi_exp = pd.Series(obs["π_exp"], index=obs_index)
    pi_observed = pd.Series(obs["π"], index=obs_index)
    delta_4_pm_lag1 = pd.Series(obs.get("Δ4ρm_1", np.zeros(len(obs_index))), index=obs_index)
    gscpi = pd.Series(obs.get("ξ_2", np.zeros(len(obs_index))), index=obs_index)

    anchor = quarterly(pi_exp)
    demand = gamma_pi * (U - nairu) / U
    supply_import = rho_pi * delta_4_pm_lag1
    supply_gscpi = xi_gscpi * gscpi ** 2 * np.sign(gscpi)
    fitted = anchor + demand + supply_import + supply_gscpi
    residual = pi_observed - fitted

    return InflationDecomposition(
        observed=pi_observed, anchor=anchor, demand=demand,
        supply_import=supply_import, supply_gscpi=supply_gscpi,
        residual=residual, fitted=fitted, index=obs_index,
        has_import_price="rho_pi" in trace.posterior,
        has_gscpi="xi_gscpi" in trace.posterior,
    )
