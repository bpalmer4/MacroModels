"""Post-sampling analysis utilities for PyMC models.

Includes:
- Diagnostics: MCMC convergence checks
- Extraction: Get variables from traces
- Plotting: Posterior visualization
- Decomposition: Inflation demand/supply attribution
"""

from src.analysis.diagnostics import check_for_zero_coeffs, check_model_diagnostics
from src.analysis.extraction import (
    get_scalar_var,
    get_scalar_var_names,
    get_vector_var,
    is_scalar_var,
)
from src.analysis.inflation_decomposition import (
    InflationDecomposition,
    decompose_inflation,
    get_policy_diagnosis,
    inflation_policy_summary,
    plot_demand_contribution,
    plot_inflation_decomposition,
    plot_inflation_drivers,
    plot_inflation_drivers_proportional,
    plot_supply_contribution,
)
from src.analysis.observations_plot import plot_obs_grid
from src.analysis.plot_posterior_timeseries import plot_posterior_timeseries
from src.analysis.plot_posteriors_bar import plot_posteriors_bar
from src.analysis.plot_posteriors_kde import plot_posteriors_kde
from src.analysis.posterior_predictive_checks import posterior_predictive_checks
from src.analysis.residual_autocorrelation import residual_autocorrelation_analysis

__all__ = [
    "check_for_zero_coeffs",
    "check_model_diagnostics",
    "decompose_inflation",
    "get_policy_diagnosis",
    "get_scalar_var",
    "get_scalar_var_names",
    "get_vector_var",
    "InflationDecomposition",
    "inflation_policy_summary",
    "is_scalar_var",
    "plot_demand_contribution",
    "plot_inflation_decomposition",
    "plot_inflation_drivers",
    "plot_inflation_drivers_proportional",
    "plot_obs_grid",
    "plot_posterior_timeseries",
    "plot_posteriors_bar",
    "plot_posteriors_kde",
    "plot_supply_contribution",
    "posterior_predictive_checks",
    "residual_autocorrelation_analysis",
]
