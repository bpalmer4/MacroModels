"""Post-sampling analysis utilities for PyMC models.

Includes:
- Diagnostics: MCMC convergence checks
- Extraction: Get variables from traces
- Plotting: Posterior visualization
- Decomposition: Inflation demand/supply attribution
"""

from src.models.nairu.analysis.diagnostics import check_for_zero_coeffs, check_model_diagnostics
from src.models.nairu.analysis.extraction import (
    get_scalar_var,
    get_scalar_var_names,
    get_vector_var,
    is_scalar_var,
)
from src.models.nairu.analysis.inflation_decomposition import (
    HCOEInflationDecomposition,
    InflationDecomposition,
    WageInflationDecomposition,
    decompose_hcoe_inflation,
    decompose_inflation,
    decompose_wage_inflation,
    get_policy_diagnosis,
    inflation_policy_summary,
    plot_demand_contribution,
    plot_hcoe_decomposition,
    plot_hcoe_drivers,
    plot_hcoe_drivers_unscaled,
    plot_inflation_decomposition,
    plot_inflation_drivers,
    plot_inflation_drivers_proportional,
    plot_supply_contribution,
    plot_wage_decomposition,
    plot_wage_drivers,
    plot_wage_drivers_unscaled,
)
from src.models.nairu.analysis.observations_plot import plot_obs_grid
from src.models.nairu.analysis.plot_nairu_output import (
    plot_gdp_vs_potential,
    plot_output_gap,
    plot_potential_growth,
    plot_r_star_input_vs_output,
)
from src.models.nairu.analysis.plot_nairu_rates import plot_equilibrium_rates, plot_taylor_rule
from src.models.nairu.analysis.plot_nairu_unemployment import plot_nairu, plot_unemployment_gap
from src.models.nairu.analysis.plot_phillips_curves import (
    plot_phillips_curve_slope,
    plot_phillips_curves,
)
from src.models.nairu.analysis.plot_posterior_timeseries import plot_posterior_timeseries
from src.models.nairu.analysis.plot_posteriors_bar import plot_posteriors_bar
from src.models.nairu.analysis.plot_posteriors_kde import plot_posteriors_kde
from src.models.nairu.analysis.posterior_predictive_checks import posterior_predictive_checks
from src.models.nairu.analysis.plot_capital_deepening import plot_capital_deepening
from src.models.nairu.analysis.residual_autocorrelation import residual_autocorrelation_analysis

__all__ = [
    "check_for_zero_coeffs",
    "check_model_diagnostics",
    "decompose_hcoe_inflation",
    "decompose_inflation",
    "decompose_wage_inflation",
    "get_policy_diagnosis",
    "get_scalar_var",
    "get_scalar_var_names",
    "get_vector_var",
    "HCOEInflationDecomposition",
    "InflationDecomposition",
    "inflation_policy_summary",
    "WageInflationDecomposition",
    "is_scalar_var",
    "plot_demand_contribution",
    "plot_equilibrium_rates",
    "plot_gdp_vs_potential",
    "plot_hcoe_decomposition",
    "plot_hcoe_drivers",
    "plot_hcoe_drivers_unscaled",
    "plot_inflation_decomposition",
    "plot_inflation_drivers",
    "plot_inflation_drivers_proportional",
    "plot_nairu",
    "plot_capital_deepening",
    "plot_obs_grid",
    "plot_output_gap",
    "plot_phillips_curve_slope",
    "plot_phillips_curves",
    "plot_posterior_timeseries",
    "plot_posteriors_bar",
    "plot_posteriors_kde",
    "plot_potential_growth",
    "plot_r_star_input_vs_output",
    "plot_supply_contribution",
    "plot_taylor_rule",
    "plot_wage_decomposition",
    "plot_wage_drivers",
    "plot_wage_drivers_unscaled",
    "plot_unemployment_gap",
    "posterior_predictive_checks",
    "residual_autocorrelation_analysis",
]
