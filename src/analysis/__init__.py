"""Post-sampling analysis utilities for PyMC models.

Includes:
- Diagnostics: MCMC convergence checks
- Extraction: Get variables from traces
- Plotting: Posterior visualization (scalars)
- Timeseries: Latent state analysis and plotting
"""

from src.analysis.diagnostics import check_for_zero_coeffs, check_model_diagnostics
from src.analysis.extraction import (
    get_scalar_var,
    get_scalar_var_names,
    get_vector_var,
    is_scalar_var,
)
from src.analysis.plotting import (
    plot_posteriors_bar,
    plot_posteriors_kde,
    plot_timeseries,
    posterior_predictive_checks,
    residual_autocorrelation_analysis,
)
from src.analysis.timeseries import (
    compute_nairu_stats,
    compute_potential_stats,
    compute_taylor_rule,
    plot_gaps_comparison,
    plot_gdp_vs_potential,
    plot_nairu,
    plot_output_gap,
    plot_potential_growth,
    plot_unemployment_gap,
)

__all__ = [
    # Diagnostics
    "check_for_zero_coeffs",
    "check_model_diagnostics",
    # Extraction
    "get_scalar_var",
    "get_scalar_var_names",
    "get_vector_var",
    "is_scalar_var",
    # Plotting (scalars)
    "plot_posteriors_bar",
    "plot_posteriors_kde",
    "plot_timeseries",
    "posterior_predictive_checks",
    "residual_autocorrelation_analysis",
    # Timeseries
    "compute_nairu_stats",
    "compute_potential_stats",
    "compute_taylor_rule",
    "plot_gaps_comparison",
    "plot_gdp_vs_potential",
    "plot_nairu",
    "plot_output_gap",
    "plot_potential_growth",
    "plot_unemployment_gap",
]
