"""Analysis and plotting modules for NAIRU + Output Gap model."""

from src.models.nairu.analysis.decompose_hcoe_inflation import decompose_hcoe_inflation
from src.models.nairu.analysis.decompose_inflation import decompose_inflation
from src.models.nairu.analysis.decompose_wage_inflation import decompose_wage_inflation
from src.models.nairu.analysis.decomposition_types import (
    HCOEInflationDecomposition,
    InflationDecomposition,
    WageInflationDecomposition,
)
from src.models.nairu.analysis.plot_decomposition import plot_decomposition
from src.models.nairu.analysis.plot_equations import plot_equations
from src.models.nairu.analysis.plot_equilibrium_rates import plot_equilibrium_rates
from src.models.nairu.analysis.plot_gdp_vs_potential import plot_gdp_vs_potential
from src.models.nairu.analysis.plot_nairu import plot_nairu
from src.models.nairu.analysis.plot_nairu_comparison import plot_nairu_comparison
from src.models.nairu.analysis.plot_obs_grid import plot_obs_grid
from src.models.nairu.analysis.plot_output_gap import plot_output_gap
from src.models.nairu.analysis.plot_output_gap_comparison import plot_output_gap_comparison
from src.models.nairu.analysis.plot_phillips_curves import plot_phillips_curves
from src.models.nairu.analysis.plot_phillips_slope import plot_phillips_slope
from src.models.nairu.analysis.plot_posteriors_bar import plot_posteriors_bar
from src.models.nairu.analysis.plot_posteriors_kde import plot_posteriors_kde
from src.models.nairu.analysis.plot_potential_growth import plot_potential_growth
from src.models.nairu.analysis.plot_potential_growth_comparison import plot_potential_growth_comparison
from src.models.nairu.analysis.plot_potential_growth_smoothing import plot_potential_growth_smoothing
from src.models.nairu.analysis.plot_taylor_rule import plot_taylor_rule
from src.models.nairu.analysis.plot_unemployment_gap import plot_unemployment_gap
from src.models.nairu.analysis.posterior_predictive import posterior_predictive_checks
from src.models.nairu.analysis.residual_autocorrelation import residual_autocorrelation_analysis

__all__ = [
    "decompose_hcoe_inflation",
    "decompose_inflation",
    "decompose_wage_inflation",
    "HCOEInflationDecomposition",
    "InflationDecomposition",
    "WageInflationDecomposition",
    "plot_decomposition",
    "plot_equations",
    "plot_equilibrium_rates",
    "plot_gdp_vs_potential",
    "plot_nairu",
    "plot_nairu_comparison",
    "plot_obs_grid",
    "plot_output_gap",
    "plot_output_gap_comparison",
    "plot_phillips_curves",
    "plot_phillips_slope",
    "plot_posteriors_bar",
    "plot_posteriors_kde",
    "plot_potential_growth",
    "plot_potential_growth_comparison",
    "plot_potential_growth_smoothing",
    "plot_taylor_rule",
    "plot_unemployment_gap",
    "posterior_predictive_checks",
    "residual_autocorrelation_analysis",
]
