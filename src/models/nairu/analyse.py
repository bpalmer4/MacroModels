"""Output analysis charts for NAIRU + Output Gap model.

This module handles:
- Core estimate plots (NAIRU, output gap, potential growth)
- Interest rate analysis (Taylor rule, equilibrium rates)
- Phillips curve plots (scatter, regime slopes)
- Inflation decomposition (price, wage-ULC, HCOE)
"""

from pathlib import Path

import mgplot as mg

from src.data import get_cash_rate_monthly
from src.models.nairu.analysis import (
    decompose_hcoe_inflation,
    decompose_inflation,
    decompose_wage_inflation,
    plot_decomposition,
    plot_equations,
    plot_equilibrium_rates,
    plot_gdp_vs_potential,
    plot_nairu,
    plot_output_gap,
    plot_phillips_curves,
    plot_phillips_slope,
    plot_potential_growth,
    plot_potential_growth_comparison,
    plot_potential_growth_smoothing,
    plot_taylor_rule,
    plot_unemployment_gap,
)
from src.models.nairu.results import DEFAULT_CHART_BASE, load_results


def run_analyse(
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
    chart_dir: Path | str | None = None,
    verbose: bool = False,
    show_plots: bool = False,
) -> "NAIRUResults":
    """Generate all output analysis charts."""
    results = load_results(output_dir=output_dir, prefix=prefix, rebuild_model=False)
    config = results.config

    if chart_dir is None:
        chart_dir = DEFAULT_CHART_BASE / config.chart_dir_name
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    rfooter = config.rfooter
    print(f"Running analysis [{config.label}]...\n")

    # --- Model equations ---
    plot_equations(
        equations=getattr(results.model, "_descriptions", None),
        constants=results.constants,
        show=show_plots,
    )

    # --- Core estimates ---
    plot_nairu(results, rfooter=rfooter, show=show_plots)
    plot_unemployment_gap(results, rfooter=rfooter, show=show_plots)
    plot_output_gap(results, rfooter=rfooter, show=show_plots)
    plot_gdp_vs_potential(results, rfooter=rfooter, show=show_plots)
    plot_potential_growth(results, rfooter=rfooter, show=show_plots)
    plot_potential_growth_smoothing(results, rfooter=rfooter, show=show_plots)
    plot_potential_growth_comparison(results, rfooter=rfooter, show=show_plots)

    # --- Interest rate analysis ---
    cash_rate_monthly = get_cash_rate_monthly().data
    if cash_rate_monthly is not None:
        plot_taylor_rule(results, cash_rate_monthly=cash_rate_monthly, rfooter=rfooter, show=show_plots)
        plot_equilibrium_rates(results, cash_rate_monthly=cash_rate_monthly, rfooter=rfooter, show=show_plots)

    # --- Phillips curves ---
    plot_phillips_curves(results, rfooter=rfooter, show=show_plots)
    if config.regime_switching:
        plot_phillips_slope(results, curve_type="price", rfooter=rfooter, show=show_plots)
        if config.include_wage_growth:
            plot_phillips_slope(results, curve_type="wage", rfooter=rfooter, show=show_plots)
        if config.include_hourly_coe:
            plot_phillips_slope(results, curve_type="hcoe", rfooter=rfooter, show=show_plots)

    # --- Inflation decomposition ---
    decomp = decompose_inflation(results.trace, results.obs, results.obs_index)
    plot_decomposition(decomp, rfooter=rfooter, show=show_plots)

    if config.include_wage_growth:
        wage_decomp = decompose_wage_inflation(
            results.trace, results.obs, results.obs_index,
            wage_expectations=config.wage_expectations,
        )
        plot_decomposition(wage_decomp, rfooter=rfooter, show=show_plots)

    if config.include_hourly_coe:
        hcoe_decomp = decompose_hcoe_inflation(
            results.trace, results.obs, results.obs_index,
            wage_expectations=config.wage_expectations,
        )
        plot_decomposition(hcoe_decomp, rfooter=rfooter, show=show_plots)

    print(f"\nAnalysis charts saved to: {chart_dir}")
    return results
