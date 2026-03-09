"""Forecast scenario plotting.

All scenario plots share the same structure: historical line,
scenario fan, hold HDI band, reference line. The generic
_plot_scenarios helper does the work; thin wrappers configure
the data source, labels, and reference lines.
"""

from collections.abc import Callable  # noqa: TC003 — used in annotations
from pathlib import Path

import mgplot as mg
import pandas as pd
from matplotlib.axes import Axes

from src.data.inflation import get_trimmed_mean_qrtly
from src.models.nairu.forecast import (
    SCENARIO_COLORS,
    SCENARIO_ORDER,
    ForecastResults,
)
from src.models.nairu.results import NAIRUResults
from src.utilities.rate_conversion import annualize


def _plot_scenarios(
    scenario_results: dict[str, ForecastResults],
    extract_fn: Callable[[ForecastResults], pd.Series],
    hist_series: pd.Series | None = None,
    title: str = "",
    ylabel: str = "",
    lheader: str = "",
    rfooter: str = "",
    ref_line: float | None = None,
    ref_label: str = "",
    n_history: int = 4,
    show: bool = False,
) -> None:
    """Plot scenario forecasts with historical overlay.

    Args:
        scenario_results: {name: ForecastResults}
        extract_fn: Takes a ForecastResults, returns a pd.Series (median)
        hist_series: Historical series to prepend
        title: Chart title (rate suffix added automatically)
        ylabel: Y-axis label
        lheader: Left header text
        rfooter: Right footer text
        ref_line: Optional horizontal reference line
        ref_label: Label for reference line
        n_history: Number of historical periods to show
        show: Display interactively

    """
    hold = scenario_results.get("hold")
    current_rate = hold.cash_rate if hold else None
    rate_str = f" (from {current_rate:.2f}%)" if current_rate is not None else ""

    # Historical
    ax = None
    if hist_series is not None:
        model_end = hold.obs_index[-1] if hold else None
        if model_end is not None:
            hist_series = hist_series.loc[hist_series.index >= model_end - n_history + 1]
        else:
            hist_series = hist_series.iloc[-n_history:]
        hist_series.name = "Actual"
        ax = mg.line_plot(hist_series, color="black", width=2)

    # Scenario medians
    for name in SCENARIO_ORDER:
        if name not in scenario_results:
            continue
        series = extract_fn(scenario_results[name])
        series.name = name
        if ax is None:
            ax = mg.line_plot(series, color=SCENARIO_COLORS[name],
                              width=2 if name == "hold" else 1.5)
        else:
            mg.line_plot(series, ax=ax, color=SCENARIO_COLORS[name],
                         width=2 if name == "hold" else 1.5)

    if ax is None:
        return

    # 90% HDI band for hold scenario
    if hold is not None:
        # Re-extract using the full samples (not just median)
        # We need the underlying DataFrame — extract from hold directly
        _add_hold_band(ax, hold, extract_fn)

    # Reference line
    if ref_line is not None:
        ax.axhline(y=ref_line, color="grey", linestyle="--",
                   linewidth=1, alpha=0.7, label=ref_label)

    mg.finalise_plot(
        ax,
        title=f"{title}{rate_str}",
        ylabel=ylabel,
        legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
        lheader=lheader,
        rheader="Scenarios assume RBA moves then holds.",
        lfooter="Australia.",
        rfooter=rfooter,
        show=show,
    )


def _add_hold_band(ax: Axes, hold: ForecastResults, extract_fn: Callable[[ForecastResults], pd.Series]) -> None:
    """Add 90% HDI shading for the hold scenario.

    We need to figure out which underlying DataFrame the extract_fn
    uses and compute quantiles from it. We use a simple approach:
    try each candidate DataFrame.
    """
    # Try to match what extract_fn returns by checking the median
    median = extract_fn(hold)
    candidates = [
        annualize(hold.inflation_forecast),
        hold.unemployment_forecast,
        hold.output_gap_forecast,
        annualize(hold.gdp_growth_forecast()),
    ]
    for df in candidates:
        if df.median(axis=1).round(6).equals(median.round(6)):
            lower = df.quantile(0.05, axis=1)
            upper = df.quantile(0.95, axis=1)
            ax.fill_between(
                [p.ordinal for p in hold.forecast_index],
                lower.to_numpy(), upper.to_numpy(),
                alpha=0.15, color="grey", label="90% HDI (hold)",
            )
            return

    # Fallback: no band if we can't match


# --- Thin wrappers ---


def plot_scenario_inflation(
    scenario_results: dict[str, ForecastResults],
    chart_obs: pd.DataFrame | None = None,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Inflation scenarios."""
    # Historical trimmed mean
    if chart_obs is not None and "π" in chart_obs.columns:
        hist = annualize(chart_obs["π"].dropna())
    else:
        hist = annualize(get_trimmed_mean_qrtly().data)

    _plot_scenarios(
        scenario_results,
        extract_fn=lambda r: annualize(r.inflation_forecast).median(axis=1),
        hist_series=hist,
        title="Inflation: Policy Rate Scenarios",
        ylabel="Per cent per annum",
        lheader="Trimmed mean, annualised",
        rfooter=rfooter,
        ref_line=2.5,
        ref_label="Target (2.5%)",
        show=show,
    )


def plot_scenario_unemployment(
    scenario_results: dict[str, ForecastResults],
    obs: dict | None = None,
    obs_index: pd.PeriodIndex | None = None,
    chart_obs: pd.DataFrame | None = None,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Unemployment scenarios."""
    hist = None
    if chart_obs is not None and "U" in chart_obs.columns:
        hist = chart_obs["U"].dropna()
    elif obs is not None and obs_index is not None:
        hist = pd.Series(obs["U"], index=obs_index)

    hold = scenario_results.get("hold")
    nairu_ref = hold.nairu_final if hold else None

    _plot_scenarios(
        scenario_results,
        extract_fn=lambda r: r.unemployment_forecast.median(axis=1),
        hist_series=hist,
        title="Unemployment: Policy Rate Scenarios",
        ylabel="Per cent",
        lheader="Unemployment rate",
        rfooter=rfooter,
        ref_line=nairu_ref,
        ref_label=f"NAIRU ({nairu_ref:.1f}%)" if nairu_ref else "",
        show=show,
    )


def plot_scenario_gdp_growth(
    scenario_results: dict[str, ForecastResults],
    obs: dict | None = None,
    obs_index: pd.PeriodIndex | None = None,
    chart_obs: pd.DataFrame | None = None,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """GDP growth scenarios."""
    hist = None
    if chart_obs is not None and "log_gdp" in chart_obs.columns:
        hist = annualize(chart_obs["log_gdp"].dropna().diff())
    elif obs is not None and obs_index is not None:
        hist = annualize(pd.Series(obs["log_gdp"], index=obs_index).diff())

    hold = scenario_results.get("hold")
    pot_growth = annualize(hold.potential_growth) if hold else None

    _plot_scenarios(
        scenario_results,
        extract_fn=lambda r: annualize(r.gdp_growth_forecast()).median(axis=1),
        hist_series=hist,
        title="GDP Growth: Policy Rate Scenarios",
        ylabel="Per cent per annum",
        lheader="Annualised quarterly growth",
        rfooter=rfooter,
        ref_line=pot_growth,
        ref_label=f"Potential ({pot_growth:.1f}%)" if pot_growth else "",
        show=show,
    )


def plot_scenario_output_gap(
    scenario_results: dict[str, ForecastResults],
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Output gap scenarios."""
    _plot_scenarios(
        scenario_results,
        extract_fn=lambda r: r.output_gap_forecast.median(axis=1),
        title="Output Gap: Policy Rate Scenarios",
        ylabel="Per cent of potential",
        lheader="Output gap = GDP - Potential (log points)",
        rfooter=rfooter,
        ref_line=0.0,
        ref_label="Potential (OG=0)",
        show=show,
    )


def plot_output_vs_potential(
    scenario_results: dict[str, ForecastResults],
    obs: dict | None = None,
    obs_index: pd.PeriodIndex | None = None,
    rfooter: str = "",
    n_history: int = 8,
    show: bool = False,
) -> None:
    """GDP vs potential output (hold scenario only)."""
    hold = scenario_results.get("hold")
    if hold is None:
        return

    # Historical GDP
    ax = None
    if obs is not None and obs_index is not None:
        log_gdp_hist = pd.Series(obs["log_gdp"], index=obs_index).iloc[-n_history:]
        log_gdp_hist.name = "GDP (actual)"
        ax = mg.line_plot(log_gdp_hist, color="black", width=2)

    # Potential (median, dashed)
    potential_median = hold.potential_forecast.median(axis=1)
    potential_with_hist = pd.concat([
        pd.Series([hold.potential_final], index=[hold.obs_index[-1]]),
        potential_median,
    ])
    potential_with_hist.name = "Potential"
    if ax is None:
        ax = mg.line_plot(potential_with_hist, color="grey", width=2)
    else:
        mg.line_plot(potential_with_hist, ax=ax, color="grey", width=2)
    ax.lines[-1].set_linestyle("--")

    # Key scenarios
    colors = {"+200bp": "crimson", "hold": "black", "-200bp": "royalblue"}
    for name in ["+200bp", "hold", "-200bp"]:
        if name in scenario_results:
            r = scenario_results[name]
            gdp_median = r.output_samples().median(axis=1)
            gdp_with_hist = pd.concat([
                pd.Series([r.log_gdp_final], index=[r.obs_index[-1]]),
                gdp_median,
            ])
            gdp_with_hist.name = f"GDP ({name})"
            mg.line_plot(gdp_with_hist, ax=ax, color=colors[name], width=1.5)

    # 90% band for hold GDP
    gdp_hold = hold.output_samples()
    lower = gdp_hold.quantile(0.05, axis=1)
    upper = gdp_hold.quantile(0.95, axis=1)
    ax.fill_between(
        [p.ordinal for p in hold.forecast_index],
        lower.to_numpy(), upper.to_numpy(),
        alpha=0.15, color="grey", label="90% HDI (hold)",
    )

    current_rate = hold.cash_rate
    mg.finalise_plot(
        ax,
        title=f"GDP vs Potential: Policy Rate Scenarios (from {current_rate:.2f}%)",
        ylabel="Log GDP",
        legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
        lheader="GDP and potential output (log scale)",
        lfooter="Australia.",
        rfooter=rfooter,
        show=show,
    )


# --- Convenience entry point ---


def plot_all_scenarios(
    scenario_results: dict[str, ForecastResults],
    results: NAIRUResults,
    chart_dir: Path | str,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Generate all forecast charts."""
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    mg.set_chart_dir(str(chart_dir))

    plot_scenario_inflation(
        scenario_results, chart_obs=results.chart_obs,
        rfooter=rfooter, show=show,
    )
    plot_scenario_unemployment(
        scenario_results, obs=results.obs, obs_index=results.obs_index,
        chart_obs=results.chart_obs, rfooter=rfooter, show=show,
    )
    plot_scenario_gdp_growth(
        scenario_results, obs=results.obs, obs_index=results.obs_index,
        chart_obs=results.chart_obs, rfooter=rfooter, show=show,
    )
    plot_scenario_output_gap(scenario_results, rfooter=rfooter, show=show)
    plot_output_vs_potential(
        scenario_results, obs=results.obs, obs_index=results.obs_index,
        rfooter=rfooter, show=show,
    )

    print(f"Forecast charts saved to: {chart_dir}")
