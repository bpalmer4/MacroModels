"""Output gap and potential GDP plotting functions."""

import mgplot as mg
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.extraction import get_vector_var
from src.analysis.plot_posterior_timeseries import plot_posterior_timeseries

# Plotting constants
START = pd.Period("1985Q1", freq="Q")
RFOOTER = "Joint NAIRU + Output Gap Model"


def plot_output_gap(
    results,  # NAIRUResults - avoid circular import
    show: bool = False,
) -> None:
    """Plot the output gap as percentage deviation from potential."""
    # Get potential output samples and calculate output gap for each sample
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    # Calculate output gap: (Y - Y*)/Y* * 100
    actual_gdp = results.obs["log_gdp"]
    output_gap = (actual_gdp[:, np.newaxis] - potential.values) / potential.values * 100
    output_gap = pd.DataFrame(output_gap, index=results.obs_index)

    plot_posterior_timeseries(
        data=output_gap,
        legend_stem="Output Gap",
        color="green",
        start=START,
        title="Output Gap Estimate for Australia",
        ylabel="Per cent of potential GDP",
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter="Australia. (log Y - log Y*) / log Y* × 100. Positive = overheating/inflationary.",
        rfooter=RFOOTER,
        axisbelow=True,
        y0=True,
        show=show,
    )


def plot_gdp_vs_potential(
    results,  # NAIRUResults - avoid circular import
    show: bool = False,
) -> None:
    """Plot actual GDP against potential GDP estimates."""
    # Get potential output samples
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    # Plot potential GDP using shared time series function
    ax = plot_posterior_timeseries(
        data=potential,
        legend_stem="Potential GDP",
        color="green",
        start=START,
        finalise=False,
    )

    # Plot actual GDP on top
    actual = pd.Series(results.obs["log_gdp"], index=results.obs_index)
    actual = actual.reindex(results.obs_index)
    actual.name = "Actual GDP"
    mg.line_plot(
        actual,
        ax=ax,
        color="black",
        width=1.5,
    )

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Actual vs Potential GDP",
            ylabel="Log GDP (scaled)",
            legend={"loc": "upper left", "fontsize": "x-small"},
            lfooter="Australia. Log real GDP scaled by 100. ",
            rfooter=RFOOTER,
            axisbelow=True,
            show=show,
        )


def plot_potential_growth(
    results,  # NAIRUResults - avoid circular import
    r_star_trend_weight: float = 0.75,
    show: bool = False,
    verbose: bool = False,
) -> None:
    """Plot annual potential GDP growth (4Q difference of log potential).

    This serves as a proxy for r* (the natural rate of interest), based on
    the theoretical relationship r* ≈ trend real GDP growth.
    """
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    # r* = annual potential growth
    r_star = potential.diff(4).dropna()

    # Plot 1: Potential growth with credible intervals
    ax = plot_posterior_timeseries(
        data=r_star,
        legend_stem="Potential Growth",
        color="purple",
        start=START,
        finalise=False,
    )

    # Add trend line
    median = r_star.quantile(0.5, axis=1)
    x = np.arange(len(median))
    slope, intercept, *_ = stats.linregress(x, median.values)
    trend = pd.Series(intercept + slope * x, index=median.index)
    trend.name = f"Trend (slope: {slope * 4:.2f}pp/year)"
    mg.line_plot(trend, ax=ax, color="darkred", width=1.5, style="--")

    if verbose:
        print("Chart 1 - Potential Growth:")
        print(f"  Median endpoint: {median.iloc[-1]:.3f}% at {median.index[-1]}")
        print(f"  Trend endpoint: {trend.iloc[-1]:.3f}%")

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Potential GDP Growth Rate (proxy for $r^*$)",
            ylabel="Per cent per annum",
            legend={"loc": "upper right", "fontsize": "x-small"},
            lfooter="Australia. 4-quarter change in log potential GDP. r* ≈ trend growth.",
            rfooter=RFOOTER,
            axisbelow=True,
            y0=True,
            show=show,
        )

    # Plot 2: r* smoothing comparison
    w = r_star_trend_weight
    hybrid = (1 - w) * median + w * trend

    if verbose:
        print("\nChart 2 - r* Comparison:")
        print(f"  Median (raw) endpoint: {median.iloc[-1]:.3f}%")
        print(f"  Trend endpoint: {trend.iloc[-1]:.3f}%")
        print(f"  Hybrid ({int(w*100)}% trend, {int((1-w)*100)}% raw) endpoint: {hybrid.iloc[-1]:.3f}%")
        print(f"  Check: {(1-w):.2f} × {median.iloc[-1]:.3f} + {w:.2f} × {trend.iloc[-1]:.3f} = {(1-w)*median.iloc[-1] + w*trend.iloc[-1]:.3f}%")

    median.name = "$r^*$ raw median (no smoothing)"
    trend.name = "Trend only"
    hybrid.name = f"Hybrid ({int(w*100)}% trend, {int((1-w)*100)}% raw)"

    ax = mg.line_plot(median, color="darkblue", width=1)
    mg.line_plot(trend, ax=ax, style="--", color="darkorange", width=1)
    mg.line_plot(hybrid, ax=ax, width=2, color="darkred", annotate=True)

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Natural Rate of Interest (r*) - Comparison",
            ylabel="Per cent per annum",
            legend={"loc": "upper right", "fontsize": "x-small"},
            lfooter="Australia. Raw model median vs linear trend vs hybrid.",
            rfooter=RFOOTER,
            axisbelow=True,
            y0=True,
            show=show,
        )
