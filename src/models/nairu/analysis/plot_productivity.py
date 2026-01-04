"""Productivity time series plots for derived MFP and Labour Productivity.

Uses productivity measures from the data layer (src/data/productivity.py).
"""

from typing import Any

import mgplot as mg
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from src.data import get_labour_productivity_growth, get_mfp_growth, hma
from src.models.nairu.equations import REGIME_COVID_START, REGIME_GFC_START
from src.utilities.rate_conversion import annualize

HMA_TERM = 41  # Henderson MA smoothing term
HP_LAMBDA = 1600  # Hodrick-Prescott smoothing parameter for quarterly data


def _apply_smoothing(
    series: pd.Series,
    filter_type: str,
) -> tuple[pd.Series, str]:
    """Apply HP or Henderson smoothing to a series.

    Args:
        series: Annual growth rate series
        filter_type: "henderson" or "hp"

    Returns:
        Tuple of (smoothed series, filter label for chart)

    """
    clean = series.dropna()
    if filter_type == "hp":
        _, trend = hpfilter(clean.values, lamb=HP_LAMBDA)
        smoothed = pd.Series(trend, index=clean.index)
        label = f"HP filter (λ={HP_LAMBDA})"
    else:
        smoothed = hma(clean, HMA_TERM).reindex(series.index)
        label = f"Henderson {HMA_TERM}-term MA"
    return smoothed, label


def _add_period_averages(ax, series: pd.Series) -> None:
    """Add period average annotation box to axes."""
    pre_gfc = series.index < REGIME_GFC_START
    post_gfc = (series.index >= REGIME_GFC_START) & (series.index < REGIME_COVID_START)
    post_covid = series.index >= REGIME_COVID_START

    avg_text = "Period averages:\n"
    if pre_gfc.any() and not np.isnan(series[pre_gfc].mean()):
        avg_text += f"  Pre-GFC: {series[pre_gfc].mean():.1f}%\n"
    if post_gfc.any() and not np.isnan(series[post_gfc].mean()):
        avg_text += f"  Post-GFC: {series[post_gfc].mean():.1f}%\n"
    if post_covid.any() and not np.isnan(series[post_covid].mean()):
        avg_text += f"  Post-COVID: {series[post_covid].mean():.1f}%"

    ax.annotate(
        avg_text,
        xy=(0.02, 0.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
    )


def _add_regime_lines(ax) -> None:
    """Add regime boundary vertical lines to axes."""
    ax.axvline(x=REGIME_GFC_START.ordinal, color="darkred", linestyle="--",
               linewidth=1, alpha=0.5, label=f"Regime change: GFC ({REGIME_GFC_START})")
    ax.axvline(x=REGIME_COVID_START.ordinal, color="darkgreen", linestyle="-.",
               linewidth=1, alpha=0.5, label=f"Regime change: COVID ({REGIME_COVID_START})")


def plot_labour_productivity(
    ulc_growth: pd.Series,
    hcoe_growth: pd.Series,
    filter_type: str = "hp",
    model_name: str = "Model",
    show: bool = False,
    **kwargs: Any,
) -> None:
    """Plot derived Labour Productivity time series.

    Uses data layer: LP = Δhcoe - Δulc

    Args:
        ulc_growth: Quarterly ULC growth rate
        hcoe_growth: Quarterly hourly COE growth rate
        filter_type: "henderson" or "hp" for Hodrick-Prescott
        model_name: Name for chart footer
        show: Whether to display the plot
        **kwargs: Additional arguments passed to finalise_plot

    """
    # Get LP from data layer
    lp = get_labour_productivity_growth(ulc_growth, hcoe_growth).data
    lp_annual = annualize(lp)

    # Apply smoothing
    lp_smoothed, filter_label = _apply_smoothing(lp_annual, filter_type)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        "Original": lp_annual,
        filter_label: lp_smoothed,
    })

    # Plot with mgplot
    ax = mg.line_plot(
        plot_data,
        color=["steelblue", "steelblue"],
        width=[0.8, 2],
        alpha=[0.4, 1.0],
    )
    ax.axhline(0, color="black", linewidth=0.5)

    _add_period_averages(ax, lp_annual)
    _add_regime_lines(ax)

    defaults = {
        "title": f"Labour Productivity Growth (Derived) - {filter_label}",
        "ylabel": "Annual growth (%)",
        "legend": {"loc": "upper right", "fontsize": "small"},
        "lfooter": f"LP = Δhcoe - Δulc. {filter_label}. HCOE = Hourly COE; ULC = Unit Labour Costs.",
        "rfooter": model_name,
    }
    for key in list(defaults.keys()):
        if key in kwargs:
            defaults.pop(key)

    mg.finalise_plot(ax, **defaults, **kwargs, show=show)


def plot_mfp(
    ulc_growth: pd.Series,
    hcoe_growth: pd.Series,
    capital_growth: pd.Series,
    hours_growth: pd.Series,
    alpha: float | pd.Series,
    filter_type: str = "hp",
    model_name: str = "Model",
    show: bool = False,
    **kwargs: Any,
) -> None:
    """Plot derived MFP time series.

    Uses data layer: MFP = LP - α × (g_K - g_L)

    Args:
        ulc_growth: Quarterly ULC growth rate
        hcoe_growth: Quarterly hourly COE growth rate
        capital_growth: Quarterly capital stock growth rate
        hours_growth: Quarterly hours worked growth rate
        alpha: Capital share of income (fixed float or time-varying Series)
        filter_type: "henderson" or "hp" for Hodrick-Prescott
        model_name: Name for chart footer
        show: Whether to display the plot
        **kwargs: Additional arguments passed to finalise_plot

    """
    # Get MFP from data layer
    mfp = get_mfp_growth(ulc_growth, hcoe_growth, capital_growth, hours_growth, alpha).data
    mfp_annual = annualize(mfp)

    # Apply smoothing
    mfp_smoothed, filter_label = _apply_smoothing(mfp_annual, filter_type)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        "Original": mfp_annual,
        filter_label: mfp_smoothed,
    })

    # Plot with mgplot
    ax = mg.line_plot(
        plot_data,
        color=["darkorange", "darkorange"],
        width=[0.8, 2],
        alpha=[0.4, 1.0],
    )
    ax.axhline(0, color="black", linewidth=0.5)

    _add_period_averages(ax, mfp_annual)
    _add_regime_lines(ax)

    alpha_desc = "time-varying α" if isinstance(alpha, pd.Series) else f"α = {alpha:.2f}"
    defaults = {
        "title": f"Multi-Factor Productivity Growth (Derived) - {filter_label}",
        "ylabel": "Annual growth (%)",
        "legend": {"loc": "upper right", "fontsize": "small"},
        "lfooter": f"MFP = LP - α×(g_K - g_L). {alpha_desc}. {filter_label}.",
        "rfooter": model_name,
    }
    for key in list(defaults.keys()):
        if key in kwargs:
            defaults.pop(key)

    mg.finalise_plot(ax, **defaults, **kwargs, show=show)


def plot_productivity_comparison(
    ulc_growth: pd.Series,
    hcoe_growth: pd.Series,
    capital_growth: pd.Series,
    hours_growth: pd.Series,
    alpha: float | pd.Series,
    filter_type: str = "hp",
    model_name: str = "Model",
    show: bool = False,
    **kwargs: Any,
) -> None:
    """Plot Labour Productivity and MFP together for comparison.

    Uses data layer for both LP and MFP derivation.

    Args:
        ulc_growth: Quarterly ULC growth rate
        hcoe_growth: Quarterly hourly COE growth rate
        capital_growth: Quarterly capital stock growth rate
        hours_growth: Quarterly hours worked growth rate
        alpha: Capital share of income (fixed float or time-varying Series)
        filter_type: "henderson" or "hp" for Hodrick-Prescott
        model_name: Name for chart footer
        show: Whether to display the plot
        **kwargs: Additional arguments passed to finalise_plot

    """
    # Get productivity from data layer
    lp = get_labour_productivity_growth(ulc_growth, hcoe_growth).data
    mfp = get_mfp_growth(ulc_growth, hcoe_growth, capital_growth, hours_growth, alpha).data

    # Annualize
    lp_annual = annualize(lp)
    mfp_annual = annualize(mfp)

    # Apply smoothing
    lp_smoothed, filter_label = _apply_smoothing(lp_annual, filter_type)
    mfp_smoothed, _ = _apply_smoothing(mfp_annual, filter_type)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        "Labour Productivity": lp_smoothed,
        "MFP (Solow Residual)": mfp_smoothed,
    })

    # Plot with mgplot
    ax = mg.line_plot(
        plot_data,
        color=["steelblue", "darkorange"],
        width=2,
    )
    ax.axhline(0, color="black", linewidth=0.5)

    _add_regime_lines(ax)

    alpha_desc = "time-varying α" if isinstance(alpha, pd.Series) else f"α = {alpha:.2f}"
    defaults = {
        "title": f"Productivity Growth Comparison (Derived) - {filter_label}",
        "ylabel": "Annual growth (%)",
        "legend": {"loc": "upper right", "fontsize": "small"},
        "lfooter": f"LP = Δhcoe - Δulc; MFP = LP - α×(g_K - g_L). {alpha_desc}. {filter_label}.",
        "rfooter": model_name,
    }
    for key in list(defaults.keys()):
        if key in kwargs:
            defaults.pop(key)

    mg.finalise_plot(ax, **defaults, **kwargs, show=show)
