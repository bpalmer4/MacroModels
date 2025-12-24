"""Productivity time series plots for derived MFP and Labour Productivity."""

from typing import Any

import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from src.analysis.rate_conversion import annualize
from src.data import hma
from src.equations import REGIME_COVID_START, REGIME_GFC_START

HMA_TERM = 41  # Henderson MA smoothing term
HP_LAMBDA = 1600  # Hodrick-Prescott smoothing parameter for quarterly data


def plot_labour_productivity(
    ulc_growth: pd.Series,
    hcoe_growth: pd.Series,
    filter_type: str = "henderson",
    model_name: str = "Model",
    show: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Plot derived Labour Productivity time series.

    Labour Productivity = Δhcoe - Δulc (from wage data identity)

    Args:
        ulc_growth: Quarterly ULC growth rate
        hcoe_growth: Quarterly hourly COE growth rate
        filter_type: "henderson" or "hp" for Hodrick-Prescott
        model_name: Name for chart footer
        show: Whether to display the plot
        **kwargs: Additional arguments passed to finalise_plot
    """
    # Derive labour productivity
    labour_productivity = hcoe_growth - ulc_growth

    # Annualize for plotting (convert quarterly to annual rates)
    lp_annual = annualize(labour_productivity)

    # Apply smoothing
    lp_clean = lp_annual.dropna()
    if filter_type == "hp":
        _, lp_trend = hpfilter(lp_clean.values, lamb=HP_LAMBDA)
        lp_smoothed = pd.Series(lp_trend, index=lp_clean.index)
        filter_label = f"HP filter (λ={HP_LAMBDA})"
    else:
        lp_smoothed = hma(lp_clean, HMA_TERM).reindex(lp_annual.index)
        filter_label = f"Henderson {HMA_TERM}-term MA"

    # Create figure
    _, ax = plt.subplots(figsize=(10, 6))

    # Plot original (faded) and smoothed
    ax.plot(lp_annual.index.to_timestamp(), lp_annual.values,
            color="steelblue", linewidth=0.8, alpha=0.4, label="Original")
    ax.plot(lp_smoothed.index.to_timestamp(), lp_smoothed.values,
            color="steelblue", linewidth=2, label=filter_label)
    ax.axhline(0, color="black", linewidth=0.5)

    # Calculate period averages for annotation (from raw data)
    pre_gfc = lp_annual.index < REGIME_GFC_START
    post_gfc = (lp_annual.index >= REGIME_GFC_START) & (lp_annual.index < REGIME_COVID_START)
    post_covid = lp_annual.index >= REGIME_COVID_START

    avg_text = "Period averages:\n"
    if pre_gfc.any() and not np.isnan(lp_annual[pre_gfc].mean()):
        avg_text += f"  Pre-GFC: {lp_annual[pre_gfc].mean():.1f}%\n"
    if post_gfc.any() and not np.isnan(lp_annual[post_gfc].mean()):
        avg_text += f"  Post-GFC: {lp_annual[post_gfc].mean():.1f}%\n"
    if post_covid.any() and not np.isnan(lp_annual[post_covid].mean()):
        avg_text += f"  Post-COVID: {lp_annual[post_covid].mean():.1f}%"

    ax.annotate(
        avg_text,
        xy=(0.02, 0.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
    )

    # Add regime boundary lines
    ax.axvline(x=REGIME_GFC_START.to_timestamp(), color="darkred", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Regime change: GFC ({REGIME_GFC_START})")
    ax.axvline(x=REGIME_COVID_START.to_timestamp(), color="darkgreen", linestyle="-.", linewidth=1, alpha=0.5,
               label=f"Regime change: COVID ({REGIME_COVID_START})")

    defaults = {
        "title": f"Labour Productivity Growth (Derived) - {filter_label}",
        "ylabel": "Annual growth (%)",
        "legend": {"loc": "upper right", "fontsize": "small"},
        "lfooter": f"LP = Δhcoe - Δulc. {filter_label}. HCOE = Hourly Compensation of Employees; ULC = Unit Labour Costs.",
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
    alpha: float = 0.25,
    filter_type: str = "henderson",
    model_name: str = "Model",
    show: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Plot derived MFP time series.

    MFP = Labour Productivity - α × Capital Deepening
    Capital Deepening = g_K - g_L

    Note: ABS MFP estimates segment labour quality, so direct comparison
    is limited. This derivation captures raw productivity growth.

    Args:
        ulc_growth: Quarterly ULC growth rate
        hcoe_growth: Quarterly hourly COE growth rate
        capital_growth: Quarterly capital stock growth rate
        hours_growth: Quarterly hours worked growth rate
        alpha: Capital share of income (default 0.25)
        filter_type: "henderson" or "hp" for Hodrick-Prescott
        model_name: Name for chart footer
        show: Whether to display the plot
        **kwargs: Additional arguments passed to finalise_plot
    """
    # Derive productivity measures
    labour_productivity = hcoe_growth - ulc_growth
    capital_deepening = capital_growth - hours_growth
    mfp = labour_productivity - alpha * capital_deepening

    # Annualize for plotting
    mfp_annual = annualize(mfp)

    # Apply smoothing
    mfp_clean = mfp_annual.dropna()
    if filter_type == "hp":
        _, mfp_trend = hpfilter(mfp_clean.values, lamb=HP_LAMBDA)
        mfp_smoothed = pd.Series(mfp_trend, index=mfp_clean.index)
        filter_label = f"HP filter (λ={HP_LAMBDA})"
    else:
        mfp_smoothed = hma(mfp_clean, HMA_TERM).reindex(mfp_annual.index)
        filter_label = f"Henderson {HMA_TERM}-term MA"

    # Create figure
    _, ax = plt.subplots(figsize=(10, 6))

    # Plot original (faded) and smoothed
    ax.plot(mfp_annual.index.to_timestamp(), mfp_annual.values,
            color="darkorange", linewidth=0.8, alpha=0.4, label="Original")
    ax.plot(mfp_smoothed.index.to_timestamp(), mfp_smoothed.values,
            color="darkorange", linewidth=2, label=filter_label)
    ax.axhline(0, color="black", linewidth=0.5)

    # Calculate period averages for annotation (from raw data)
    pre_gfc = mfp_annual.index < REGIME_GFC_START
    post_gfc = (mfp_annual.index >= REGIME_GFC_START) & (mfp_annual.index < REGIME_COVID_START)
    post_covid = mfp_annual.index >= REGIME_COVID_START

    avg_text = "Period averages:\n"
    if pre_gfc.any() and not np.isnan(mfp_annual[pre_gfc].mean()):
        avg_text += f"  Pre-GFC: {mfp_annual[pre_gfc].mean():.1f}%\n"
    if post_gfc.any() and not np.isnan(mfp_annual[post_gfc].mean()):
        avg_text += f"  Post-GFC: {mfp_annual[post_gfc].mean():.1f}%\n"
    if post_covid.any() and not np.isnan(mfp_annual[post_covid].mean()):
        avg_text += f"  Post-COVID: {mfp_annual[post_covid].mean():.1f}%"

    ax.annotate(
        avg_text,
        xy=(0.02, 0.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
    )

    # Add regime boundary lines
    ax.axvline(x=REGIME_GFC_START.to_timestamp(), color="darkred", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Regime change: GFC ({REGIME_GFC_START})")
    ax.axvline(x=REGIME_COVID_START.to_timestamp(), color="darkgreen", linestyle="-.", linewidth=1, alpha=0.5,
               label=f"Regime change: COVID ({REGIME_COVID_START})")

    defaults = {
        "title": f"Multi-Factor Productivity Growth (Derived) - {filter_label}",
        "ylabel": "Annual growth (%)",
        "legend": {"loc": "upper right", "fontsize": "small"},
        "lfooter": f"MFP = LP - α×(g_K - g_L). α = {alpha}. {filter_label}.",
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
    alpha: float = 0.25,
    filter_type: str = "henderson",
    model_name: str = "Model",
    show: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Plot Labour Productivity and MFP together for comparison.

    Args:
        ulc_growth: Quarterly ULC growth rate
        hcoe_growth: Quarterly hourly COE growth rate
        capital_growth: Quarterly capital stock growth rate
        hours_growth: Quarterly hours worked growth rate
        alpha: Capital share of income (default 0.25)
        filter_type: "henderson" or "hp" for Hodrick-Prescott
        model_name: Name for chart footer
        show: Whether to display the plot
        **kwargs: Additional arguments passed to finalise_plot
    """
    # Derive productivity measures
    labour_productivity = hcoe_growth - ulc_growth
    capital_deepening = capital_growth - hours_growth
    mfp = labour_productivity - alpha * capital_deepening

    # Annualize
    lp_annual = annualize(labour_productivity)
    mfp_annual = annualize(mfp)

    # Apply smoothing
    lp_clean = lp_annual.dropna()
    mfp_clean = mfp_annual.dropna()

    if filter_type == "hp":
        _, lp_trend = hpfilter(lp_clean.values, lamb=HP_LAMBDA)
        _, mfp_trend = hpfilter(mfp_clean.values, lamb=HP_LAMBDA)
        lp_smoothed = pd.Series(lp_trend, index=lp_clean.index)
        mfp_smoothed = pd.Series(mfp_trend, index=mfp_clean.index)
        filter_label = f"HP filter (λ={HP_LAMBDA})"
    else:
        lp_smoothed = hma(lp_clean, HMA_TERM).reindex(lp_annual.index)
        mfp_smoothed = hma(mfp_clean, HMA_TERM).reindex(mfp_annual.index)
        filter_label = f"Henderson {HMA_TERM}-term MA"

    # Create figure
    _, ax = plt.subplots(figsize=(10, 6))

    # Plot both smoothed series
    ax.plot(lp_smoothed.index.to_timestamp(), lp_smoothed.values,
            color="steelblue", linewidth=2, label="Labour Productivity")
    ax.plot(mfp_smoothed.index.to_timestamp(), mfp_smoothed.values,
            color="darkorange", linewidth=2, label="MFP (Solow Residual)")
    ax.axhline(0, color="black", linewidth=0.5)

    # Add regime boundary lines
    ax.axvline(x=REGIME_GFC_START.to_timestamp(), color="darkred", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Regime change: GFC ({REGIME_GFC_START})")
    ax.axvline(x=REGIME_COVID_START.to_timestamp(), color="darkgreen", linestyle="-.", linewidth=1, alpha=0.5,
               label=f"Regime change: COVID ({REGIME_COVID_START})")

    defaults = {
        "title": f"Productivity Growth Comparison (Derived) - {filter_label}",
        "ylabel": "Annual growth (%)",
        "legend": {"loc": "upper right", "fontsize": "small"},
        "lfooter": f"LP = Δhcoe - Δulc; MFP = LP - α×(g_K - g_L). α = {alpha}. {filter_label}.",
        "rfooter": model_name,
    }
    for key in list(defaults.keys()):
        if key in kwargs:
            defaults.pop(key)

    mg.finalise_plot(ax, **defaults, **kwargs, show=show)
