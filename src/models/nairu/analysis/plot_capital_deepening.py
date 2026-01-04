"""Capital deepening/shallowing plot."""

from typing import Any

import mgplot as mg
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from src.models.nairu.equations import REGIME_COVID_START, REGIME_GFC_START
from src.utilities.rate_conversion import annualize

HP_LAMBDA = 1600


def plot_capital_deepening(
    capital_growth: pd.Series,
    hours_growth: pd.Series,
    model_name: str = "Model",
    show: bool = False,
    **kwargs: Any,
) -> None:
    """Plot capital deepening (g_K - g_L).

    Positive = capital deepening (more capital per worker)
    Negative = capital shallowing (less capital per worker)

    Args:
        capital_growth: Capital stock growth (quarterly %)
        hours_growth: Hours worked growth (quarterly %)
        model_name: Model name for chart footer
        show: Display plot interactively
        **kwargs: Additional arguments passed to finalise_plot

    """
    # Align series
    common_idx = capital_growth.index.intersection(hours_growth.index)
    capital = capital_growth.loc[common_idx]
    hours = hours_growth.loc[common_idx]

    # Capital deepening (quarterly)
    deepening = capital - hours

    # Annualize
    deepening_annual = annualize(deepening)

    # HP filter for trend
    deepening_clean = deepening_annual.dropna()
    _, deepening_trend = hpfilter(deepening_clean, lamb=HP_LAMBDA)
    deepening_trend = pd.Series(deepening_trend, index=deepening_clean.index)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        "Original": deepening_annual,
        f"HP filter (λ={HP_LAMBDA})": deepening_trend,
    })

    # Plot with mgplot
    ax = mg.line_plot(
        plot_data,
        color=["steelblue", "steelblue"],
        width=[0.8, 2],
        alpha=[0.4, 1.0],
    )
    ax.axhline(0, color="black", linewidth=0.8)

    # Add regime lines
    ax.axvline(
        x=REGIME_GFC_START.ordinal, color="darkred", linestyle="--",
        linewidth=1, alpha=0.5, label=f"Regime change: GFC ({REGIME_GFC_START})"
    )
    ax.axvline(
        x=REGIME_COVID_START.ordinal, color="darkgreen", linestyle="-.",
        linewidth=1, alpha=0.5, label=f"Regime change: COVID ({REGIME_COVID_START})"
    )

    # Period averages
    pre_gfc = deepening_trend[deepening_trend.index < REGIME_GFC_START]
    post_gfc = deepening_trend[
        (deepening_trend.index >= REGIME_GFC_START) &
        (deepening_trend.index < REGIME_COVID_START)
    ]
    post_covid = deepening_trend[deepening_trend.index >= REGIME_COVID_START]

    avg_text = "Period averages:\n"
    if len(pre_gfc) > 0:
        avg_text += f"  Pre-GFC: {pre_gfc.mean():.2f}%\n"
    if len(post_gfc) > 0:
        avg_text += f"  Post-GFC: {post_gfc.mean():.2f}%\n"
    if len(post_covid) > 0:
        avg_text += f"  Post-COVID: {post_covid.mean():.2f}%"

    ax.annotate(
        avg_text,
        xy=(0.02, 0.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
    )

    defaults = {
        "title": "Capital Deepening: (Δlog(K) − Δlog(L)) × 100",
        "ylabel": "Annual growth (%)",
        "legend": {"loc": "upper right", "fontsize": "small"},
        "lfooter": f"Capital deepening = K growth − L growth. +ve = deepening. HP filter (λ={HP_LAMBDA}).",
        "rfooter": model_name,
    }
    for key in list(defaults.keys()):
        if key in kwargs:
            defaults.pop(key)

    mg.finalise_plot(ax, **defaults, **kwargs, show=show)
