"""Potential growth smoothing comparison plot."""

import mgplot as mg
import numpy as np
import pandas as pd
from scipy import stats

from src.models.common.extraction import get_vector_var
from src.models.nairu.results import NAIRUResults

START = pd.Period("1985Q1", freq="Q")


def plot_potential_growth_smoothing(
    results: NAIRUResults,
    *,
    r_star_trend_weight: float = 0.75,
    rfooter: str = "",
    show: bool = False,
    verbose: bool = False,
) -> None:
    """Plot potential growth smoothing comparison (raw, trend, hybrid)."""
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index
    potential_growth = potential.diff(4).dropna()

    median = potential_growth.quantile(0.5, axis=1)
    x = np.arange(len(median))
    slope, intercept, *_ = stats.linregress(x, median.to_numpy())
    trend = pd.Series(intercept + slope * x, index=median.index)

    w = r_star_trend_weight
    hybrid = (1 - w) * median + w * trend

    if verbose:
        print("Potential Growth Smoothing Comparison:")
        print(f"  Median (raw) endpoint: {median.iloc[-1]:.3f}%")
        print(f"  Trend endpoint: {trend.iloc[-1]:.3f}%")
        print(f"  Hybrid ({int(w*100)}% trend, {int((1-w)*100)}% raw) endpoint: {hybrid.iloc[-1]:.3f}%")
        check_val = (1-w)*median.iloc[-1] + w*trend.iloc[-1]
        print(f"  Check: {(1-w):.2f}×{median.iloc[-1]:.3f} + {w:.2f}×{trend.iloc[-1]:.3f} = {check_val:.3f}%")

    median.name = "Potential growth (raw median)"
    trend.name = "Trend only"
    hybrid.name = f"Hybrid ({int(w*100)}% trend, {int((1-w)*100)}% raw)"

    ax = mg.line_plot(median, color="darkblue", width=1)
    mg.line_plot(trend, ax=ax, style="--", color="darkorange", width=1)
    mg.line_plot(hybrid, ax=ax, width=2, color="darkred", annotate=True)

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Potential GDP Growth - Smoothing Comparison",
            ylabel="Per cent per annum",
            legend={"loc": "upper right", "fontsize": "x-small"},
            lfooter="Australia. Raw posterior median potential growth vs linear trend vs hybrid.",
            rfooter=rfooter,
            axisbelow=True,
            y0=True,
            show=show,
        )
