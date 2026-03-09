"""Potential GDP growth rate plot."""

import mgplot as mg
import numpy as np
import pandas as pd
from scipy import stats

from src.models.common.extraction import get_vector_var
from src.models.common.timeseries import plot_posterior_timeseries
from src.models.nairu.results import NAIRUResults

START = pd.Period("1985Q1", freq="Q")


def plot_potential_growth(
    results: NAIRUResults,
    *,
    rfooter: str = "",
    show: bool = False,
    verbose: bool = False,
) -> None:
    """Plot annual potential GDP growth (4Q difference of log potential)."""
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index
    potential_growth = potential.diff(4).dropna()

    ax = plot_posterior_timeseries(
        data=potential_growth,
        legend_stem="Potential Growth",
        color="purple",
        start=START,
        finalise=False,
    )

    median = potential_growth.quantile(0.5, axis=1)
    x = np.arange(len(median))
    slope, intercept, *_ = stats.linregress(x, median.to_numpy())
    trend = pd.Series(intercept + slope * x, index=median.index)
    trend.name = f"Trend (slope: {slope * 4:.2f}pp/year)"
    mg.line_plot(trend, ax=ax, color="darkred", width=1.5, style="--")

    if verbose:
        print("Potential Growth:")
        print(f"  Median endpoint: {median.iloc[-1]:.3f}% at {median.index[-1]}")
        print(f"  Trend endpoint: {trend.iloc[-1]:.3f}%")

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Potential GDP Growth Rate",
            ylabel="Per cent per annum",
            legend={"loc": "upper right", "fontsize": "x-small"},
            lfooter="Australia. 4-quarter change in log potential GDP (Cobb-Douglas production function).",
            rfooter=rfooter,
            axisbelow=True,
            y0=True,
            show=show,
        )
