"""Unified posterior time series plotting with finalisation."""

from collections.abc import Sequence
from typing import Any

import arviz as az
import mgplot as mg
import pandas as pd
from matplotlib.axes import Axes

from src.models.nairu.analysis.extraction import get_vector_var


def plot_posterior_timeseries(
    trace: az.InferenceData | None = None,
    var: str | None = None,
    index: pd.PeriodIndex | None = None,
    data: pd.DataFrame | None = None,
    legend_stem: str = "Model Estimate",
    color: str = "blue",
    start: pd.Period | None = None,
    cuts: Sequence[float] = (0.005, 0.025, 0.16),
    alphas: Sequence[float] = (0.1, 0.2, 0.3),
    finalise: bool = True,
    **finalise_kwargs: Any,
) -> Axes | None:
    """Plot posterior time series with credible intervals.

    Supports two modes:
    1. Direct from trace: provide (trace, var, index)
    2. Pre-computed DataFrame: provide data

    Args:
        trace: ArviZ InferenceData object
        var: Variable name to extract from trace
        index: PeriodIndex for the time series
        data: Pre-computed DataFrame of samples (time × draws)
        legend_stem: Stem for legend labels
        color: Color for the plot
        start: Start period for filtering
        cuts: Quantile cuts for credible intervals
        alphas: Alpha values for credible interval bands
        finalise: If True, call finalise_plot and return None.
            If False, return Axes for composition.
        **finalise_kwargs: Passed to mg.finalise_plot (title, lfooter, rfooter, etc.)

    Returns:
        Axes if finalise=False, None if finalise=True.

    """
    if len(cuts) != len(alphas):
        raise ValueError("Cuts and alphas must have the same length")

    if data is not None:
        samples = data
    elif trace is not None and var is not None and index is not None:
        samples = get_vector_var(var, trace)
        samples.index = index
    else:
        raise ValueError("Must provide either (trace, var, index) or data")

    if start is not None:
        samples = samples[samples.index >= start]

    ax = None
    for cut, alpha in zip(cuts, alphas):
        if not (0 < cut < 0.5):
            raise ValueError("Cuts must be between 0 and 0.5")

        lower = samples.quantile(q=cut, axis=1)
        upper = samples.quantile(q=1 - cut, axis=1)
        band = pd.DataFrame({"lower": lower, "upper": upper}, index=samples.index)
        ax = mg.fill_between_plot(
            band,
            ax=ax,
            color=color,
            alpha=alpha,
            label=f"{legend_stem} {int((1 - 2 * cut) * 100)}% Credible Interval",
            zorder=3,  # Above grid (zorder ~2.5)
        )

    median = samples.quantile(q=0.5, axis=1)
    median.name = f"{legend_stem} Median"
    ax = mg.line_plot(median, ax=ax, color=color, width=1, annotate=True, zorder=4)

    if finalise and ax is not None:
        finalise_kwargs.setdefault("axisbelow", True)
        mg.finalise_plot(ax, **finalise_kwargs)
        return None

    return ax
