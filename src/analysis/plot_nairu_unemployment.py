"""NAIRU and unemployment gap plotting functions."""

import mgplot as mg
import pandas as pd

from src.analysis.extraction import get_vector_var
from src.analysis.plot_posterior_timeseries import plot_posterior_timeseries
from src.data.rba_loader import PI_TARGET_FULL, PI_TARGET_START

# Plotting constants
START = pd.Period("1985Q1", freq="Q")
RFOOTER = "Joint NAIRU + Output Gap Model"

# NAIRU warning region (before inflation target fully anchored)
NAIRU_WARN = {
    "axvspan": {
        "xmin": START.ordinal,
        "xmax": PI_TARGET_FULL.ordinal,
        "label": r"NAIRU ($U^*$) WRT $\pi^e$ (before inflation target fully anchored)",
        "color": "goldenrod",
        "alpha": 0.2,
        "zorder": -2,
    }
}

ANNUAL_RANGE = {
    "axhspan": {
        "ymin": 2,
        "ymax": 3,
        "color": "#dddddd",
        "label": "2-3% annual inflation target range",
        "zorder": -1,
    }
}

ANNUAL_TARGET = {
    "axhline": {
        "y": 2.5,
        "linestyle": "dashed",
        "linewidth": 0.75,
        "color": "darkred",
        "label": "2.5% annual inflation target",
    }
}


def plot_nairu(
    results,  # NAIRUResults - avoid circular import
    show: bool = False,
) -> None:
    """Plot the NAIRU with unemployment and inflation overlay."""
    # NAIRU with credible intervals
    ax = plot_posterior_timeseries(
        trace=results.trace,
        var="nairu",
        index=results.obs_index,
        legend_stem="NAIRU",
        color="blue",
        start=START,
        finalise=False,
    )

    # Unemployment and inflation overlay with white background trick
    U = pd.Series(results.obs["U"], index=results.obs_index)
    U = U[U.index >= START]
    π4 = pd.Series(results.obs["π4"], index=results.obs_index)
    π4 = π4[π4.index >= START]

    back, front = 3, 1.5
    for color, width, label in zip(["white", ""], [back, front], ["_", ""]):
        U.name = "Unemployment Rate" if not label else label
        mg.line_plot(U, ax=ax, color=color if color else "brown", width=width, zorder=4)
        π4.name = "Inflation rate" if not label else label
        mg.line_plot(π4, ax=ax, color=color if color else "darkorange", width=width, zorder=4)

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="NAIRU Estimate for Australia",
            ylabel="Per cent",
            legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
            lfooter=r"Australia. $NAIRU = U^*$ "
            f"WRT inflation expectations → {PI_TARGET_START} → blended to target → {PI_TARGET_FULL} → inflation target. ",
            rheader=f"From {PI_TARGET_FULL} onwards, the model estimates NAIRU as the unemployment rate needed to hit the inflation target.",
            rfooter=RFOOTER,
            axisbelow=True,
            **ANNUAL_RANGE,
            **ANNUAL_TARGET,
            **NAIRU_WARN,
            show=show,
        )


def plot_unemployment_gap(
    results,  # NAIRUResults - avoid circular import
    show: bool = False,
    verbose: bool = False,
) -> None:
    """Plot the unemployment gap (U - U*)."""
    # Get NAIRU samples and calculate unemployment gap for each sample
    nairu = get_vector_var("nairu", results.trace)
    nairu.index = results.obs_index
    U = pd.Series(results.obs["U"], index=results.obs_index)
    u_gap = nairu.apply(lambda col: U - col)
    if verbose:
        print("Last data point:", u_gap.index[-1])

    plot_posterior_timeseries(
        data=u_gap,
        legend_stem="Unemployment Gap",
        color="darkred",
        start=START,
        title="Unemployment Gap Estimate for Australia",
        ylabel="Percentage points (U - U*)",
        lfooter=r"Australia. $U\text{-}gap = U - U^*$. Positive = slack/disinflationary, Negative = tight/inflationary.",
        rfooter=RFOOTER,
        legend={"loc": "best", "fontsize": "x-small"},
        axisbelow=True,
        y0=True,
        **NAIRU_WARN,
        show=show,
    )
