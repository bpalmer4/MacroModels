"""NAIRU estimate plot."""

import mgplot as mg
import pandas as pd

from src.data.observations import PHASE_END
from src.models.common.timeseries import plot_posterior_timeseries

START = pd.Period("1985Q1", freq="Q")

PRE_POLICY_PERIOD = {
    "axvspan": {
        "xmin": START.ordinal,
        "xmax": PHASE_END.ordinal,
        "color": "goldenrod",
        "alpha": 0.2,
        "label": "Pre-policy-relevant NAIRU",
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
    results,
    *,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Plot the NAIRU with unemployment and inflation overlay."""
    ax = plot_posterior_timeseries(
        trace=results.trace,
        var="nairu",
        index=results.obs_index,
        legend_stem="NAIRU",
        color="blue",
        start=START,
        finalise=False,
    )

    U = pd.Series(results.obs["U"], index=results.obs_index)
    U = U[U.index >= START]
    π4 = pd.Series(results.obs["π4"], index=results.obs_index)
    π4 = π4[π4.index >= START]

    back, front = 3, 1.5
    for color, width, label in zip(["white", ""], [back, front], ["_", ""]):
        U.name = "Unemployment Rate" if not label else label
        mg.line_plot(U, ax=ax, color=color if color else "brown", width=width, zorder=4)
        π4.name = "Inflation rate (trimmed mean, annual)" if not label else label
        mg.line_plot(π4, ax=ax, color=color if color else "darkorange", width=width, zorder=4)

    if ax is not None:
        lfooter = rf"Australia. $NAIRU = U^*$. {results.anchor_label}."
        mg.finalise_plot(
            ax,
            title="NAIRU Estimate for Australia",
            ylabel="Per cent",
            legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
            lfooter=lfooter,
            rfooter=rfooter,
            axisbelow=True,
            **PRE_POLICY_PERIOD,
            **ANNUAL_RANGE,
            **ANNUAL_TARGET,
            show=show,
        )
