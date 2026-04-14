"""Plot the inflation expectations input (π_exp) used by the model.

Renders the anchor series actually fed into the Phillips curves, overlaid
with observed annual trimmed-mean inflation and the 2.5% target, so the
reader can see exactly what drives the expectations term.
"""

import mgplot as mg
import pandas as pd

from src.models.nairu.results import NAIRUResults

START = pd.Period("1983Q1", freq="Q")

ANNUAL_RANGE = {
    "axhspan": {
        "ymin": 2,
        "ymax": 3,
        "color": "#dddddd",
        "label": "2-3% target range",
        "zorder": -1,
    }
}

ANNUAL_TARGET = {
    "axhline": {
        "y": 2.5,
        "linestyle": "dashed",
        "linewidth": 0.75,
        "color": "darkred",
        "label": "2.5% target",
    }
}


def plot_expectations_input(
    results: NAIRUResults,
    *,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Plot the π_exp series used by the model, with observed inflation overlay."""
    π_exp = pd.Series(results.obs["π_exp"], index=results.obs_index, name="π_exp (model input)")
    π4 = pd.Series(results.obs["π4"], index=results.obs_index, name="Trimmed mean inflation (through-the-year)")

    π_exp = π_exp[π_exp.index >= START]
    π4 = π4[π4.index >= START]

    ax = mg.line_plot(
        π_exp,
        color="navy",
        width=2,
        annotate=True,
    )
    mg.line_plot(π4, ax=ax, color="darkorange", width=1.2, annotate=True)

    mg.finalise_plot(
        ax,
        title="Inflation Expectations Input",
        ylabel="Per cent per annum",
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter=f"Australia. Series fed to the Phillips curves. {results.anchor_label}.",
        rfooter=rfooter,
        axisbelow=True,
        **ANNUAL_RANGE,
        **ANNUAL_TARGET,
        show=show,
    )
