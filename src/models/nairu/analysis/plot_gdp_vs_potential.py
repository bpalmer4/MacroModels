"""GDP vs potential output plot."""

import mgplot as mg
import pandas as pd

from src.models.common.extraction import get_vector_var
from src.models.common.timeseries import plot_posterior_timeseries
from src.models.nairu.results import NAIRUResults

START = pd.Period("1985Q1", freq="Q")


def plot_gdp_vs_potential(
    results: NAIRUResults,
    *,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Plot actual GDP against potential GDP estimates."""
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    ax = plot_posterior_timeseries(
        data=potential,
        legend_stem="Potential GDP",
        color="green",
        start=START,
        finalise=False,
    )

    actual = pd.Series(results.obs["log_gdp"], index=results.obs_index)
    actual.name = "Actual GDP"
    mg.line_plot(actual, ax=ax, color="black", width=1.5)

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Actual vs Potential GDP",
            ylabel="Log GDP (scaled)",
            legend={"loc": "upper left", "fontsize": "x-small"},
            lfooter="Australia. Log real GDP scaled by 100.",
            rfooter=rfooter,
            axisbelow=True,
            show=show,
        )
