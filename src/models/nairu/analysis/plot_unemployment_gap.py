"""Unemployment gap plot."""

import pandas as pd

from src.models.nairu.observations import PHASE_END
from src.models.common.extraction import get_vector_var
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


def plot_unemployment_gap(
    results,
    *,
    rfooter: str = "",
    show: bool = False,
    verbose: bool = False,
) -> None:
    """Plot the unemployment gap (U - U*)."""
    nairu = get_vector_var("nairu", results.trace)
    nairu.index = results.obs_index
    U = pd.Series(results.obs["U"], index=results.obs_index)
    u_gap = nairu.apply(lambda col: U - col)
    if verbose:
        print("Last data point:", u_gap.index[-1])

    lfooter = rf"Australia. $U\text{{-}}gap = U - U^*$. {results.anchor_label}."
    plot_posterior_timeseries(
        data=u_gap,
        legend_stem="Unemployment Gap",
        color="darkred",
        start=START,
        title="Unemployment Gap Estimate for Australia",
        ylabel="Percentage points (U - U*)",
        lfooter=lfooter,
        rfooter=rfooter,
        legend={"loc": "best", "fontsize": "x-small"},
        axisbelow=True,
        y0=True,
        show=show,
        **PRE_POLICY_PERIOD,
    )
