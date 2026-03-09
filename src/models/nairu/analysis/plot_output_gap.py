"""Output gap plot."""

import numpy as np
import pandas as pd

from src.models.common.extraction import get_vector_var
from src.models.common.timeseries import plot_posterior_timeseries
from src.models.nairu.results import NAIRUResults

START = pd.Period("1985Q1", freq="Q")


def plot_output_gap(
    results: NAIRUResults,
    *,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Plot the output gap as percentage deviation from potential."""
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index
    actual_gdp = results.obs["log_gdp"]
    output_gap = actual_gdp[:, np.newaxis] - potential.to_numpy()
    output_gap = pd.DataFrame(output_gap, index=results.obs_index)

    plot_posterior_timeseries(
        data=output_gap,
        legend_stem="Output Gap",
        color="green",
        start=START,
        title="Output Gap Estimate for Australia",
        ylabel="Per cent of potential GDP",
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter="Australia. log Y − log Y* (log points × 100 ≈ per cent). Positive = overheating.",
        rfooter=rfooter,
        axisbelow=True,
        y0=True,
        show=show,
    )
