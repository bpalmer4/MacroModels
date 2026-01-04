"""Plot r* (natural rate of interest) estimates from any DSGE model."""

import mgplot as mg
import pandas as pd

from src.models.dsge.plot_output_gap import EXCLUDED_PERIOD


def plot_rstar(
    rstar: pd.Series,
    model_name: str = "DSGE",
    show: bool = False,
) -> None:
    """Plot r* estimate.

    Args:
        rstar: Series with PeriodIndex
        model_name: Model name for title/footer
        show: Whether to display plot

    """
    series = rstar.copy()
    series.name = "r* (Natural Rate)"

    ax = mg.line_plot(series, color="blue", width=1.5, annotate=True)
    mg.finalise_plot(
        ax,
        title=f"r* (Natural Rate of Interest) - {model_name}",
        ylabel="Per cent (annual)",
        lfooter="Australia. Two-stage estimation: parameters from non-crisis periods, states for full sample.",
        rfooter=f"{model_name} (MLE)",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        show=show,
        **EXCLUDED_PERIOD,
    )
