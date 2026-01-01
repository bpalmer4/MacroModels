"""Plot NAIRU estimates from any DSGE model."""

import mgplot as mg
import pandas as pd

from src.models.dsge.plot_output_gap import EXCLUDED_PERIOD


def plot_nairu(
    nairu: pd.Series,
    unemployment: pd.Series | None = None,
    model_name: str = "DSGE",
    show: bool = False,
) -> None:
    """Plot NAIRU estimate, optionally with unemployment overlay.

    Args:
        nairu: NAIRU series with PeriodIndex
        unemployment: Optional unemployment rate to overlay
        model_name: Model name for title/footer
        show: Whether to display plot
    """
    series = nairu.copy()
    series.name = "NAIRU"

    ax = mg.line_plot(series, color="blue", width=1.5, annotate=True)

    if unemployment is not None:
        u_series = unemployment.copy()
        u_series.name = "Unemployment Rate"
        mg.line_plot(u_series, ax=ax, color="brown", width=1.5)

    mg.finalise_plot(
        ax,
        title=f"NAIRU - {model_name}",
        ylabel="Per cent",
        lfooter="Australia. Two-stage estimation: parameters from non-crisis periods, states for full sample.",
        rfooter=f"{model_name} (MLE)",
        legend={"loc": "best", "fontsize": "x-small"},
        show=show,
        **EXCLUDED_PERIOD,
    )
