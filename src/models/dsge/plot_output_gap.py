"""Plot output gap estimates from any DSGE model."""


import mgplot as mg
import pandas as pd

EXCLUDED_PERIOD = {
    "axvspan": {
        "xmin": pd.Period("2008Q4", freq="Q").ordinal,
        "xmax": pd.Period("2020Q4", freq="Q").ordinal,
        "color": "red",
        "alpha": 0.1,
        "label": "Excluded from estimation (GFC-COVID)",
        "zorder": -1,
    }
}


def plot_output_gap(
    output_gap: pd.Series,
    model_name: str = "DSGE",
    show: bool = False,
) -> None:
    """Plot output gap estimate.

    Args:
        output_gap: Series with PeriodIndex
        model_name: Model name for title/footer
        show: Whether to display plot

    """
    series = output_gap.copy()
    series.name = "Output Gap"

    ax = mg.line_plot(series, color="blue", width=1.5, annotate=True)
    mg.finalise_plot(
        ax,
        title=f"Output Gap - {model_name}",
        ylabel="Per cent of potential",
        lfooter="Australia. Two-stage estimation: parameters from non-crisis periods, states for full sample.",
        rfooter=f"{model_name} (MLE)",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        show=show,
        **EXCLUDED_PERIOD,
    )
