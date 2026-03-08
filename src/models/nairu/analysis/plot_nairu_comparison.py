"""NAIRU comparison across model variants."""

import mgplot as mg
import pandas as pd

from src.models.common.extraction import get_vector_var

START = pd.Period("2000Q1", freq="Q")
COLORS = ["darkblue", "darkred", "darkgreen", "darkorange"]


def plot_nairu_comparison(
    results_list: list,
    *,
    chart_dir: str | None = None,
    show: bool = False,
) -> None:
    """Plot NAIRU posterior medians from multiple model variants."""
    if len(results_list) < 2:
        return

    # Chart directory from variant names
    if chart_dir is None:
        names = "_".join(r.config.label for r in results_list)
        chart_dir = f"charts/nairu_{names}"
    mg.set_chart_dir(chart_dir)

    medians = {}
    for results in results_list:
        samples = get_vector_var("nairu", results.trace)
        samples.index = results.obs_index
        median = samples.quantile(0.5, axis=1)
        medians[results.config.rfooter] = median[median.index >= START]

    ax = None
    for i, (label, s) in enumerate(medians.items()):
        s = s.copy()
        s.name = label
        ax = mg.line_plot(s, ax=ax, color=COLORS[i % len(COLORS)], width=1.5, annotate=True, zorder=4)

    if len(medians) == 2:
        keys = list(medians.keys())
        idx = medians[keys[0]].index.intersection(medians[keys[1]].index)
        band = pd.DataFrame({
            "lower": medians[keys[0]][idx].clip(upper=medians[keys[1]][idx]),
            "upper": medians[keys[0]][idx].clip(lower=medians[keys[1]][idx]),
        }, index=idx)
        mg.fill_between_plot(band, ax=ax, color="purple", alpha=0.15, label="NAIRU range")

    # Unemployment overlay
    U = pd.Series(results_list[0].obs["U"], index=results_list[0].obs_index)
    U = U[U.index >= START]
    U.name = "Unemployment Rate"
    mg.line_plot(U, ax=ax, color="brown", width=1.0, zorder=3)

    rfooter = "NAIRU-" + "-".join(r.config.label for r in results_list)
    mg.finalise_plot(
        ax,
        title="NAIRU Estimate Comparison",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter="Australia. Posterior median from each model variant.",
        rfooter=rfooter,
        axisbelow=True,
        show=show,
    )
