"""Output gap comparison across model variants."""

import mgplot as mg
import numpy as np
import pandas as pd

from src.models.common.extraction import get_vector_var

START = pd.Period("2000Q1", freq="Q")
COLORS = ["darkblue", "darkorange", "darkgreen", "darkred"]


def plot_output_gap_comparison(
    results_list: list,
    *,
    chart_dir: str | None = None,
    show: bool = False,
) -> None:
    """Plot output gap posterior medians from multiple model variants."""
    if len(results_list) < 2:
        return

    if chart_dir is None:
        names = "_".join(r.config.label for r in results_list)
        chart_dir = f"charts/nairu_{names}"
    mg.set_chart_dir(chart_dir)

    medians = {}
    for results in results_list:
        potential = get_vector_var("potential_output", results.trace)
        potential.index = results.obs_index
        log_gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)
        gap = log_gdp.to_numpy()[:, np.newaxis] - potential.to_numpy()
        gap_df = pd.DataFrame(gap, index=results.obs_index, columns=potential.columns)
        median = gap_df.quantile(0.5, axis=1)
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
        mg.fill_between_plot(band, ax=ax, color="purple", alpha=0.15, label="Output gap range")

    rfooter = "NAIRU-" + "-".join(r.config.label for r in results_list)
    mg.finalise_plot(
        ax,
        title="Output Gap Estimate Comparison",
        ylabel="Per cent of potential GDP",
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter="Australia. Posterior median from each model variant.",
        rfooter=rfooter,
        axisbelow=True,
        y0=True,
        show=show,
    )
