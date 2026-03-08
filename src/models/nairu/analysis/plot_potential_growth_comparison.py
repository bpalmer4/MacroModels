"""Potential growth: deterministic input vs modeled output."""

import mgplot as mg
import pandas as pd

from src.data.henderson import hma
from src.models.common.extraction import get_vector_var

START = pd.Period("1985Q1", freq="Q")


def plot_potential_growth_comparison(
    results,
    *,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Compare deterministic r* (input) vs modeled potential growth."""
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index
    modeled_growth = potential.diff(4).dropna()
    modeled_median = modeled_growth.quantile(0.5, axis=1)

    det_r_star = pd.Series(results.obs["det_r_star"], index=results.obs_index)
    common_idx = modeled_median.index.intersection(det_r_star.index)
    common_idx = common_idx[common_idx >= START]

    modeled_smoothed = hma(modeled_median.dropna(), 13)
    modeled_smoothed = modeled_smoothed.reindex(common_idx)

    plot_data = pd.DataFrame({
        "Deterministic r* (Cobb-Douglas)": det_r_star.loc[common_idx],
        "Modeled Potential Growth": modeled_median.loc[common_idx],
        "Modeled Potential Growth (HMA 13)": modeled_smoothed,
    })

    ax = mg.line_plot(
        plot_data,
        width=[2, 1, 2],
        color=["darkgreen", "purple", "darkorange"],
        style=["--", "-", "-"],
        annotate=True,
    )

    mg.finalise_plot(
        ax,
        title="Potential Growth: Input vs Modeled Output",
        ylabel="Per cent per annum",
        legend={"loc": "upper right", "fontsize": "small"},
        lfooter="Australia. Cobb-Douglas potential growth (input), posterior median potential growth (output), and HMA(13) smoothed.",
        rfooter=rfooter,
        y0=True,
        show=show,
    )
