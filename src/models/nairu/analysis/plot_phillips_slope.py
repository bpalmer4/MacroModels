"""Phillips curve slope over time with regime boundaries."""

import mgplot as mg
import numpy as np
import pandas as pd

from src.models.common.extraction import get_scalar_var, get_vector_var
from src.models.nairu.config import REGIME_COVID_START, REGIME_GFC_START

_CONFIG = {
    "price": ("gamma_pi", "steelblue", "Price Phillips Curve Slope Over Time", ""),
    "wage": ("gamma_wg", "darkorange", "Wage-ULC Phillips Curve Slope Over Time", "ULC = Unit Labour Costs."),
    "hcoe": ("gamma_hcoe", "green", "Wage-HCOE Phillips Curve Slope Over Time",
             "HCOE = Hourly Compensation of Employees."),
}


def plot_phillips_slope(
    results,
    *,
    curve_type: str = "price",
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Plot how Phillips curve steepness has evolved over time."""
    prefix, color, title, lheader = _CONFIG[curve_type]
    trace = results.trace
    obs_index = results.obs_index

    gamma_pre = float(np.median(get_scalar_var(f"{prefix}_pre_gfc", trace).to_numpy()))
    gamma_gfc = float(np.median(get_scalar_var(f"{prefix}_gfc", trace).to_numpy()))
    gamma_covid = float(np.median(get_scalar_var(f"{prefix}_covid", trace).to_numpy()))

    nairu_median = np.median(get_vector_var("nairu", trace).to_numpy(), axis=1)

    gamma_by_period = np.where(
        obs_index < REGIME_GFC_START, gamma_pre,
        np.where(obs_index < REGIME_COVID_START, gamma_gfc, gamma_covid),
    )

    slope = pd.Series(gamma_by_period / nairu_median, index=obs_index, name="Phillips Curve Slope")

    ax = mg.line_plot(slope, color=color, width=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(REGIME_GFC_START.ordinal, color="darkred", linestyle="--",
               linewidth=1, alpha=0.7, label=f"GFC ({REGIME_GFC_START})")
    ax.axvline(REGIME_COVID_START.ordinal, color="darkgreen", linestyle="-.",
               linewidth=1, alpha=0.7, label=f"COVID ({REGIME_COVID_START})")

    mg.finalise_plot(
        ax,
        title=title,
        ylabel="Slope at NAIRU (gamma/u*)",
        legend={"loc": "best", "fontsize": "x-small"},
        lheader=lheader,
        lfooter=f"γ: pre-GFC={gamma_pre:.2f}, post-GFC={gamma_gfc:.2f}, post-COVID={gamma_covid:.2f}",
        rfooter=rfooter,
        show=show,
    )
