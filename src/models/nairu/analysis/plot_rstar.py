"""Real neutral interest rate (r*) and its anchors.

Unlike plot_equilibrium_rates (which derives a nominal neutral from modeled
potential growth), this plots the actual deterministic r* the IS curve uses:
the fixed growth/yield blend stored in obs["det_r_star"], alongside its two
anchors (pure Cobb-Douglas growth and the real bond yield) and the real cash
rate, so both the blend and the rate gap are visible. r* here is an input, not
a latent, so there is no posterior band.
"""

import mgplot as mg
import pandas as pd

from src.models.nairu.config import DEFAULT_RSTAR_ALPHA
from src.models.nairu.results import NAIRUResults


def plot_rstar(
    results: NAIRUResults,
    *,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Plot the real r* blend, its growth and yield anchors, and the real cash rate."""
    obs, idx = results.obs, results.obs_index
    a = DEFAULT_RSTAR_ALPHA

    real_cash_rate = pd.Series(obs["cash_rate"] - obs["π_exp"], index=idx)
    plot_data = pd.DataFrame({
        f"r* ({a:.0%} growth / {1 - a:.0%} yield blend)": pd.Series(obs["det_r_star"], index=idx),
        "Growth anchor (Cobb-Douglas)": pd.Series(obs["rstar_growth"], index=idx),
        "Yield anchor (real 10y)": pd.Series(obs["yield_anchor"], index=idx),
        "Real cash rate": real_cash_rate,
    })

    mg.line_plot_finalise(
        plot_data,
        width=[2.6, 1.3, 1.3, 1.3],
        color=["black", "tab:green", "tab:red", "tab:blue"],
        style=["-", "--", "--", "-."],
        annotate=True,
        rounding=1,
        title="Real Neutral Rate (r*) and Anchors",
        ylabel="Per cent per annum (real)",
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter=(
            f"Australia. r* = {a:.0%} Cobb-Douglas growth + {1 - a:.0%} real 10y yield "
            "(deterministic IS-curve input)."
        ),
        rfooter=rfooter,
        y0=True,
        show=show,
    )
