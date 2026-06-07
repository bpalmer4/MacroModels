"""Unemployment rate needed to return inflation to target now.

Only meaningful for excess_expectations variants. The structural NAIRU (U*)
is the rate consistent with target inflation once expectations have
re-anchored. While actual expectations sit off target (excess != 0), hitting
2.5% immediately requires enough labour-market slack to offset the
beta x excess pass-through. Setting the Phillips curve to target:

    gamma x (U - U*)/U = -beta x excess   =>   U_2.5 = U* / (1 + beta x excess / gamma)

The wedge between U_2.5 and U* is the temporary unemployment cost of
de-anchored expectations; it closes as the excess decays to zero.
"""

import mgplot as mg
import numpy as np
import pandas as pd

from src.models.common.extraction import get_scalar_var
from src.models.nairu.config import REGIME_COVID_START, REGIME_GFC_START
from src.models.nairu.results import NAIRUResults
from src.utilities.rate_conversion import quarterly

# Start at the fully target-anchored era (excess is zero/phasing before this)
START = pd.Period("1999Q1", freq="Q")


def _gamma_draws(results: NAIRUResults, index: pd.PeriodIndex) -> np.ndarray:
    """Price Phillips slope draws as a (periods x samples) array (regime-aware)."""
    trace = results.trace
    if "gamma_pi_pre_gfc" in trace.posterior:
        draws = {name: get_scalar_var(f"gamma_pi_{name}", trace).to_numpy()
                 for name in ["pre_gfc", "gfc", "covid"]}
        gamma = np.empty((len(index), len(draws["pre_gfc"])))
        gamma[index < REGIME_GFC_START, :] = draws["pre_gfc"]
        gamma[(index >= REGIME_GFC_START) & (index < REGIME_COVID_START), :] = draws["gfc"]
        gamma[index >= REGIME_COVID_START, :] = draws["covid"]
        return gamma
    single = get_scalar_var("gamma_pi", trace).to_numpy()
    return np.broadcast_to(single, (len(index), len(single)))


def plot_target_consistent_unemployment(
    results: NAIRUResults,
    *,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Plot U* alongside the unemployment rate that returns inflation to 2.5% now."""
    if "beta_pi" not in results.trace.posterior or "π_exp_gap" not in results.obs:
        return  # not an excess-expectations variant

    obs_index = results.obs_index
    nairu = results.nairu_posterior().to_numpy()  # (T, S)
    beta = get_scalar_var("beta_pi", results.trace).to_numpy()  # (S,)
    gamma = _gamma_draws(results, obs_index)  # (T, S)
    excess_q = quarterly(results.obs["π_exp"] + results.obs["π_exp_gap"]) - quarterly(results.obs["π_exp"])

    # U_2.5 per posterior draw
    u_tn = nairu / (1 + beta[None, :] * excess_q[:, None] / gamma)

    bands = pd.DataFrame(
        {
            "lower": np.quantile(u_tn, 0.05, axis=1),
            "median": np.median(u_tn, axis=1),
            "upper": np.quantile(u_tn, 0.95, axis=1),
            "nairu": np.median(nairu, axis=1),
        },
        index=obs_index,
    )
    bands = bands[bands.index >= START]
    U = pd.Series(results.obs["U"], index=obs_index)
    U = U[U.index >= START]
    U.name = "Unemployment rate"

    ax = mg.fill_between_plot(
        bands[["lower", "upper"]], color="purple", alpha=0.15,
        label="Re-anchoring effort: 90% credible interval",
    )
    median = bands["median"]
    median.name = "U to re-anchor expectations (2.5% inflation now, median)"
    mg.line_plot(median, ax=ax, color=["purple"], width=2, annotate=True, rounding=1)
    nairu_med = bands["nairu"]
    nairu_med.name = "NAIRU U* (once expectations re-anchor to target)"
    mg.line_plot(nairu_med, ax=ax, color=["blue"], width=1.5, style="--", annotate=True, rounding=1)
    mg.line_plot(U, ax=ax, color=["darkorange"], width=1.5, annotate=True, rounding=1)

    mg.finalise_plot(
        ax,
        title="Unemployment Rate Needed to Return Inflation to Target",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "x-small"},
        lheader=r"$U_{2.5} = U^*/(1 + \beta \cdot excess/\gamma)$: slack needed to offset de-anchored expectations.",
        lfooter="Australia. The wedge between $U_{2.5}$ and U* closes as expectations re-anchor.",
        rfooter=rfooter,
        axisbelow=True,
        show=show,
    )
