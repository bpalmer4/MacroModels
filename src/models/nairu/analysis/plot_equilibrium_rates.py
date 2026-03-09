"""Neutral interest rate vs RBA cash rate plot."""

import mgplot as mg
import numpy as np
import pandas as pd
from scipy import stats

from src.data.rba_loader import PI_TARGET
from src.models.common.extraction import get_vector_var
from src.models.nairu.results import NAIRUResults


def _quarterly_to_monthly(data: pd.Series) -> pd.Series:
    """Convert quarterly PeriodIndex data to monthly with interpolation."""
    monthly_idx = data.index.to_timestamp(how="end").to_period("M")
    full_monthly = pd.period_range(start=monthly_idx.min(), end=monthly_idx.max(), freq="M")
    result = data.copy()
    result.index = monthly_idx
    return result.reindex(full_monthly).interpolate(limit=2)


def plot_equilibrium_rates(
    results: NAIRUResults,
    *,
    cash_rate_monthly: pd.Series,
    pi_target: float = PI_TARGET,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Plot neutral interest rate vs actual RBA cash rate."""
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    r_star = potential.diff(4).dropna().quantile(0.5, axis=1)
    x = np.arange(len(r_star))
    slope, intercept, *_ = stats.linregress(x, r_star.to_numpy())
    trend = pd.Series(intercept + slope * x, index=r_star.index)

    neutral = trend + pi_target
    neutral.name = "Nominal Neutral Rate"
    neutral = _quarterly_to_monthly(neutral)

    cash_rate_monthly.name = "RBA Cash Rate"
    ax = mg.line_plot(neutral, color="darkorange", width=2, annotate=True)
    mg.line_plot(cash_rate_monthly, ax=ax, color="darkblue", width=1,
                 drawstyle="steps-post", annotate=True)

    mg.finalise_plot(
        ax,
        title="Neutral Interest Rate vs RBA Cash Rate",
        ylabel="Per cent per annum",
        legend={"loc": "upper right", "fontsize": "x-small"},
        lfooter=f"Australia. Neutral rate = trend potential GDP growth + inflation target ({pi_target}%).",
        rfooter="Equilibrium rate when output gap = 0 and U = NAIRU.",
        rheader=rfooter,
        axisbelow=True,
        y0=True,
        show=show,
    )
