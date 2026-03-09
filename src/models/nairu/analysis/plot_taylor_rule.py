"""Taylor rule plot."""

import mgplot as mg
import numpy as np
import pandas as pd
from scipy import stats

from src.data.rba_loader import PI_TARGET
from src.models.common.extraction import get_vector_var
from src.models.common.timeseries import plot_posterior_timeseries
from src.models.nairu.results import NAIRUResults


def _quarterly_to_monthly(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Convert quarterly PeriodIndex data to monthly with interpolation."""
    monthly_idx = data.index.to_timestamp(how="end").to_period("M")
    full_monthly = pd.period_range(start=monthly_idx.min(), end=monthly_idx.max(), freq="M")
    result = data.copy()
    result.index = monthly_idx
    return result.reindex(full_monthly).interpolate(limit=2)


def plot_taylor_rule(
    results: NAIRUResults,
    *,
    cash_rate_monthly: pd.Series,
    pi_target: float = PI_TARGET,
    pi_coef_start: float = 1.6,
    pi_coef_end: float = 1.25,
    r_star_trend_weight: float = 0.75,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Plot Taylor Rule prescribed rate vs actual RBA cash rate.

    Taylor Rule: i = r* + pi_coef * pi - 0.5 * pi_target + 0.5 * y_gap
    """
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    # r* = annual potential growth (with trend smoothing)
    r_star = potential.diff(4).dropna()
    median = r_star.quantile(0.5, axis=1)
    slope, intercept, *_ = stats.linregress(np.arange(len(median)), median.to_numpy())
    trend = intercept + slope * np.arange(len(median))

    r_star_raw = median.iloc[-1]
    r_star_trend = trend[-1]
    w = r_star_trend_weight
    r_star_hybrid = (1 - w) * r_star_raw + w * r_star_trend

    if w > 0:
        r_star = r_star.multiply(1 - w).add(trend * w, axis=0)

    # Output gap
    log_gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)
    actual_gdp = log_gdp.to_numpy()
    output_gap = actual_gdp[:, np.newaxis] - potential.to_numpy()
    output_gap = pd.DataFrame(output_gap, index=results.obs_index, columns=potential.columns)
    output_gap = output_gap.reindex(r_star.index)

    # Time-varying inflation coefficient
    pi_coef = pd.Series(
        np.linspace(pi_coef_start, pi_coef_end, len(r_star)),
        index=r_star.index,
    )

    # Taylor Rule rate
    inflation_annual = pd.Series(results.obs["π4"], index=results.obs_index)
    pi = inflation_annual.reindex(r_star.index)
    taylor = (
        r_star.add(pi_coef * pi, axis=0)
        .add(-0.5 * pi_target)
        .add(output_gap.multiply(0.5))
    ).dropna()

    taylor_monthly = _quarterly_to_monthly(taylor)

    ax = plot_posterior_timeseries(
        data=taylor_monthly,
        legend_stem="Taylor Rule",
        color="darkblue",
        start=pd.Period("1993-01", freq="M"),
        finalise=False,
    )

    cash_rate_monthly.name = "RBA Cash Rate"
    cash_plot = cash_rate_monthly.loc[cash_rate_monthly.index >= pd.Period("1993-01", freq="M")]
    mg.line_plot(cash_plot, ax=ax, color="#dd0000", width=1, drawstyle="steps-post", annotate=True)

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Taylor Rule vs RBA Cash Rate",
            ylabel="Per cent per annum",
            legend={"loc": "best", "fontsize": "x-small"},
            lfooter=r"Australia. Taylor Rule: $i = r^* + \phi_\pi \pi - 0.5\bar{\pi} + 0.5\tilde{y}$; "
            f"$\\phi_\\pi$={pi_coef_start}→{pi_coef_end}; $\\bar{{\\pi}}$={pi_target}%",
            rfooter=f"Final r*={r_star_hybrid:.1f}% ({int(w*100)}% trend {r_star_trend:.1f}%, "
            f"{int((1-w)*100)}% raw {r_star_raw:.1f}%)",
            rheader=rfooter,
            axisbelow=True,
            y0=True,
            show=show,
        )
