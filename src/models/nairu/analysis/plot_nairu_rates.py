"""Taylor rule and equilibrium rate plotting functions."""

from typing import TypeVar

import mgplot as mg
import numpy as np
import pandas as pd
from scipy import stats

from src.data.rba_loader import PI_TARGET
from src.models.nairu.analysis.extraction import get_vector_var
from src.models.nairu.analysis.plot_posterior_timeseries import plot_posterior_timeseries

# Plotting constants
START = pd.Period("1985Q1", freq="Q")
RFOOTER = "Joint NAIRU + Output Gap Model"

# Type variable for Series or DataFrame
PandasT = TypeVar("PandasT", pd.Series, pd.DataFrame)


def _quarterly_to_monthly(data: PandasT) -> PandasT:
    """Convert quarterly PeriodIndex data to monthly with interpolation.

    Args:
        data: Series or DataFrame with quarterly PeriodIndex

    Returns:
        Data with monthly PeriodIndex, linearly interpolated (limit=2)

    """
    # Convert Q periods to end-of-quarter months
    monthly_idx = data.index.to_timestamp(how="end").to_period("M")

    # Create full monthly range
    full_monthly = pd.period_range(start=monthly_idx.min(), end=monthly_idx.max(), freq="M")

    # Reindex and interpolate
    result = data.copy()
    result.index = monthly_idx
    result = result.reindex(full_monthly).interpolate(limit=2)

    return result


def plot_taylor_rule(
    results,  # NAIRUResults - avoid circular import
    inflation_annual: pd.Series,
    cash_rate_monthly: pd.Series,
    pi_target: float = PI_TARGET,
    pi_coef_start: float = 1.6,
    pi_coef_end: float = 1.25,
    r_star_trend_weight: float = 0.75,
    show: bool = False,
) -> None:
    """Plot Taylor Rule prescribed rate vs actual RBA cash rate.

    Taylor Rule: i = r* + pi_coef * pi - 0.5 * pi_target + 0.5 * y_gap
    where pi_target is the inflation target (2.5%)
    """
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    # r* = annual potential growth
    r_star = potential.diff(4).dropna()

    # Calculate raw median and trend for reporting
    median = r_star.quantile(0.5, axis=1)
    slope, intercept, *_ = stats.linregress(np.arange(len(median)), median.to_numpy())
    trend = intercept + slope * np.arange(len(median))

    # Current r* values for annotation
    r_star_raw = median.iloc[-1]
    r_star_trend = trend[-1]
    w = r_star_trend_weight
    r_star_hybrid = (1 - w) * r_star_raw + w * r_star_trend

    # Smooth r* toward trend
    if r_star_trend_weight > 0:
        r_star = r_star.multiply(1 - w).add(trend * w, axis=0)

    # Output gap: (Y - Y*)/Y* * 100
    log_gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)
    actual_gdp = log_gdp.reindex(results.obs_index).to_numpy()
    output_gap = (actual_gdp[:, np.newaxis] - potential.to_numpy()) / potential.to_numpy() * 100
    output_gap = pd.DataFrame(output_gap, index=results.obs_index, columns=potential.columns)
    output_gap = output_gap.reindex(r_star.index)

    # Time-varying inflation coefficient
    pi = inflation_annual.reindex(r_star.index)
    pi_coef = pd.Series(
        np.linspace(pi_coef_start, pi_coef_end, len(r_star)),
        index=r_star.index,
    )

    # Taylor Rule for each sample
    taylor = (
        r_star.add(pi_coef * pi, axis=0)
        .add(-0.5 * pi_target)
        .add(output_gap.multiply(0.5))
    ).dropna()

    # Convert to monthly for cash rate alignment
    taylor_monthly = _quarterly_to_monthly(taylor)

    # Plot
    ax = plot_posterior_timeseries(
        data=taylor_monthly,
        legend_stem="Taylor Rule",
        color="darkblue",
        start=pd.Period("1993-01", freq="M"),
        finalise=False,
    )

    cash_rate_monthly.name = "RBA Cash Rate"
    cash_plot = cash_rate_monthly.loc[cash_rate_monthly.index >= pd.Period("1993-01", freq="M")]
    mg.line_plot(
        cash_plot,
        ax=ax,
        color="#dd0000",
        width=1,
        drawstyle="steps-post",
        annotate=True,
    )

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
            rheader=RFOOTER,
            axisbelow=True,
            y0=True,
            show=show,
        )


def plot_equilibrium_rates(
    results,  # NAIRUResults - avoid circular import
    cash_rate_monthly: pd.Series,
    pi_target: float = PI_TARGET,
    show: bool = False,
) -> None:
    """Plot neutral interest rate vs actual RBA cash rate."""
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    # r* trend from potential growth
    r_star = potential.diff(4).dropna().quantile(0.5, axis=1)
    x = np.arange(len(r_star))
    slope, intercept, *_ = stats.linregress(x, r_star.to_numpy())
    trend = pd.Series(intercept + slope * x, index=r_star.index)

    # Neutral = trend r* + pi_target
    neutral = trend + pi_target
    neutral.name = "Nominal Neutral Rate"

    # Convert to monthly
    neutral = _quarterly_to_monthly(neutral)

    # Plot
    cash_rate_monthly.name = "RBA Cash Rate"
    ax = mg.line_plot(neutral, color="darkorange", width=2, annotate=True)
    ax = mg.line_plot(
        cash_rate_monthly,
        ax=ax,
        color="darkblue",
        width=1,
        drawstyle="steps-post",
        annotate=True,
    )

    mg.finalise_plot(
        ax,
        title="Neutral Interest Rate vs RBA Cash Rate",
        ylabel="Per cent per annum",
        legend={"loc": "upper right", "fontsize": "x-small"},
        lfooter=f"Australia. Neutral rate = trend r* + pi_t (where pi_t = {pi_target}%).",
        rfooter="Equilibrium rate when output gap = 0 and U = NAIRU: i = r* + pi_t",
        rheader=RFOOTER,
        axisbelow=True,
        y0=True,
        show=show,
    )
