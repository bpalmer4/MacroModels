"""Analysis and charts for the HLW r-star model."""

from pathlib import Path

import mgplot as mg
import pandas as pd

from src.models.rstar.results import DEFAULT_CHART_BASE, RStarResults, load_results

LFOOTER = "Australia. "
RFOOTER = "HLW Bayesian r-star model. "


def _chart_dir_for(resolution: str) -> Path:
    return DEFAULT_CHART_BASE / f"rstar-hlw-{resolution}"


def _fan_chart(
    posterior: pd.DataFrame,
    title: str,
    ylabel: str,
    rfooter: str = "",
    y0: bool = True,
    show: bool = False,
) -> None:
    """Plot a fan chart from posterior samples (50% and 90% credible bands)."""
    q05 = posterior.quantile(0.05, axis=1)
    q25 = posterior.quantile(0.25, axis=1)
    q75 = posterior.quantile(0.75, axis=1)
    q95 = posterior.quantile(0.95, axis=1)
    median = posterior.median(axis=1)

    ax = mg.fill_between_plot(
        pd.DataFrame({"lower": q05, "upper": q95}),
        color="navy",
        alpha=0.12,
    )
    mg.fill_between_plot(
        pd.DataFrame({"lower": q25, "upper": q75}),
        color="navy",
        alpha=0.22,
        ax=ax,
    )
    mg.line_plot(median, ax=ax, color=["navy"], width=2, annotate=True, rounding=1)
    mg.finalise_plot(
        ax,
        title=title,
        ylabel=ylabel,
        lfooter=LFOOTER,
        rfooter=RFOOTER + rfooter,
        y0=y0,
        legend=False,
        show=show,
    )


def plot_r_star(results: RStarResults, show: bool = False) -> None:
    """r* fan chart with the real cash rate overlaid for monetary policy context."""
    posterior = results.r_star_posterior()
    q05 = posterior.quantile(0.05, axis=1)
    q25 = posterior.quantile(0.25, axis=1)
    q75 = posterior.quantile(0.75, axis=1)
    q95 = posterior.quantile(0.95, axis=1)
    median = posterior.median(axis=1)

    # Real cash rate (apples-to-apples with r*): nominal cash - inflation expectations
    cash = pd.Series(results.obs["cash_rate"], index=results.obs_index)
    pi_exp = pd.Series(results.obs["pi_exp"], index=results.obs_index)
    real_cash = cash - pi_exp

    ax = mg.fill_between_plot(
        pd.DataFrame({"lower": q05, "upper": q95}),
        color="navy",
        alpha=0.12,
    )
    mg.fill_between_plot(
        pd.DataFrame({"lower": q25, "upper": q75}),
        color="navy",
        alpha=0.22,
        ax=ax,
    )
    mg.line_plot(median, ax=ax, color=["navy"], width=2, annotate=True, rounding=1)
    mg.line_plot(
        real_cash, ax=ax, color=["darkred"], width=1.2, style="--",
        annotate=True, rounding=1,
    )
    mg.finalise_plot(
        ax,
        title="Natural Rate of Interest (r*) and Real Cash Rate",
        ylabel="Per cent per annum",
        lfooter=LFOOTER,
        rfooter=RFOOTER + "Real cash rate = cash rate − π_exp. 50% and 90% credible bands on r*.",
        y0=True,
        legend=False,
        show=show,
    )


def plot_trend_growth(results: RStarResults, show: bool = False) -> None:
    _fan_chart(
        results.trend_growth_posterior(),
        title="Trend Output Growth (g)",
        ylabel="Per cent per annum",
        rfooter="Annualised. 50% and 90% credible bands.",
        show=show,
    )


def plot_output_gap(results: RStarResults, show: bool = False) -> None:
    _fan_chart(
        results.output_gap_posterior(),
        title="Output Gap (HLW)",
        ylabel="Per cent of potential GDP",
        rfooter="log-difference x 100. 50% and 90% credible bands.",
        show=show,
    )


def plot_r_star_decomposition(results: RStarResults, show: bool = False) -> None:
    """Median r* against the structural and market anchors that define it.

    For the blend specification: r* = alpha*g + (1-alpha)*(indexed_10y - k) + eps.
    Plot r*, the trend-growth anchor (g), and the bond-implied anchor
    (indexed_10y - k_median) so the user can see where r* sits between them.
    """
    from src.models.common.extraction import get_scalar_var  # noqa: PLC0415

    indexed = pd.Series(results.obs["indexed_10y"], index=results.obs_index)
    k_median = float(get_scalar_var("k", results.trace).median())
    bond_anchor = indexed - k_median

    df = pd.DataFrame({
        "r*": results.r_star_median(),
        "Trend growth (g)": results.trend_growth_median(),
        f"Bond anchor (indexed_10y − {k_median:.2f})": bond_anchor,
    })

    mg.line_plot_finalise(
        df,
        title="r* Decomposition: blend of structural and market anchors",
        ylabel="Per cent per annum",
        color=["navy", "steelblue", "darkorange"],
        width=[2.5, 1.5, 1.5],
        style=["-", "--", "--"],
        annotate=True,
        rounding=1,
        y0=True,
        lfooter=LFOOTER + "r* is robust, but component levels are estimated to satisfy the IS curve.",
        rfooter=RFOOTER,
        legend=True,
        show=show,
    )


def plot_alpha_posterior(results: RStarResults, show: bool = False) -> None:
    """Histogram of alpha_rstar posterior — the weight on the structural anchor.

    alpha=1 means r* tracks trend growth (Resolution A);
    alpha=0 means r* tracks the bond-implied real rate (Resolution B).
    """
    from src.models.common.extraction import get_scalar_var  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    alpha = get_scalar_var("alpha_rstar", results.trace)
    median = float(alpha.median())
    hdi_lo, hdi_hi = float(alpha.quantile(0.05)), float(alpha.quantile(0.95))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(alpha, bins=40, color="navy", alpha=0.7, density=True)
    ax.axvline(median, color="red", linestyle="--", linewidth=2,
               label=f"Median = {median:.2f}")
    ax.axvspan(hdi_lo, hdi_hi, color="red", alpha=0.1,
               label=f"90% CI: [{hdi_lo:.2f}, {hdi_hi:.2f}]")
    ax.set_xlim(0, 1)
    ax.set_xlabel("alpha (weight on trend growth g vs bond anchor)")
    ax.set_ylabel("Posterior density")
    ax.legend(loc="upper left", fontsize="small")

    mg.finalise_plot(
        ax,
        title="Posterior of alpha (structural vs market r* anchor weight)",
        lfooter=LFOOTER,
        rfooter=RFOOTER + "alpha=1: r*=g; alpha=0: r*=indexed_10y-k.",
        show=show,
    )


def run_analyse(
    prefix: str = "rstar_hlw_C",
    chart_dir: Path | str | None = None,
    resolution: str = "C",
    verbose: bool = False,
    show: bool = False,
) -> None:
    """Load saved results and produce all standard charts."""
    if chart_dir is None:
        chart_dir = _chart_dir_for(resolution)

    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    print(f"Loading results: {prefix}")
    results = load_results(prefix=prefix)

    posterior_vars = results.trace.posterior.data_vars
    is_resolution_c = "alpha_rstar" in posterior_vars
    is_resolution_b = (not is_resolution_c) and ("tp" in posterior_vars)
    label = (
        "C (blend)" if is_resolution_c
        else "B (canonical + indexed bond)" if is_resolution_b
        else "A (canonical, r* = g + z)"
    )

    if verbose:
        from src.models.common.extraction import get_scalar_var  # noqa: PLC0415

        r_star = results.r_star_median()
        g = results.trend_growth_median()
        print(f"  Sample:        {results.obs_index[0]} to {results.obs_index[-1]}")
        print(f"  Resolution:    {label}")
        print(f"  r* range:      [{r_star.min():.2f}, {r_star.max():.2f}]%")
        print(f"  r* latest:     {r_star.iloc[-1]:.2f}%")
        print(f"  g  latest:     {g.iloc[-1]:.2f}%")
        if is_resolution_c:
            alpha_med = float(get_scalar_var("alpha_rstar", results.trace).median())
            k_med = float(get_scalar_var("k", results.trace).median())
            print(f"  alpha median:  {alpha_med:.3f}")
            print(f"  k median:      {k_med:.3f}")
        elif is_resolution_b:
            tp_med = float(get_scalar_var("tp", results.trace).median())
            print(f"  tp median:     {tp_med:.3f}")

    plot_r_star(results, show=show)
    plot_trend_growth(results, show=show)
    plot_output_gap(results, show=show)
    if is_resolution_c:
        plot_r_star_decomposition(results, show=show)
        plot_alpha_posterior(results, show=show)

    print(f"Charts saved to: {chart_dir}")
