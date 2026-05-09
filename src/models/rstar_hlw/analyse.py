"""Analysis and charts for the HLW r-star model."""

from pathlib import Path

import mgplot as mg
import pandas as pd

from src.data.world_rstar import get_world_rstar
from src.models.rstar_hlw.results import DEFAULT_CHART_BASE, RStarResults, load_results

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


def plot_r_star_bimodal_decomposition(
    results: RStarResults,
    show: bool = False,
    n_draws: int = 500,
    seed: int = 42,
    low_thresh: float = 0.05,
    high_thresh: float = 0.95,
) -> None:
    """Decomposition chart for Resolution G (bimodal α posterior).

    The standard ``plot_r_star_decomposition`` plots the posterior median r*,
    g, and bond anchor — which is misleading under G because the median r*
    sits in the middle of two modes that contain almost no posterior mass.

    This chart replaces it with a direct visualisation of the bimodality:

    - Thin translucent lines: ``n_draws`` randomly-sampled r* paths from the
      joint posterior. These cluster around the two anchors (g at the upper
      mode, indexed_10y − k at the lower mode), with very few in between.
    - Two thick lines: median r* conditional on α near 0 (``< low_thresh``;
      pure bond-anchor mode) and α near 1 (``> high_thresh``; pure
      trend-growth mode).

    Together they show what G's posterior actually says: r* is *either* g
    *or* indexed_10y − k, not a 50/50 blend.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    posterior = results.trace.posterior
    r_star_stacked = posterior["r_star"].stack(sample=("chain", "draw"))
    # PyMC time dim is auto-named (e.g. 'r_star_dim_0'); pick whichever is not 'sample'.
    time_dim = next(d for d in r_star_stacked.dims if d != "sample")
    r_star = r_star_stacked.transpose(time_dim, "sample")
    alpha = posterior["alpha_rstar"].stack(sample=("chain", "draw"))

    r_star_arr = r_star.values  # shape (T, n_samples)
    alpha_arr = alpha.values    # shape (n_samples,)
    n_samples = r_star_arr.shape[1]
    dates = results.obs_index.to_timestamp()

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n_samples, size=min(n_draws, n_samples), replace=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for idx in sample_idx:
        ax.plot(dates, r_star_arr[:, idx], color="navy", alpha=0.07, linewidth=0.4)

    low_mask = alpha_arr < low_thresh
    high_mask = alpha_arr > high_thresh
    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())

    if n_low > 0:
        low_med = np.median(r_star_arr[:, low_mask], axis=1)
        ax.plot(
            dates, low_med, color="darkorange", linewidth=2.5,
            label=f"α near 0 mode (n={n_low}/{n_samples} draws)  →  r* tracks indexed_10y − k",
        )
    if n_high > 0:
        high_med = np.median(r_star_arr[:, high_mask], axis=1)
        ax.plot(
            dates, high_med, color="steelblue", linewidth=2.5,
            label=f"α near 1 mode (n={n_high}/{n_samples} draws)  →  r* tracks trend growth g",
        )

    ax.set_ylabel("Per cent per annum")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)

    mg.finalise_plot(
        ax,
        title="r* posterior under bimodal α (Resolution G)",
        lheader=(
            f"{len(sample_idx)} thin lines = posterior draws of r*. "
            f"Bimodal α posterior → draws cluster near g (upper) or "
            f"indexed_10y − k (lower); few in between."
        ),
        lfooter=LFOOTER,
        rfooter=RFOOTER + "Mode lines: median r* | α < 0.05 (orange), α > 0.95 (blue).",
        show=show,
    )


def plot_g_vs_anchor(results: RStarResults, show: bool = False) -> None:
    """Diagnostic: posterior g vs its linear-trend anchor and raw YoY GDP growth.

    Tests whether the IS curve / y* equation move g away from the soft
    linear-trend anchor. If posterior g overlays the linear trend, the IS
    curve is not pulling g; if it parts ways, the IS curve is doing real
    work via the y* drift channel.
    """
    g = results.trend_growth_median()
    anchor = pd.Series(results.obs["trend_growth_obs"], index=results.obs_index)
    log_gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)
    yoy = log_gdp.diff(4)  # annual log-growth in log x 100 units = YoY % growth

    df = pd.DataFrame({
        "g (posterior median)":              g,
        "Linear trend of YoY growth (anchor)": anchor,
        "YoY GDP growth (raw)":              yoy,
    })

    diff = g - anchor
    rmse = float((diff ** 2).mean() ** 0.5)
    max_abs = float(diff.abs().max())

    mg.line_plot_finalise(
        df,
        title="Trend growth: posterior g vs linear-trend anchor",
        ylabel="Per cent per annum",
        color=["navy", "darkorange", "lightsteelblue"],
        width=[2.5, 1.5, 0.9],
        style=["-", "--", "-"],
        annotate=True,
        rounding=1,
        y0=True,
        lheader=(
            f"If g and the linear-trend anchor overlap, the IS curve is not "
            f"moving g. RMSE = {rmse:.2f} pp, max |gap| = {max_abs:.2f} pp."
        ),
        lfooter=LFOOTER,
        rfooter=RFOOTER + "Diagnostic: is the IS curve identifying g away from its anchor?",
        legend=True,
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
        lheader="r* is robustly estimated - the g / bond-anchor split is interpretive scaffolding — these components are not independently estimated.",
        lfooter=LFOOTER,
        rfooter=RFOOTER,
        legend=True,
        show=show,
    )


def plot_world_rstar_overlay(results: RStarResults, show: bool = False) -> None:
    """Median r*_AU vs the NY Fed HLW r* estimates for US, Euro Area, Canada.

    Always pulls fresh data from the NY Fed (force_download=True) so the
    comparison reflects the latest published HLW estimates rather than a
    stale cached file. The chart starts at the AU sample start.
    """
    au = results.r_star_median()
    components = get_world_rstar(force_download=True)

    df = pd.DataFrame({
        "r* (Australia)": au,
        "US":             components["US"],
        "Euro Area":      components["Euro Area"],
        "Canada":         components["Canada"],
    })
    df = df.loc[df.index >= au.index[0]]

    mg.line_plot_finalise(
        df,
        title="r* Comparison: Australia, US, Canada and Euro Area",
        ylabel="Per cent per annum",
        color=["navy", "steelblue", "seagreen", "firebrick"],
        width=[2.5, 1.4, 1.4, 1.4],
        style=["-", "--", "--", "--"],
        annotate=True,
        rounding=1,
        y0=True,
        lfooter=LFOOTER,
        rfooter=RFOOTER + "Foreign r* = NY Fed HLW (US, Euro Area, Canada).",
        legend=True,
        show=show,
    )


def plot_alpha_path(results: RStarResults, show: bool = False) -> None:
    """For Resolution H: fan chart of time-varying alpha_t over the sample.

    Replaces the scalar-alpha posterior histogram (which doesn't apply when
    alpha varies over time). Shows the median alpha_t with 50% and 90%
    credible bands. The shape over time tells us whether the data wants a
    drift in the structural-vs-market anchor weight (e.g. toward 0 in
    recent years if Bullock's "shifts in r*" framing is data-supported).
    """
    posterior = results.trace.posterior["alpha_rstar"].stack(sample=("chain", "draw"))
    time_dim = next(d for d in posterior.dims if d != "sample")
    alpha = posterior.transpose(time_dim, "sample").values  # (T, n_samples)
    dates = results.obs_index

    df = pd.DataFrame(alpha, index=dates)
    q05 = df.quantile(0.05, axis=1)
    q25 = df.quantile(0.25, axis=1)
    q75 = df.quantile(0.75, axis=1)
    q95 = df.quantile(0.95, axis=1)
    median = df.median(axis=1)

    ax = mg.fill_between_plot(
        pd.DataFrame({"lower": q05, "upper": q95}),
        color="navy", alpha=0.12,
    )
    mg.fill_between_plot(
        pd.DataFrame({"lower": q25, "upper": q75}),
        color="navy", alpha=0.22, ax=ax,
    )
    mg.line_plot(median, ax=ax, color=["navy"], width=2, annotate=True, rounding=2)
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axhline(0.0, color="darkorange", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(1.0, color="steelblue", linewidth=0.5, linestyle="--", alpha=0.5)

    mg.finalise_plot(
        ax,
        title="Time-varying alpha_t (Resolution H)",
        ylabel="alpha (weight on trend growth g vs bond anchor)",
        lheader=(
            "alpha=1: r* tracks trend growth; alpha=0: r* tracks indexed_10y - k. "
            "Drift in alpha_t reveals which anchor the data thinks matters more in each era."
        ),
        lfooter=LFOOTER,
        rfooter=RFOOTER + "50% and 90% credible bands.",
        legend=False,
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
    has_blend = "alpha_rstar" in posterior_vars
    has_blended_z = "rho_z" in posterior_vars  # E and F use AR(1) z
    has_soe = "gamma_icp" in posterior_vars or "gamma_twi" in posterior_vars
    has_indexed_bond_obs = "tp" in posterior_vars
    has_alpha_hierarchical = "alpha_a_hyper" in posterior_vars  # G
    has_logit_alpha = "logit_alpha" in posterior_vars  # H — time-varying alpha

    is_resolution_h = has_blend and has_logit_alpha
    is_resolution_g = has_blend and has_alpha_hierarchical and not is_resolution_h
    is_resolution_f = has_blend and has_blended_z and has_soe and not is_resolution_g and not is_resolution_h
    is_resolution_e = has_blend and has_blended_z and not has_soe and not is_resolution_g and not is_resolution_h
    is_resolution_c = has_blend and not has_blended_z and not is_resolution_g and not is_resolution_h
    is_resolution_b = (not has_blend) and has_indexed_bond_obs
    is_resolution_d = (not has_blend) and (not is_resolution_b) and has_soe
    is_blend = is_resolution_c or is_resolution_e or is_resolution_f or is_resolution_g or is_resolution_h

    label = (
        "H (blend with time-varying alpha_t via logit-RW)" if is_resolution_h
        else "G (blend + hierarchical Beta(a,b) on alpha)" if is_resolution_g
        else "F (blend + AR(1) z + SOE IS)"               if is_resolution_f
        else "E (blend + AR(1) z)"                        if is_resolution_e
        else "C (blend)"                                  if is_resolution_c
        else "B (canonical + indexed bond)"               if is_resolution_b
        else "D (canonical r* + SOE IS curve)"            if is_resolution_d
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
        if is_blend:
            k_med = float(get_scalar_var("k", results.trace).median())
            if is_resolution_h:
                # alpha_rstar is a vector (T,) — show first/last/range
                from src.models.common.extraction import get_vector_var  # noqa: PLC0415
                alpha_path = get_vector_var("alpha_rstar", results.trace)
                alpha_path.index = results.obs_index
                alpha_med = alpha_path.median(axis=1)
                print(f"  alpha_t (time-varying): start {alpha_med.iloc[0]:.3f}, "
                      f"latest {alpha_med.iloc[-1]:.3f}, "
                      f"range [{alpha_med.min():.3f}, {alpha_med.max():.3f}]")
            else:
                alpha_med = float(get_scalar_var("alpha_rstar", results.trace).median())
                print(f"  alpha median:  {alpha_med:.3f}  (note: median is misleading for bimodal posteriors — see chart)" if is_resolution_g else f"  alpha median:  {alpha_med:.3f}")
            print(f"  k median:      {k_med:.3f}")
            if has_alpha_hierarchical:
                a_med = float(get_scalar_var("alpha_a_hyper", results.trace).median())
                b_med = float(get_scalar_var("alpha_b_hyper", results.trace).median())
                print(f"  alpha_a_hyper: {a_med:.3f}  alpha_b_hyper: {b_med:.3f}  "
                      f"({'data prefers near-Jeffreys (sub-1)' if (a_med < 1 and b_med < 1) else 'data prefers central-mass (>1)'})")
            if has_blended_z:
                rho_med = float(get_scalar_var("rho_z", results.trace).median())
                z = results.trace.posterior["z_star"].stack(s=("chain", "draw")).values
                z_median_path = pd.DataFrame(z).median(axis=1)
                print(f"  rho_z median:  {rho_med:.3f}")
                print(f"  |z| mean:      {abs(z_median_path).mean():.3f} pp"
                      f"  (range [{z_median_path.min():.2f}, {z_median_path.max():.2f}])")
        elif is_resolution_b:
            tp_med = float(get_scalar_var("tp", results.trace).median())
            print(f"  tp median:     {tp_med:.3f}")

    plot_r_star(results, show=show)
    plot_trend_growth(results, show=show)
    plot_output_gap(results, show=show)
    if is_blend:
        if is_resolution_g:
            # Median-based decomposition is misleading under G's bimodal α
            # posterior — replace with the draw-cloud + mode-conditional version.
            plot_r_star_bimodal_decomposition(results, show=show)
        elif is_resolution_h:
            # Time-varying alpha — alpha_t per period; standard decomposition
            # using a scalar alpha doesn't apply.
            plot_alpha_path(results, show=show)
        else:
            plot_r_star_decomposition(results, show=show)
        plot_g_vs_anchor(results, show=show)
        if not is_resolution_h:
            # plot_alpha_posterior assumes scalar alpha — skip for H.
            plot_alpha_posterior(results, show=show)
    plot_world_rstar_overlay(results, show=show)

    print(f"Charts saved to: {chart_dir}")
