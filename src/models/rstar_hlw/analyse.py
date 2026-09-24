"""Analysis and charts for the HLW r-star model."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd

from src.data.world_rstar import get_world_rstar
from src.models.common import prior_posterior
from src.models.common.diagnostics import save_diagnostics
from src.models.common.extraction import get_scalar_var, get_vector_var
from src.models.rstar_hlw.observations import DEFAULT_G_ANCHOR, G_ANCHOR_LABELS
from src.models.rstar_hlw.results import DEFAULT_CHART_BASE, RStarResults, load_results

if TYPE_CHECKING:
    from collections.abc import Container, Hashable
    from pathlib import Path

LFOOTER = "Australia. "
RFOOTER = "HLW Bayesian r-star model. "


def _chart_dir_for(resolution: str) -> Path:
    return DEFAULT_CHART_BASE / f"rstar-hlw-{resolution}"


def _rstar_caveat(flags: _ResolutionFlags) -> str:
    """Build the health warning that goes on every chart plotting r*.

    These charts are the one place the package still asserts an r* path, and a
    chart asserts more confidently than a caveat three sections into
    MODEL_NOTES. The wording differs by resolution because the failure does:
    canonical HLW returns trend growth, B returns the bond yield, and the
    blends return whatever alpha was assumed.
    """
    if flags.is_b:
        return ("NOT AN ESTIMATE: r* here is the indexed bond yield less a constant premium. "
                "See MODEL_NOTES.md.")
    if flags.is_blend:
        return ("NOT AN ESTIMATE: r* here is a blend whose weight the data does not identify, "
                "so its level is the analyst's prior. See MODEL_NOTES.md.")
    return ("NOT AN ESTIMATE: z has no observation equation, so r* here is trend growth plus a "
            "level (corr 0.998). See MODEL_NOTES.md.")


def _excluded_span(results: RStarResults) -> dict[str, Any] | None:
    """Shading for the window dropped from the likelihood, if there was one.

    The states still run through the window, so the fan is drawn there like
    anywhere else. The shading is the warning that nothing in it was fitted.
    """
    window = results.constants.get("exclude_window")
    if window is None:
        return None
    lo, hi = window
    return {
        "xmin": pd.Period(lo, freq="Q"),
        "xmax": pd.Period(hi, freq="Q"),
        "color": "grey",
        "alpha": 0.15,
        "zorder": -1,
    }


def _fan_chart(
    posterior: pd.DataFrame,
    title: str,
    ylabel: str,
    *,
    rfooter: str = "",
    y0: bool = True,
    show: bool = False,
    axvspan: dict[str, Any] | None = None,
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
        axvspan=axvspan,
    )


def plot_r_star(results: RStarResults, show: bool = False, caveat: str = "") -> None:
    """r* fan chart with the real cash rate overlaid for monetary policy context.

    `caveat` is the health warning from `_rstar_caveat`, and it is not optional
    in practice: this chart puts an r* path next to the policy rate, which is
    exactly the reading the model cannot support.
    """
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

    median.name = "r* (median)"
    real_cash.name = "Real cash rate"
    ax = mg.fill_between_plot(
        pd.DataFrame({"lower": q05, "upper": q95}),
        color="navy",
        alpha=0.12,
        label="r*: 90% credible band",
    )
    mg.fill_between_plot(
        pd.DataFrame({"lower": q25, "upper": q75}),
        color="navy",
        alpha=0.22,
        ax=ax,
        label="r*: 50% credible band",
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
        lheader=caveat,
        lfooter=LFOOTER,
        rfooter=RFOOTER + "Real cash rate = cash rate − π_exp.",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        show=show,
    )


def plot_trend_growth(results: RStarResults, show: bool = False) -> None:
    """Fan chart of the latent trend growth rate g."""
    _fan_chart(
        results.trend_growth_posterior(),
        title="Trend Output Growth (g)",
        ylabel="Per cent per annum",
        rfooter="Annualised. 50% and 90% credible bands.",
        show=show,
    )


def plot_output_gap(results: RStarResults, show: bool = False) -> None:
    """Fan chart of the output gap, with any excluded window shaded."""
    span = _excluded_span(results)
    note = " Shaded: dropped from the likelihood." if span else ""
    _fan_chart(
        results.output_gap_posterior(),
        title="Output Gap (HLW)",
        ylabel="Per cent of potential GDP",
        rfooter=f"log-difference x 100. 50% and 90% credible bands.{note}",
        show=show,
        axvspan=span,
    )


def _mode_conditional_r_star(
    results: RStarResults,
    low_thresh: float = 0.05,
    high_thresh: float = 0.95,
) -> tuple[pd.Series, pd.Series, int, int, int]:
    """Median r* paths conditional on the two α modes (Resolution G).

    Returns ``(low_med, high_med, n_low, n_high, n_samples)``: the median r*
    among draws with α < ``low_thresh`` (bond-market mode: r* tracks
    indexed_10y − k) and α > ``high_thresh`` (trend-growth mode: r* tracks g),
    plus the draw counts. Empty Series where a mode has no draws.
    """
    posterior = results.trace["posterior"]
    # xarray's stack/.values throughout this function, not pandas': the PD
    # rules cannot tell the two apart, and `.melt` / `.to_numpy` are not
    # xarray methods.
    r_star_stacked = posterior["r_star"].stack(sample=("chain", "draw"))
    # PyMC time dim is auto-named (e.g. 'r_star_dim_0'); pick whichever is not 'sample'.
    time_dim = next(d for d in r_star_stacked.dims if d != "sample")
    r_star_arr = r_star_stacked.transpose(time_dim, "sample").to_numpy()
    alpha_arr = posterior["alpha_rstar"].stack(sample=("chain", "draw")).to_numpy()

    low_mask = alpha_arr < low_thresh
    high_mask = alpha_arr > high_thresh
    low_med = (
        pd.Series(np.median(r_star_arr[:, low_mask], axis=1), index=results.obs_index)
        if low_mask.any() else pd.Series(dtype=float)
    )
    high_med = (
        pd.Series(np.median(r_star_arr[:, high_mask], axis=1), index=results.obs_index)
        if high_mask.any() else pd.Series(dtype=float)
    )
    return low_med, high_med, int(low_mask.sum()), int(high_mask.sum()), r_star_arr.shape[1]


def plot_r_star_bimodal_decomposition(
    results: RStarResults,
    *,
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
    posterior = results.trace["posterior"]
    # xarray's stack/.values throughout this function, not pandas': the PD
    # rules cannot tell the two apart, and `.melt` / `.to_numpy` are not
    # xarray methods.
    r_star_stacked = posterior["r_star"].stack(sample=("chain", "draw"))
    # PyMC time dim is auto-named (e.g. 'r_star_dim_0'); pick whichever is not 'sample'.
    time_dim = next(d for d in r_star_stacked.dims if d != "sample")
    r_star_arr = r_star_stacked.transpose(time_dim, "sample").to_numpy()  # (T, n_samples)
    n_samples = r_star_arr.shape[1]

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n_samples, size=min(n_draws, n_samples), replace=False)

    # Thin translucent draw paths (legend-suppressed)
    draws = pd.DataFrame(r_star_arr[:, sample_idx], index=results.obs_index)
    ax = mg.line_plot(
        draws, color="navy", alpha=0.07, width=0.4,
        annotate=False, label_series=False, dropna=False,
    )

    low_med, high_med, n_low, n_high, _ = _mode_conditional_r_star(results, low_thresh, high_thresh)

    if n_low > 0:
        low_med.name = f"α near 0 mode (n={n_low}/{n_samples} draws)  →  r* tracks indexed_10y − k"
        mg.line_plot(low_med, ax=ax, color=["darkorange"], width=2.5, annotate=True, rounding=1)
    if n_high > 0:
        high_med.name = f"α near 1 mode (n={n_high}/{n_samples} draws)  →  r* tracks trend growth g"
        mg.line_plot(high_med, ax=ax, color=["steelblue"], width=2.5, annotate=True, rounding=1)

    mg.finalise_plot(
        ax,
        title="r* posterior under bimodal α (Resolution G)",
        ylabel="Per cent per annum",
        lheader=(
            f"{len(sample_idx)} thin lines = posterior draws of r*. "
            f"Bimodal α posterior → draws cluster near g (upper) or "
            f"indexed_10y − k (lower); few in between."
        ),
        lfooter=LFOOTER,
        rfooter=RFOOTER + "Mode lines: median r* | α < 0.05 (orange), α > 0.95 (blue).",
        y0=True,
        legend={"loc": "best", "fontsize": 9, "framealpha": 0.9},
        figsize=(10, 5.5),
        show=show,
    )


def plot_g_vs_anchor(results: RStarResults, show: bool = False) -> None:
    """Diagnostic: posterior g vs its soft anchor and raw YoY GDP growth.

    Tests whether the IS curve / y* equation move g away from the anchor. If
    posterior g overlays it, the IS curve is not pulling g; if it parts ways,
    the IS curve is doing real work via the y* drift channel.

    Which anchor the run used is read from the saved constants, so the labels
    cannot go on describing a series the run did not use.
    """
    g = results.trend_growth_median()
    anchor = pd.Series(results.obs["trend_growth_obs"], index=results.obs_index)
    log_gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)
    yoy = log_gdp.diff(4)  # annual log-growth in log x 100 units = YoY % growth

    kind = str(results.constants.get("g_anchor", DEFAULT_G_ANCHOR))
    anchor_label = G_ANCHOR_LABELS.get(kind, kind)

    df = pd.DataFrame({
        "g (posterior median)":                 g,
        f"{anchor_label.capitalize()} (anchor)": anchor,
        "YoY GDP growth (raw)":                 yoy,
    })

    diff = g - anchor
    rmse = float((diff ** 2).mean() ** 0.5)
    max_abs = float(diff.abs().max())

    mg.line_plot_finalise(
        df,
        title=f"Trend growth: posterior g vs {anchor_label} anchor",
        ylabel="Per cent per annum",
        color=["navy", "darkorange", "lightsteelblue"],
        width=[2.5, 1.5, 0.9],
        style=["-", "--", "-"],
        annotate=True,
        rounding=1,
        y0=True,
        lheader=(
            f"If g and the anchor overlap, the IS curve is not moving g. "
            f"RMSE = {rmse:.2f} pp, max |gap| = {max_abs:.2f} pp."
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
        lheader=(
            "The g / bond-anchor split is interpretive scaffolding: "
            "these components are not independently estimated."
        ),
        lfooter=LFOOTER,
        rfooter=RFOOTER,
        legend=True,
        show=show,
    )


def plot_world_rstar_overlay(
    results: RStarResults, show: bool = False, bond_mode: bool = False, caveat: str = "",
) -> None:
    """r*_AU vs the NY Fed HLW r* estimates for US, Euro Area, Canada.

    The HLW loader re-checks the NY Fed file on every call, so the comparison
    reflects the latest published estimates rather than a stale cached file.
    The chart starts at the AU sample start.

    With ``bond_mode=True`` (Resolution G), the Australian line is the
    bond-market mode median (α near 0: r* tracks indexed_10y − k) rather than
    the overall posterior median, which is misleading under G's bimodal α.
    """
    if bond_mode:
        au, _, _, _, _ = _mode_conditional_r_star(results)
        au_label = "r* (Australia, bond-market mode)"
        au_note = " AU r* = bond-market mode median (α < 0.05)."
    else:
        au = results.r_star_median()
        au_label = "r* (Australia)"
        au_note = ""
    components = get_world_rstar()

    df = pd.DataFrame({
        au_label:    au,
        "US":        components["US"],
        "Euro Area": components["Euro Area"],
        "Canada":    components["Canada"],
    })
    df = df.loc[df.index >= au.index[0]]

    mg.line_plot_finalise(
        df,
        title="r* Comparison: Australia, US, Canada and Euro Area",
        ylabel="Per cent per annum",
        lheader=caveat,
        color=["navy", "steelblue", "seagreen", "firebrick"],
        width=[2.5, 1.4, 1.4, 1.4],
        style=["-", "--", "--", "--"],
        annotate=True,
        rounding=1,
        y0=True,
        lfooter=LFOOTER,
        rfooter=RFOOTER + "Foreign r* = NY Fed HLW (US, Euro Area, Canada)." + au_note,
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
    # xarray stack/.values again, not pandas: see the note in
    # `_mode_conditional_r_star`.
    posterior = results.trace["posterior"]["alpha_rstar"].stack(sample=("chain", "draw"))
    time_dim = next(d for d in posterior.dims if d != "sample")
    alpha = posterior.transpose(time_dim, "sample").to_numpy()  # (T, n_samples)
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
    alpha = get_scalar_var("alpha_rstar", results.trace)
    median = float(alpha.median())
    hdi_lo, hdi_hi = float(alpha.quantile(0.05)), float(alpha.quantile(0.95))

    _fig, ax = plt.subplots(figsize=(8, 5))
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


# What each free scalar is, for the x-axis label. A name missing from here is
# charted under its own name rather than skipped, so a new parameter appears
# the moment it is added to an equation.
_PARAM_LABEL = {
    "sigma_g": "sigma_g: innovation sd of trend growth (pp p.a.)",
    "sigma_ystar": "sigma_ystar: innovation sd of potential (log points x 100)",
    "initial_potential": "initial_potential: y* at the first quarter",
    "sigma_z": "sigma_z: innovation sd of z (pp)",
    "a_y1": "a_y1: output gap at t-1",
    "a_y2": "a_y2: output gap at t-2",
    "a_r": "a_r: real rate gap (the IS slope)",
    "sigma_IS": "sigma_IS: residual sd of the IS curve",
    "b_y": "b_y: output gap in the Phillips curve (the slope)",
    "sigma_pi": "sigma_pi: residual sd of the Phillips curve",
    "gamma_fi": "gamma_fi: fiscal impulse at t-1",
    "gamma_tot": "gamma_tot: terms of trade growth at t-1",
    "gamma_twi": "gamma_twi: TWI change at t-1",
    "gamma_icp": "gamma_icp: ICP growth at t-1",
    "alpha_rstar": "alpha: weight on trend growth g vs the bond anchor",
    "alpha_a_hyper": "a: first shape of the hierarchical Beta on alpha",
    "alpha_b_hyper": "b: second shape of the hierarchical Beta on alpha",
    "logit_alpha_0": "logit_alpha_0: starting level of time-varying alpha",
    "k": "k: term-premium offset on the indexed bond (pp)",
    "sigma_r": "sigma_r: i.i.d. noise on r* (pp)",
    "rho_z": "rho_z: AR(1) persistence of z",
    "tp": "tp: term premium in the indexed-bond equation (pp)",
    "sigma_tp": "sigma_tp: residual sd of the indexed-bond equation",
}


def _density(draws: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Density of `draws` on `edges`, normalised over ALL draws, not the range.

    numpy's own density=True renormalises to the bins it was given, which would
    redraw a prior that is mostly off the chart as though it were concentrated
    on the part that is visible. Dividing by the full draw count instead keeps
    the two curves on one scale: a prior far wider than the posterior then
    reads as the near-flat line it is.
    """
    counts, _ = np.histogram(draws, bins=edges)
    return counts / (len(draws) * np.diff(edges))


def plot_prior_posterior(results: RStarResults, show: bool = False) -> int:
    """One chart per free scalar parameter: its posterior against its own prior.

    The question these answer is how much of each number is data. A posterior
    sitting on top of its prior means the sample said nothing and the figure is
    the prior read back; a posterior well inside it means the likelihood moved
    it. For this model that matters most for `sigma_z`, which decides how fast
    r* is allowed to wander and which the sweep suggests the data cannot pin.

    Unlike the other models here the prior is not stated analytically: it is
    drawn from the model itself at estimation time (see
    `base.add_scalar_priors`), so `common.prior_posterior` is handed the draws
    and takes their density. Traces saved before that was added have no
    `prior` group and are skipped with a message rather than charted against a
    guess.
    """
    if "prior" not in results.trace.groups():
        print("  no prior group in this trace, re-estimate to get prior/posterior charts")
        return 0

    prior_group = results.trace["prior"]
    return prior_posterior.plot_all(
        results.trace["posterior"],
        lambda name: (
            np.asarray(prior_group[name].values) if name in prior_group.data_vars else None
        ),
        footers={
            "lfooter": LFOOTER,
            "rfooter": RFOOTER + "Prior drawn from the model.",
        },
        labels=_PARAM_LABEL,
    )


@dataclass(frozen=True)
class _ResolutionFlags:
    """Which resolution a saved trace came from, read off its posterior vars.

    The resolution is not recorded in the trace, so it is inferred from which
    parameters exist: only the blends have `alpha_rstar`, only E and F have
    `rho_z`, only B has `tp`, and so on.
    """

    label: str
    is_blend: bool
    is_b: bool
    is_g: bool
    is_h: bool
    has_blended_z: bool
    has_alpha_hierarchical: bool


def _detect_resolution(posterior_vars: Container[Hashable]) -> _ResolutionFlags:
    """Work out which resolution produced a trace from its posterior variables."""
    has_blend = "alpha_rstar" in posterior_vars
    has_blended_z = "rho_z" in posterior_vars  # E and F use AR(1) z
    has_soe = "gamma_icp" in posterior_vars or "gamma_twi" in posterior_vars
    has_indexed_bond_obs = "tp" in posterior_vars
    has_alpha_hierarchical = "alpha_a_hyper" in posterior_vars  # G
    has_logit_alpha = "logit_alpha" in posterior_vars  # H — time-varying alpha

    is_h = has_blend and has_logit_alpha
    is_g = has_blend and has_alpha_hierarchical and not is_h
    plain_blend = has_blend and not is_g and not is_h
    is_f = plain_blend and has_blended_z and has_soe
    is_e = plain_blend and has_blended_z and not has_soe
    is_c = plain_blend and not has_blended_z
    is_b = (not has_blend) and has_indexed_bond_obs
    is_d = (not has_blend) and (not is_b) and has_soe

    label = (
        "H (blend with time-varying alpha_t via logit-RW)" if is_h
        else "G (blend + hierarchical Beta(a,b) on alpha)" if is_g
        else "F (blend + AR(1) z + SOE IS)"               if is_f
        else "E (blend + AR(1) z)"                        if is_e
        else "C (blend)"                                  if is_c
        else "B (canonical + indexed bond)"               if is_b
        else "D (canonical r* + SOE IS curve)"            if is_d
        else "A (canonical, r* = g + z)"
    )

    return _ResolutionFlags(
        label=label,
        is_blend=is_c or is_e or is_f or is_g or is_h,
        is_b=is_b,
        is_g=is_g,
        is_h=is_h,
        has_blended_z=has_blended_z,
        has_alpha_hierarchical=has_alpha_hierarchical,
    )


def _print_blend_summary(results: RStarResults, flags: _ResolutionFlags) -> None:
    """Print the alpha / k / z lines that only the blend resolutions have."""
    if flags.is_h:
        # alpha_rstar is a vector (T,) under H — show first, last and range.
        alpha_path = get_vector_var("alpha_rstar", results.trace)
        alpha_path.index = results.obs_index
        alpha_med = alpha_path.median(axis=1)
        print(f"  alpha_t (time-varying): start {alpha_med.iloc[0]:.3f}, "
              f"latest {alpha_med.iloc[-1]:.3f}, "
              f"range [{alpha_med.min():.3f}, {alpha_med.max():.3f}]")
    else:
        alpha_scalar = float(get_scalar_var("alpha_rstar", results.trace).median())
        bimodal_note = (
            "  (note: median is misleading for bimodal posteriors, see chart)"
            if flags.is_g else ""
        )
        print(f"  alpha median:  {alpha_scalar:.3f}{bimodal_note}")

    print(f"  k median:      {float(get_scalar_var('k', results.trace).median()):.3f}")

    if flags.has_alpha_hierarchical:
        a_med = float(get_scalar_var("alpha_a_hyper", results.trace).median())
        b_med = float(get_scalar_var("alpha_b_hyper", results.trace).median())
        shape_note = (
            "data prefers near-Jeffreys (sub-1)"
            if (a_med < 1 and b_med < 1) else "data prefers central-mass (>1)"
        )
        print(f"  alpha_a_hyper: {a_med:.3f}  alpha_b_hyper: {b_med:.3f}  ({shape_note})")

    if flags.has_blended_z:
        rho_med = float(get_scalar_var("rho_z", results.trace).median())
        z = results.trace["posterior"]["z_star"].stack(s=("chain", "draw")).to_numpy()
        z_median_path = pd.DataFrame(z).median(axis=1)
        print(f"  rho_z median:  {rho_med:.3f}")
        print(f"  |z| mean:      {abs(z_median_path).mean():.3f} pp"
              f"  (range [{z_median_path.min():.2f}, {z_median_path.max():.2f}])")


def _print_summary(results: RStarResults, flags: _ResolutionFlags) -> None:
    """Print the headline numbers for a loaded trace."""
    r_star = results.r_star_median()
    g = results.trend_growth_median()
    print(f"  Sample:        {results.obs_index[0]} to {results.obs_index[-1]}")
    print(f"  Resolution:    {flags.label}")
    print(f"  r* range:      [{r_star.min():.2f}, {r_star.max():.2f}]%")
    print(f"  r* latest:     {r_star.iloc[-1]:.2f}%")
    print(f"  g  latest:     {g.iloc[-1]:.2f}%")

    if flags.is_blend:
        _print_blend_summary(results, flags)
    elif flags.is_b:
        print(f"  tp median:     {float(get_scalar_var('tp', results.trace).median()):.3f}")


def run_analyse(
    prefix: str = "rstar_hlw_A",
    chart_dir: Path | str | None = None,
    resolution: str = "A",
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
    save_diagnostics(results.trace, chart_dir, prefix, model="rstar_hlw")

    flags = _detect_resolution(results.trace["posterior"].data_vars)

    if verbose:
        _print_summary(results, flags)

    caveat = _rstar_caveat(flags)
    plot_r_star(results, show=show, caveat=caveat)
    plot_trend_growth(results, show=show)
    plot_output_gap(results, show=show)
    if flags.is_blend:
        if flags.is_g:
            # Median-based decomposition is misleading under G's bimodal α
            # posterior — replace with the draw-cloud + mode-conditional version.
            plot_r_star_bimodal_decomposition(results, show=show)
        elif flags.is_h:
            # Time-varying alpha — alpha_t per period; standard decomposition
            # using a scalar alpha doesn't apply.
            plot_alpha_path(results, show=show)
        else:
            plot_r_star_decomposition(results, show=show)
        plot_g_vs_anchor(results, show=show)
        if not flags.is_h:
            # plot_alpha_posterior assumes scalar alpha — skip for H.
            plot_alpha_posterior(results, show=show)
    plot_world_rstar_overlay(results, show=show, bond_mode=flags.is_g, caveat=caveat)
    plot_prior_posterior(results, show=show)

    print(f"Charts saved to: {chart_dir}")
