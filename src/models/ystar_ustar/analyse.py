"""Charts and printed diagnostics for the joint y*/u* model.

The diagnostics matter more than the charts here. The model was built to
estimate one number, `sigma_v`, and the first question about that number is
whether the data moved it at all.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, halfnorm, norm

from src.models.ustar import analyse as ustar_analyse
from src.models.ustar.results import UStarResults
from src.models.ystar import analyse as ystar_analyse
from src.models.ystar.decompose import decompose_potential_growth, print_decomposition
from src.models.ystar.results import PotentialResults
from src.models.ystar_ustar.config import CHART_DIR
from src.models.ystar_ustar.results import JointResults, load_results

# Reference values from the separately estimated parents, for the comparison
# table. Both are 2026Q2 vintage and both are recorded in their MODEL_NOTES.
_YSTAR_C = 0.188
_YSTAR_SIGMA_E = 0.508
_YSTAR_POTENTIAL_GROWTH = 1.94
_YSTAR_GAP = 0.21
_YSTAR_GAP_SD = 0.188
_USTAR_BETA = 2.033
_USTAR_GAMMA = -1.055
_USTAR_LEVEL = 4.71

_LFOOTER = "Australia. Joint y* and u* model. "
_SOURCE = "Source: ABS 5206.0, 6401.0, 6202.0, 6457.0"

# How much the posterior sd must fall below the prior's before sigma_v counts as
# having been moved by the data. A judgement, not a test: at 20% the posterior is
# still overwhelmingly the prior, so this is a floor for "not identified" rather
# than a threshold for "identified".
_MIN_SHRINKAGE = 0.20

_CHAIN_COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:red")

# The tail probability reported on the variance charts. A HalfNormal(1) already
# puts 8% below this, which is the point of showing prior and posterior together.
_TAIL = 0.10

# A scalar parameter's posterior array is (chain, draw); a vector latent's is
# (chain, draw, time). This is how the two are told apart.
_SCALAR_NDIM = 2

# Where the parents put the same parameter, for the reference line.
_SEPARATE_VALUE = {"sigma_e": _YSTAR_SIGMA_E, "sigma_okun": 0.685}


def _halfnormal_moments(sigma: float) -> tuple[float, float]:
    """Return the mean and sd of HalfNormal(sigma), for the prior-posterior check."""
    return sigma * math.sqrt(2.0 / math.pi), sigma * math.sqrt(1.0 - 2.0 / math.pi)


def _print_sigma_v_check(results: JointResults, sigma_v_prior: float) -> None:
    """Report whether the data moved sigma_v away from its prior."""
    if results.gap_spec == "cycle":
        rho = float(np.asarray(results.posterior["rho_gap"].values).mean())
        print("\n--- Gap specification: free AR(1) cycle ---")
        print(f"  rho_gap {rho:.3f}   (sigma_c = 0.60 imposed as the innovation sd)")
        print("  No defined/free split: inflation observes the gap rather than defining it.")
        return
    print("\n--- Did the data move sigma_v? ---")
    if not results.has_free_gap:
        print("  No free gap component in this run (v is off).")
        return
    if "sigma_v" not in results.posterior:
        print(f"  sigma_v was imposed at {results.constants.get('sigma_v')}.")
        return

    draws = np.asarray(results.posterior["sigma_v"].values).ravel()
    prior_mean, prior_sd = _halfnormal_moments(sigma_v_prior)
    post_mean, post_sd = float(draws.mean()), float(draws.std())
    shrink = 1.0 - post_sd / prior_sd
    print(f"  prior  HalfNormal({sigma_v_prior:g}) :  mean {prior_mean:.3f}, sd {prior_sd:.3f}")
    print(f"  posterior              :  mean {post_mean:.3f}, sd {post_sd:.3f}")
    print(f"  sd shrinkage vs prior  :  {shrink:+.1%}")
    if shrink < _MIN_SHRINKAGE:
        print("  ** The posterior is close to the prior. sigma_v is NOT identified here,")
        print("     which means the joint model has failed at the one thing it was for.")
    else:
        print("  The data have moved it. Read the variance shares below.")


def _print_gap_composition(results: JointResults) -> None:
    """Report the gap's split between its defined and free parts."""
    shares = results.gap_variance_share()
    if shares:
        print("\n--- What is the gap made of? ---")
        print(f"  defined, c x (pi - anchor) : {shares['defined']:6.1%}")
        print(f"  free, v                    : {shares['free']:6.1%}")
        print(f"  covariance term            : {shares['covariance']:+6.1%}")

    gap = results.output_gap_posterior().median(axis=1)
    defined = results.defined_gap_posterior().median(axis=1)
    print(f"\n  sd(gap)         {gap.std():.3f}   against ystar's defined gap at {_YSTAR_GAP_SD}")
    print(f"  autocorr(gap)   {gap.autocorr(1):+.3f}")
    if results.gap_spec != "cycle":
        print(f"  sd(defined)     {defined.std():.3f}")
    if results.has_free_gap:
        print(f"  sd(v)           {results.free_gap_posterior().median(axis=1).std():.3f}")

    if "beta_okun" in results.posterior:
        print("\n--- The moment that identifies sigma_v ---")
        print(f"  corr(e_c, e_o) on medians  : {results.residual_correlation():+.3f}")
        print("  (a covariance away from zero is what separates v from e_c;")
        print("   near zero means GDP's residual and unemployment's do not share a cycle)")


def _print_headline(results: JointResults) -> None:
    """Report the 2026Q2 endpoint against the separately estimated parents."""
    print("\n--- Headline, latest quarter ---")
    growth = results.potential_growth_posterior().median(axis=1)
    gap = results.output_gap_posterior().median(axis=1)
    ugap = results.unemployment_gap_posterior().median(axis=1)
    ustar = results.ustar_posterior().median(axis=1)
    print(f"  potential growth (y/y) : {growth.iloc[-1]:6.2f}   (ystar alone: {_YSTAR_POTENTIAL_GROWTH})")
    print(f"  output gap             : {gap.iloc[-1]:+6.2f}   (ystar alone: {_YSTAR_GAP:+.2f})")
    print(f"  u*                     : {ustar.iloc[-1]:6.2f}   (ustar alone: {_USTAR_LEVEL})")
    print(f"  u - u*                 : {ugap.iloc[-1]:+6.2f}")


def _print_comparison(results: JointResults) -> None:
    """Report each shared parameter against its separately estimated value."""
    print("\n--- Against the models this replaces ---")
    rows = [("c", _YSTAR_C), ("sigma_e", _YSTAR_SIGMA_E),
            ("beta_okun", _USTAR_BETA), ("gamma_pi", _USTAR_GAMMA)]
    if results.gap_spec == "cycle":
        rows = [(k, v) for k, v in rows if k not in ("c", "gamma_pi")]
    print(f"  {'parameter':<14}{'joint':>10}{'separate':>12}{'change':>10}")
    for name, separate in rows:
        if name not in results.posterior:
            continue
        joint = float(np.asarray(results.posterior[name].values).mean())
        print(f"  {name:<14}{joint:>10.3f}{separate:>12.3f}{joint - separate:>+10.3f}")
    print()


def print_diagnostics(results: JointResults, sigma_v_prior: float = 1.0) -> None:
    """Print the parameter table and the checks this model exists to run."""
    print("\n" + "=" * 72)
    print("JOINT y*/u* — POSTERIOR")
    print("=" * 72)
    print(results.summary().to_string())

    _print_sigma_v_check(results, sigma_v_prior)
    _print_gap_composition(results)
    _print_headline(results)
    _print_comparison(results)




# ---------------------------------------------------------------------------
# Charts
#
# Drawn by calling `ystar`'s and `ustar`'s own plotting functions rather than
# reimplementing them. Those modules carry a good deal of care this model has no
# reason to duplicate: quarterly-axis handling, the excluded-window shading, the
# off-scale annotation on the gap composition, and the band-widening convention
# on u*. Redrawing them here would mean maintaining two versions that drift.
#
# The two parents' results classes read from the trace by variable name and from
# `obs` by key, so the adapters below only have to remap the observation keys.
# The joint trace already carries every latent each of them looks for.
# ---------------------------------------------------------------------------


def _as_ystar_results(results: JointResults) -> PotentialResults:
    """View the joint run as a `ystar` result, for `ystar`'s charts.

    `PotentialResults.spec` reads the trace's contents and resolves to
    "inflation" here, because `c` is present and the labour and production
    latents are not. That is the correct reading: the potential block *is*
    `ystar`'s inflation spec, imported rather than copied.

    One difference worth knowing when reading the charts. `ystar`'s
    `output_gap_posterior` returns the trace's `output_gap`, which in this model
    is `c·d + v` rather than `c·d` alone. So "the gap" on those charts is the
    joint model's whole gap, which is the object it exists to estimate. The
    inflation-defined part alone is on the decomposition chart below.
    """
    return PotentialResults(
        trace=results.trace,
        obs={"log_gdp": results.obs["log_gdp"], "pi": results.obs["pi_gap"]},
        obs_index=results.obs_index,
        constants=results.constants,
    )


def _as_ustar_results(results: JointResults) -> UStarResults:
    """View the joint run as a `ustar` result, for `ustar`'s charts.

    `ustar` names the Phillips curve's left-hand side `pi`; here it is `pi_qtr`,
    because this model carries two inflation horizons and has to say which. The
    gap moments it would normally read from a completed `ystar` run are supplied
    from this model's own posterior, which is the whole point of joining them.
    """
    gap = results.output_gap_posterior()
    obs = {
        "u": results.obs["u"],
        "gap_mean": gap.mean(axis=1).to_numpy(),
        "gap_sd": gap.std(axis=1).to_numpy(),
    }
    for source, target in (("pi_qtr", "pi"), ("pi_exp", "pi_exp"),
                           ("d4pm", "d4pm"), ("gscpi", "gscpi")):
        if source in results.obs:
            obs[target] = results.obs[source]

    return UStarResults(
        trace=results.trace,
        obs=obs,
        obs_index=results.obs_index,
        constants=results.constants,
    )


def _excluded_span(results: JointResults) -> list[dict[str, object]]:
    """Return an axvspan dict shading the quarters that carry no likelihood.

    Needed on the two charts written here, because `ystar`'s and `ustar`'s own
    charts get it from their module-level global instead. Styled from
    `ystar.analyse.excluded_span_style` so it looks the same on all of them.

    It matters most on the decomposition chart: inside the window `v`'s median
    across draws is zero, so the total gap and the inflation-defined part
    coincide exactly, and an unmarked chart reads as "inflation explained the
    pandemic perfectly" when in fact nothing there is estimated at all.
    """
    window = results.excluded_window
    if window is None:
        return []
    lo, hi = window
    style = ystar_analyse.excluded_span_style()
    return [{
        "xmin": pd.Period(lo, freq="Q"),
        "xmax": pd.Period(hi, freq="Q"),
        **style,
        "label": f"{style['label']}, {lo}-{hi}",
    }]


def _chart_gap_decomposition(results: JointResults) -> None:
    """Draw the chart neither parent can: how much of the gap is not inflation.

    `ystar` cannot draw it because it has no free component to separate out, and
    `ustar` cannot because it receives the gap as data.
    """
    frame = pd.DataFrame({
        "Output gap": results.output_gap_posterior().median(axis=1),
        "Defined by inflation": results.defined_gap_posterior().median(axis=1),
    })
    colours = ["black", "darkorange"]
    widths: list[float] = [2.0, 1.5]
    styles = ["-", "--"]
    if results.has_free_gap:
        frame["Free component, v"] = results.free_gap_posterior().median(axis=1)
        colours.append("teal")
        widths.append(1.5)
        styles.append("-")

    shares = results.gap_variance_share()
    mg.line_plot_finalise(
        frame,
        title="The output gap, and how much of it inflation explains",
        ylabel="Per cent of potential",
        width=widths,
        color=colours,
        style=styles,
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"Inflation-defined share of gap variance: {shares['defined']:.0%}",
        lfooter=_LFOOTER + "Medians. Shaded: no likelihood. ",
        rfooter=_SOURCE,
        axvspan=_excluded_span(results),
        show=False,
    )


def _chart_residuals(results: JointResults) -> None:
    """Draw the two residuals whose covariance identifies sigma_v.

    The Okun residual is sign-flipped, so a shared cycle shows as the lines
    moving together. That is the whole diagnostic: if these co-move, GDP's
    "noise" and unemployment's "noise" are the same business cycle, and the free
    gap component is real rather than a prior.
    """
    if "beta_okun" not in results.posterior:
        return
    frame = pd.DataFrame({
        "GDP residual, e_c": results.gdp_residual_posterior().median(axis=1),
        "Okun residual, -e_o": -results.okun_residual_posterior().median(axis=1),
    })
    mg.line_plot_finalise(
        frame,
        title="The residuals that identify the free gap",
        ylabel="Per cent / percentage points",
        width=[1.8, 1.8],
        color=["navy", "darkorange"],
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"corr(e_c, e_o) = {results.residual_correlation():+.2f}",
        lfooter=_LFOOTER + "Okun residual sign-flipped. Shaded: no likelihood. ",
        rfooter=_SOURCE,
        axvspan=_excluded_span(results),
        show=False,
    )


# Every estimated scalar, with the prior it was sampled under. Kept beside the
# priors in `estimate.py` deliberately: if the two drift apart the charts lie,
# so a change in one is meant to be an obvious omission in the other.
#
# Entries are (kind, mu, sd). "half" ignores mu. Priors whose scale is a config
# setting read it from the run's recorded constants instead, see `_prior_for`.
_PRIORS: dict[str, tuple[str, float, float]] = {
    "sigma_e": ("half", 0.0, 1.0),
    "sigma_okun": ("half", 0.0, 1.0),
    "epsilon_pi": ("half", 0.0, 0.25),
    "gamma_pi": ("normal", -1.5, 1.0),
    "beta_pi": ("normal", 0.5, 0.3),
    "rho_pi": ("normal", 0.0, 0.1),
    "xi_gscpi": ("normal", 0.0, 0.1),
    "kappa_gap": ("normal", 0.5, 0.5),
    "initial_trend_growth": ("normal", 0.90, 0.40),
    "rho_gap": ("normal", 0.8, 0.2),
}

_CHAIN_COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:red")

# Where a parent model put the same parameter, for the reference line.
_SEPARATE_VALUE = {
    "c": _YSTAR_C, "sigma_e": _YSTAR_SIGMA_E,
    "beta_okun": _USTAR_BETA, "gamma_pi": _USTAR_GAMMA, "sigma_okun": 0.685,
}

# The tail probability reported on the variance charts. A HalfNormal(1) already
# puts 8% below this, which is the point of showing prior and posterior together.
_TAIL = 0.10

# A scalar parameter's posterior array is (chain, draw); a vector latent's is
# (chain, draw, time). This is how the two are told apart.
_SCALAR_NDIM = 2


def _prior_for(results: JointResults, name: str) -> tuple[str, float, float] | None:
    """Return (kind, mu, sd) for a parameter's prior, or None if unknown.

    Three priors depend on run settings rather than being fixed in the source,
    so they are read from the constants the run recorded.
    """
    constants = results.constants
    if name == "sigma_v":
        return ("half", 0.0, float(constants.get("sigma_v_prior", 1.0)))
    if name == "c":
        return (
            ("normal", 0.0, 2.0) if constants.get("two_sided_c")
            else ("half", 0.0, 2.0)
        )
    if name == "beta_okun":
        return ("normal", 0.5, float(constants.get("beta_okun_prior_sd", 0.5)))
    return _PRIORS.get(name)


def _prior_pdf(kind: str, mu: float, sd: float, grid: np.ndarray) -> np.ndarray:
    """Evaluate the prior density on a grid."""
    return halfnorm.pdf(grid, scale=sd) if kind == "half" else norm.pdf(grid, loc=mu, scale=sd)


def _chart_parameter_posterior(results: JointResults, name: str) -> None:
    """Draw one parameter's posterior against its prior, chain by chain.

    Worth a chart per parameter rather than a summary table, because the table
    hides what matters. `sigma_okun` reports a tidy mean and a plausible
    interval while its four chains peak in four different places, which is
    visible here and invisible in a row of numbers. The prior is drawn alongside
    because "the posterior sits at x" means nothing without knowing where the
    prior already put it.

    Built with raw matplotlib rather than `mg.line_plot`, because the x axis is
    a parameter grid and mgplot requires a PeriodIndex or RangeIndex. It still
    goes through `mg.finalise_plot`, so styling, footers and the filename match
    every other chart in the directory.
    """
    prior = _prior_for(results, name)
    if prior is None:
        return
    kind, mu, sd = prior

    draws = np.asarray(results.posterior[name])
    flat = draws.ravel()
    lo = min(0.0 if kind == "half" else mu - 3.5 * sd, float(flat.min()))
    hi = max(mu + 3.5 * sd, float(flat.max()))
    pad = 0.08 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 500)

    _, ax = plt.subplots(figsize=(9, 5))
    label = f"Prior, {'HalfNormal' if kind == 'half' else 'Normal'}"
    label += f"({sd:g})" if kind == "half" else f"({mu:g}, {sd:g})"
    ax.plot(grid, _prior_pdf(kind, mu, sd, grid), color="grey", ls="--", lw=1.8, label=label)
    for i, colour in zip(range(draws.shape[0]), _CHAIN_COLOURS, strict=False):
        ax.plot(grid, gaussian_kde(draws[i])(grid), color=colour, lw=1.0, ls=":",
                label=f"Chain {i}")
    ax.plot(grid, gaussian_kde(flat)(grid), color="black", lw=2.5, label="Posterior")
    if name in _SEPARATE_VALUE:
        ax.axvline(_SEPARATE_VALUE[name], color="darkred", lw=1.2, ls="-.",
                   label=f"separately estimated, {_SEPARATE_VALUE[name]:g}")
    ax.set_xlim(grid[0], grid[-1])

    header = f"posterior mean {flat.mean():.3f}, prior mean {mu if kind == 'normal' else sd * 0.798:.3f}"
    mg.finalise_plot(
        ax,
        title=f"{name}: posterior against prior",
        xlabel=name,
        ylabel="Density",
        legend={"loc": "best", "fontsize": "x-small"},
        lheader=header,
        lfooter=_LFOOTER + "Dotted lines are individual chains. ",
        rfooter=_SOURCE,
        show=False,
    )


def _chart_parameter_posteriors(results: JointResults) -> None:
    """Draw one chart per estimated scalar the model reports."""
    for name in results.posterior.data_vars:
        if results.posterior[name].ndim == _SCALAR_NDIM and _prior_for(results, str(name)) is not None:
            _chart_parameter_posterior(results, str(name))


def run_analysis(
    prefix: str = "ystar_ustar",
    sigma_v_prior: float = 1.0,
    chart_dir: Path | str | None = None,
) -> JointResults:
    """Load a saved run, print the diagnostics, write every chart.

    `chart_dir` defaults to `charts/YStarUStar`. Pass a different one for a
    variant run: the charting clears its directory first, so analysing a
    variant into the default would delete the headline run's charts.
    """
    results = load_results(prefix=prefix)
    print_diagnostics(results, sigma_v_prior=sigma_v_prior)

    ystar_view = _as_ystar_results(results)
    ustar_view = _as_ustar_results(results)

    # The parents' charts carry their parents' footers, and one of them is
    # actively wrong here: `ustar`'s says "u* from a given output gap", but in
    # this model the gap is estimated rather than given. Both are overridden so
    # every chart in this directory names the model that drew it. Restored
    # afterwards, so a session that analyses this model and then one of its
    # parents does not mislabel the parent's charts.
    ystar_footer, ustar_footer = ystar_analyse._LFOOTER, ustar_analyse._LFOOTER  # noqa: SLF001
    ustar_band_footer = ustar_analyse._LFOOTER_BAND  # noqa: SLF001
    ystar_analyse._LFOOTER = _LFOOTER  # noqa: SLF001
    ustar_analyse._LFOOTER = _LFOOTER  # noqa: SLF001
    ustar_analyse._LFOOTER_BAND = (  # noqa: SLF001
        _LFOOTER + "Band widened x2 for the imposed drift; see notes. "
    )

    # `ystar`'s chart module keeps the excluded window in a module-level global,
    # set inside its own run_analysis, which we are bypassing. Setting it here
    # is what makes its charts shade the quarters that carry no likelihood.
    ystar_analyse._EXCLUDED_WINDOW = results.excluded_window  # noqa: SLF001
    # Same for `ustar`'s charts. Its own runs exclude nothing, so this is dead
    # for `ustar` itself, but here the window is dropped from all three
    # equations and u* inside it is a prior extrapolation.
    ustar_excluded = ustar_analyse._EXCLUDED_WINDOW  # noqa: SLF001
    ustar_analyse._EXCLUDED_WINDOW = (  # noqa: SLF001
        results.excluded_window if results.constants.get("exclude_scope") == "all" else None
    )

    chart_dir = Path(chart_dir) if chart_dir is not None else CHART_DIR
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    # --- The y* side, exactly what `ystar` draws for its inflation spec ---
    ystar_analyse.plot_potential(ystar_view, tag="full")
    ystar_analyse.plot_potential(ystar_view, plot_from="2015Q1", tag="recent")
    ystar_analyse.plot_actual_output_gap(ystar_view, tag="full")
    ystar_analyse.plot_actual_output_gap(ystar_view, plot_from="2015Q1", tag="recent")
    ystar_analyse.plot_inflation_defined_gap(ystar_view)
    ystar_analyse.plot_gap_composition(ystar_view)
    ystar_analyse.plot_growth_vs_potential(ystar_view, tag="full")
    ystar_analyse.plot_growth_vs_potential(ystar_view, plot_from="2015Q1", tag="recent")
    ystar_analyse.plot_trend_growth(ystar_view)

    # The post-modelling split of potential growth into hours and productivity.
    # It loads labour force data, so it is the only part of the analysis that
    # touches ABS sources directly.
    decomposition = decompose_potential_growth(ystar_view)
    print_decomposition(decomposition)
    ystar_analyse.plot_growth_accounting(decomposition)
    ystar_analyse.plot_growth_wedge(decomposition)
    ystar_analyse.plot_growth_contributions(decomposition)

    # --- The u* side, exactly what `ustar` draws ---
    ustar_analyse.plot_ustar(ustar_view)
    ustar_analyse.plot_ustar(ustar_view, shade_inflation=True, tag="inflation")
    ustar_analyse.plot_ugap(ustar_view)
    ustar_analyse.plot_ustar_components(ustar_view)
    if results.has_phillips:
        ustar_analyse.plot_inflation_decomposition(ustar_view)

    # --- What neither parent can draw ---
    if results.gap_spec != "cycle":
        # Nothing to decompose: the gap is one state, and the whole point of the
        # cycle spec is that inflation does not own a share of it.
        _chart_gap_decomposition(results)
    _chart_residuals(results)
    _chart_parameter_posteriors(results)

    ystar_analyse._LFOOTER = ystar_footer  # noqa: SLF001
    ustar_analyse._LFOOTER = ustar_footer  # noqa: SLF001
    ustar_analyse._LFOOTER_BAND = ustar_band_footer  # noqa: SLF001
    ustar_analyse._EXCLUDED_WINDOW = ustar_excluded  # noqa: SLF001

    print(f"Charts written to: {chart_dir}")
    return results
