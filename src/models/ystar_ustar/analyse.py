"""Charts and printed diagnostics for the joint y*/u* model.

The diagnostics matter more than the charts here. The model was built to
estimate one number, `sigma_v`, and the first question about that number is
whether the data moved it at all.
"""

import math
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd

from src.models.common import prior_posterior
from src.models.common.charts import excluded_span_style
from src.models.common.diagnostics import save_diagnostics
from src.models.ustar import analyse as ustar_analyse
from src.models.ustar.results import UStarResults
from src.models.ystar import analyse as ystar_analyse
from src.models.ystar.decompose import decompose_potential_growth, print_decomposition
from src.models.ystar.results import PotentialResults
from src.models.ystar_ustar.config import CHART_DIR
from src.models.ystar_ustar.results import JointResults, load_results

if TYPE_CHECKING:
    from collections.abc import Iterator

# Reference values from the separately estimated parents, for the comparison
# table. Both are 2026Q2 vintage and both are recorded in their MODEL_NOTES.
_YSTAR_C = 0.188
_YSTAR_SIGMA_E = 0.506
_YSTAR_POTENTIAL_GROWTH = 1.94
_YSTAR_GAP = 0.21
_YSTAR_GAP_SD = 0.188
_USTAR_BETA = 2.147
_USTAR_GAMMA = -1.148
_USTAR_LEVEL = 4.83

_LFOOTER = "Australia. Joint y* and u* model. "

# The quarters where u* is not well identified: the sample opens one quarter
# after a five-point collapse in inflation expectations, and neither the
# inflation-defined gap nor a phased anchor can place a level there. Ends
# 1999Q4, on the level rather than on the band: across 1993-98 u* averages 8.71
# against an unemployment rate of 8.90, a reported gap of -0.19 through six
# years that opened at 10.85, which is the model calling the deepest slack in
# the sample equilibrium. The band criterion is looser and would stop at
# 1995Q4, the 90% band being 2.61x its mid-sample width in 1993 and 1.45x by
# 1995, but a narrow band around a level that tracks unemployment is false
# precision rather than identification. See MODEL_NOTES, "The early sample".
UNIDENTIFIED_WINDOW = ("1993Q1", "1999Q4")

# Used only for runs saved before `build_observations` began recording where its
# series came from. A current run carries its own records and `_rfooter` reads
# those instead. The unemployment rate is 1364.0.15.003, not 6202.0: an earlier
# version of this constant said otherwise, which is the reason the records exist.
_SOURCE = "Built using: ABS 1364.0.15.003, 5206.0, 6401.0, 6457.0; NY Fed"

# How much the posterior sd must fall below the prior's before sigma_v counts as
# having been moved by the data. A judgement, not a test: at 20% the posterior is
# still overwhelmingly the prior, so this is a floor for "not identified" rather
# than a threshold for "identified".
_MIN_SHRINKAGE = 0.20

# Where the parents put the same parameter, for the reference line.
_SEPARATE_VALUE = {"sigma_e": _YSTAR_SIGMA_E, "sigma_okun": 0.485}


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


def _rfooter(results: JointResults) -> str:
    """Return the source line this run recorded, falling back for older runs."""
    return results.source_footer or _SOURCE


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
    `common.charts.excluded_span_style` so it looks the same on all of them.

    It matters most on the decomposition chart: inside the window `v`'s median
    across draws is zero, so the total gap and the inflation-defined part
    coincide exactly, and an unmarked chart reads as "inflation explained the
    pandemic perfectly" when in fact nothing there is estimated at all.
    """
    window = results.excluded_window
    if window is None:
        return []
    lo, hi = window
    style = excluded_span_style()
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
        rfooter=_rfooter(results),
        axvspan=_excluded_span(results),
        show=False,
    )


_PHILLIPS_ERAS: tuple[tuple[str, str, str], ...] = (
    ("1993Q1", "1999Q4", "1993-1999"),
    ("2000Q1", "2009Q4", "2000-2009"),
    ("2010Q1", "2019Q4", "2010-2019"),
    ("2020Q1", "2026Q4", "2020-"),
)
_ERA_COLOURS = ("tab:blue", "tab:green", "goldenrod", "crimson")


def _chart_phillips_curve(results: JointResults) -> None:
    """Draw the Phillips curve the model actually fits.

    A partial-regression plot: the equation's own regressor on the horizontal
    axis, and inflation with every non-demand term removed on the vertical, so
    the fitted line is `gamma_pi` through the origin with nothing else in the
    way. Points are coloured by era, which is how a reader can see whether the
    relationship is one line or several.
    """
    if not results.has_phillips:
        return

    frame = results.phillips_frame()
    keep = results.fitted_mask()
    frame = frame[keep]

    gamma = np.asarray(results.posterior["gamma_pi"].values).ravel()
    gamma_median = float(np.median(gamma))

    _, ax = plt.subplots()
    for (lo, hi, label), colour in zip(_PHILLIPS_ERAS, _ERA_COLOURS, strict=False):
        window = frame[lo:hi]
        if window.empty:
            continue
        ax.scatter(
            window["demand_slack"], window["inflation_ex_other"],
            s=22, color=colour, alpha=0.8, label=label, zorder=3,
        )

    grid = np.linspace(frame["demand_slack"].min(), frame["demand_slack"].max(), 50)
    lo_g, hi_g = np.percentile(gamma, [5, 95])
    ax.fill_between(grid, lo_g * grid, hi_g * grid, color="grey", alpha=0.20,
                    label="90% interval for the slope", zorder=1)
    ax.plot(grid, gamma_median * grid, color="black", linewidth=2,
            label=f"Fitted: gamma = {gamma_median:.2f}", zorder=2)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(0.0, color="grey", linewidth=0.8)

    mg.finalise_plot(
        ax,
        title="The Phillips curve as specified",
        xlabel="Unemployment gap, (u - u*) / u",
        ylabel="Quarterly inflation less anchor,\nexpectations and supply terms",
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"gamma = {gamma_median:.2f}, 90% interval [{lo_g:.2f}, {hi_g:.2f}]",
        lfooter=_LFOOTER + "Excluded quarters dropped. Axes not independent: see notes. ",
        rfooter=_rfooter(results),
        show=False,
    )


def _chart_implied_ustar(results: JointResults) -> None:
    """Show how much of the reported u* path is prior rather than likelihood.

    The grey line is the u* each quarter's inflation would imply on its own,
    from `results.implied_ustar()`. The state law's whole job is to turn that
    into something a NAIRU could plausibly be, so the question the chart answers
    is whether it does that by filtering the series or by ignoring it.
    """
    if not results.has_phillips:
        return

    ustar = results.ustar_posterior()
    implied = results.implied_ustar()
    fitted = ustar.median(axis=1)
    ratio = implied.diff().std() / fitted.diff().std()

    band = pd.DataFrame({
        "lower": ustar.quantile(0.05, axis=1),
        "upper": ustar.quantile(0.95, axis=1),
    })
    ax = mg.fill_between_plot(band, color="cornflowerblue", alpha=0.25,
                              label="u* 90% credible interval")
    mg.line_plot(
        pd.DataFrame({
            "Implied by inflation alone": implied,
            "u*": fitted,
        }),
        ax=ax,
        color=["grey", "darkorange"],
        width=[1.0, 2.5],
        style=["-", "-"],
        alpha=[0.7, 1.0],
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="What inflation alone says u* is, quarter by quarter",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "small"},
        axvspan=[*ustar_analyse._unidentified_span(), *_excluded_span(results)],  # noqa: SLF001
        lheader=(
            f"Implied series moves {ratio:.0f}x as much quarter to quarter; "
            f"correlation with u* {implied.corr(fitted):.2f}"
        ),
        lfooter=_LFOOTER + "Phillips inverted at posterior medians. ",
        rfooter=_rfooter(results),
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
        rfooter=_rfooter(results),
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

# Where a parent model put the same parameter, for the reference line.
_SEPARATE_VALUE = {
    "c": _YSTAR_C, "sigma_e": _YSTAR_SIGMA_E,
    "beta_okun": _USTAR_BETA, "gamma_pi": _USTAR_GAMMA, "sigma_okun": 0.685,
}


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


def _chart_parameter_posteriors(results: JointResults) -> int:
    """Draw one chart per estimated scalar the model reports, posterior against prior.

    The drawing is `common.prior_posterior`, shared with `ustar`. What stays
    here is the part only this model knows: which name carries which prior,
    and where a parent model put the same parameter for the reference line.
    """
    return prior_posterior.plot_all(
        results.posterior,
        lambda name: _prior_for(results, name),
        footers={"lfooter": _LFOOTER, "rfooter": _rfooter(results)},
        references=_SEPARATE_VALUE,
    )


@contextmanager
def _parent_chart_settings(results: JointResults) -> Iterator[None]:
    """Point `ystar`'s and `ustar`'s chart modules at this model, then restore them.

    Their charts carry their own footers, and one of them is actively wrong
    here: `ustar`'s says "u* from a given output gap", but in this model the gap
    is estimated rather than given. Both are overridden so every chart in this
    directory names the model that drew it.

    The right footers need no override for a current run: the parents read the
    source records off the results they are handed, which are this model's.
    Their fallback constants do need one, for a run saved before those records
    existed, or every chart here would name its parent's inputs instead.

    Restored on the way out, so a session that analyses this model and then one
    of its parents does not mislabel the parent's charts.
    """
    ystar_footer, ustar_footer = ystar_analyse._LFOOTER, ustar_analyse._LFOOTER  # noqa: SLF001
    ustar_band_footer = ustar_analyse._LFOOTER_BAND  # noqa: SLF001
    ystar_fallbacks = (
        ystar_analyse._RFOOTER, ystar_analyse._RFOOTER_CORE, ystar_analyse._RFOOTER_PRODUCTION,  # noqa: SLF001
    )
    ustar_fallback = ustar_analyse._RFOOTER  # noqa: SLF001
    ustar_excluded = ustar_analyse._EXCLUDED_WINDOW  # noqa: SLF001
    ustar_unidentified = ustar_analyse._UNIDENTIFIED_WINDOW  # noqa: SLF001

    ystar_analyse._LFOOTER = _LFOOTER  # noqa: SLF001
    ustar_analyse._LFOOTER = _LFOOTER  # noqa: SLF001
    ustar_analyse._LFOOTER_BAND = (  # noqa: SLF001
        _LFOOTER + "Band widened x2 for the imposed drift; see notes. "
    )
    ystar_analyse._RFOOTER = _SOURCE  # noqa: SLF001
    ystar_analyse._RFOOTER_CORE = _SOURCE  # noqa: SLF001
    ystar_analyse._RFOOTER_PRODUCTION = _SOURCE  # noqa: SLF001
    ustar_analyse._RFOOTER = _SOURCE  # noqa: SLF001

    # `ystar`'s chart module keeps the excluded window in a module-level global,
    # set inside its own run_analysis, which we are bypassing. Setting it here
    # is what makes its charts shade the quarters that carry no likelihood.
    ystar_analyse._EXCLUDED_WINDOW = results.excluded_window  # noqa: SLF001
    # Same for `ustar`'s charts. Its own runs exclude nothing, so this is dead
    # for `ustar` itself, but here the window is dropped from all three
    # equations and u* inside it is a prior extrapolation.
    ustar_analyse._EXCLUDED_WINDOW = (  # noqa: SLF001
        results.excluded_window if results.constants.get("exclude_scope") == "all" else None
    )
    # The early window, where u*'s level is set by the state law and by Okun
    # rather than by inflation: the 90% band runs 2.6x its mid-sample width in
    # 1993, the Phillips residuals are systematically negative until 1999, and
    # expectations do not reach the target until 1998. See MODEL_NOTES.
    ustar_analyse._UNIDENTIFIED_WINDOW = UNIDENTIFIED_WINDOW  # noqa: SLF001

    try:
        yield
    finally:
        ystar_analyse._LFOOTER = ystar_footer  # noqa: SLF001
        ustar_analyse._LFOOTER = ustar_footer  # noqa: SLF001
        ustar_analyse._LFOOTER_BAND = ustar_band_footer  # noqa: SLF001
        (
            ystar_analyse._RFOOTER,  # noqa: SLF001
            ystar_analyse._RFOOTER_CORE,  # noqa: SLF001
            ystar_analyse._RFOOTER_PRODUCTION,  # noqa: SLF001
        ) = ystar_fallbacks
        ustar_analyse._RFOOTER = ustar_fallback  # noqa: SLF001
        ustar_analyse._EXCLUDED_WINDOW = ustar_excluded  # noqa: SLF001
        ustar_analyse._UNIDENTIFIED_WINDOW = ustar_unidentified  # noqa: SLF001


def _draw_charts(
    results: JointResults,
    ystar_view: PotentialResults,
    ustar_view: UStarResults,
) -> None:
    """Write every chart for this run, the parents' and this model's own."""
    # --- The y* side, exactly what `ystar` draws for its inflation spec ---
    ystar_analyse.plot_potential(ystar_view, tag="full")
    ystar_analyse.plot_potential(ystar_view, plot_from="2015Q1", tag="recent")
    ystar_analyse.plot_actual_output_gap(ystar_view, tag="full")
    ystar_analyse.plot_actual_output_gap(ystar_view, plot_from="2015Q1", tag="recent")
    ystar_analyse.plot_inflation_defined_gap(ystar_view)
    ystar_analyse.plot_gap_composition(ystar_view)
    ystar_analyse.plot_growth_vs_potential(ystar_view, tag="full")
    ystar_analyse.plot_growth_vs_potential(ystar_view, plot_from="2015Q1", tag="recent")
    ystar_analyse.plot_gdp_growth_against_potential(ystar_view, tag="full")
    ystar_analyse.plot_gdp_growth_against_potential(
        ystar_view, plot_from="2015Q1", tag="recent",
    )
    ystar_analyse.plot_gov_growth_against_potential(ystar_view, tag="full")
    ystar_analyse.plot_gov_growth_against_potential(
        ystar_view, plot_from="2015Q1", tag="recent",
    )
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
    _chart_implied_ustar(results)
    _chart_phillips_curve(results)
    _chart_parameter_posteriors(results)


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

    chart_dir = Path(chart_dir) if chart_dir is not None else CHART_DIR
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()
    save_diagnostics(results.trace, chart_dir, prefix, model="ystar_ustar")

    with _parent_chart_settings(results):
        _draw_charts(results, ystar_view, ustar_view)


    print(f"Charts written to: {chart_dir}")
    return results
