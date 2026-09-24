"""Charts and printed diagnostics for the ustar model.

One object is being estimated, so there are few charts: u* against the
unemployment rate, and the gap between them. The printed diagnostics carry the
weight, because the question that matters is not what u* is but whether the
data placed it there.
"""

from pathlib import Path
from typing import Any

import mgplot as mg
import pandas as pd

from src.data.inflation import get_trimmed_mean_qrtly
from src.models.common import chart_annotations, prior_posterior
from src.models.common.chart_annotations import Window
from src.models.common.charts import excluded_span_style, ustar_structure_note
from src.models.common.diagnostics import save_diagnostics
from src.models.ustar.results import DEFAULT_CHART_BASE, UStarResults, load_results
from src.utilities.rate_conversion import annualize

CHART_DIR = DEFAULT_CHART_BASE / "UStar"

# Used only for runs saved before `build_observations` began recording where its
# series came from. A current run carries its own records and `_rfooter` reads
# those instead, which is also how a chart drawn from the joint y*/u* run names
# that model's sources rather than this one's. The GSCPI is a Phillips curve
# input here and was missing from this line entirely.
_RFOOTER = "Built using: ABS 1364.0.15.003, 5206.0, 6401.0, 6457.0; NY Fed"
# The fixed part of the left footer. `run_analysis` attaches the full footer per
# run, carrying the structure imposed on u* as well.
_MODEL = "Australia. ustar model. "
# Short because the structure it refers to is now named immediately before it
# in the same footer, and the long form ran into the source line on the right.
# Only for charts that actually draw a band. The decomposition chart is bars
# and a line built from median parameters, with no interval on it to widen.
_BAND_NOTE = "Band x2; see notes. "

# Two windows a run may attach (see `common.chart_annotations`):
#
# The EXCLUDED window, quarters that carried no likelihood. `ustar` never
# excludes anything, so its own runs attach none. It exists because the joint
# y*/u* model reuses these plotting functions and *does* exclude the pandemic
# quarters from all three of its equations, which makes u* there a prior
# extrapolation rather than an estimate. Without it the u* chart draws a
# confident line through six quarters nothing was fitted to.
#
# The UNIDENTIFIED window, the early quarters where u* is placed by the state
# law rather than by inflation, shaded so the chart does not read as a
# confident estimate there. Attached by whichever model has the evidence for it.

# The window this model's own diagnostics support, applied in `run_analysis`.
# The joint model sets its own, to the same dates and on its own evidence.
#
# Ends 1999Q4, on the level rather than on the band. Across 1993-98 u* averages
# 8.69 against an unemployment rate of 8.90, so the model reports a gap of
# -0.22 through six years that opened with unemployment at 10.85: it is saying
# the labour market was at equilibrium in the deepest slack of the sample. The
# band criterion is looser and would stop at 1995Q4, the 90% band being 2.79x
# its mid-sample width in 1993, 1.53x by 1995 and 1.36x by 1996, but a narrow
# band around a level that tracks unemployment is false precision rather than
# identification. Independent readings of the same years differ by 2.5 points
# and close to within 0.5 only by 2000Q2.
UNIDENTIFIED_WINDOW = ("1993Q1", "1999Q4")

# Orange rather than the excluded window's yellow, so the two are told apart at
# a glance where both appear. Low alpha: it sits under the u* line, which is
# darkorange itself, and must not compete with it.
_UNIDENTIFIED_SPAN: dict[str, Any] = {
    "color": "darkorange",
    "alpha": 0.12,
    "label": "u* not well identified",
}


def unidentified_span(window: Window | None) -> list[dict[str, Any]]:
    """Return an axvspan dict for the weakly identified early window, or nothing.

    It carries its own legend label and no footer note, for the reason
    `_excluded_span` gives: a footer would be a quieter second statement of what
    the legend already says.
    """
    if window is None:
        return []
    lo, hi = window
    return [{
        "xmin": pd.Period(lo, freq="Q"),
        "xmax": pd.Period(hi, freq="Q"),
        **_UNIDENTIFIED_SPAN,
        "label": f"{_UNIDENTIFIED_SPAN['label']}, {lo}-{hi}",
    }]


def _excluded_span(window: Window | None) -> list[dict[str, Any]]:
    """Return an axvspan dict marking the unfitted window, or nothing.

    Styled from `common.charts.excluded_span_style` rather than restyled here,
    so the pandemic window looks identical on every chart in the package.
    Copying the styling into each package is how it would drift.

    It carries its own legend label, which is why no footer note is added: a
    footer would be a second, quieter statement of the same thing, and on the
    inflation-shaded chart the reader has to tell this span apart from the
    band-breach spans by looking at it.
    """
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


def _with_excluded(
    kwargs: dict[str, Any], results: UStarResults, *, unidentified: bool = True,
) -> dict[str, Any]:
    """Add the run's unfitted-window and weakly-identified markers to finalise kwargs.

    `unidentified` is off for the inflation-shaded chart, where a third block
    of colour over the first seven years sits on top of the red and blue
    band-breach shading and makes both unreadable. The window is still marked
    on the plain u* chart beside it.
    """
    unidentified_window = chart_annotations.window(results, chart_annotations.UNIDENTIFIED_WINDOW)
    excluded_window = chart_annotations.window(results, chart_annotations.EXCLUDED_WINDOW)
    spans = (unidentified_span(unidentified_window) if unidentified else []) + _excluded_span(excluded_window)
    if not spans:
        return kwargs
    existing = kwargs.get("axvspan") or []
    kwargs["axvspan"] = [*spans, *existing] if isinstance(existing, list) else [*spans, existing]
    return kwargs

# The Phillips curve as drawn, with each term labelled by its bar colour.
# Written out here rather than imported from `nairu.analysis`: that builder is a
# private helper carrying the regime and wage variants this model does not have,
# and this equation is fixed, so a literal is clearer than a constructor call.
_PHILLIPS_EQUATION = (
    r"$\pi_t = \underbrace{\pi^{target}}_{\mathrm{grey}}"
    r" + \underbrace{\beta\,(\pi^e_t - \pi^{target})}_{\mathrm{purple}}"
    r" + \underbrace{\gamma\frac{U_t - U^*_t}{U_t}}_{\mathrm{orange}}"
    r" + \underbrace{\rho\Delta_4\rho^m_t + \xi\cdot GSCPI^2}_{\mathrm{blue}}"
    r" + \underbrace{\varepsilon_t}_{\mathrm{light\ blue}}$"
)

# The drawn band is the posterior band scaled about its median by this factor,
# to stand in for uncertainty about the IMPOSED STRUCTURE, which the posterior
# cannot express. u* is a spline with a knot placed by hand, so the interval
# answers "where is u* given this shape" and says nothing about the shape.
# Under `--ustar-structure decay` the imposed thing is `sigma_ustar` instead, and
# same argument applies to the drift rate.
#
# TWO IS INHERITED, NOT RE-DERIVED. It was calibrated against the decay law's
# `sigma_ustar` sweep: the conditional band was 0.48pp at the endpoint, u*
# moved 0.39pp across sigma_ustar from 0.024 to 0.05, and the union of the
# conditional bands over that range was about 0.89pp, which doubling
# reproduced. No equivalent calibration exists for the spline, and the two
# obvious candidates disagree: varying the knot count moves u* by 0.05pp over
# the sample, which would argue for less than 2, while at 1993Q1 the spread
# across specifications is 2.87pp, which would argue for far more.
#
# So it is a convention that errs wide in the settled part of the sample and
# nowhere near wide enough in the early part, where the shaded window is the
# warning instead. It is an approximation, not a posterior, and every chart
# drawn with it says so in its left footer.
_BAND_WIDEN = 2.0

_BAND_KWARGS: dict[str, Any] = {
    "color": "cornflowerblue",
    "alpha": 0.25,
    # The widening is disclosed in the footer rather than here: the legend
    # names the series, and the caveat crowds it.
    "label": "90% credible interval",
}


def _band(posterior: pd.DataFrame, widen: float = _BAND_WIDEN) -> pd.DataFrame:
    """Return the 5th and 95th posterior percentiles, scaled about the median.

    `widen` = 1 gives the posterior band itself. The default widens it to stand
    in for uncertainty about the imposed structure; see `_BAND_WIDEN`.
    """
    median = posterior.median(axis=1)
    lower = posterior.quantile(0.05, axis=1)
    upper = posterior.quantile(0.95, axis=1)
    return pd.DataFrame({
        "lower": median - widen * (median - lower),
        "upper": median + widen * (upper - median),
    })


def print_diagnostics(results: UStarResults) -> None:
    """Print the parameters and the checks that would show the model failing."""
    print("\nPosterior summary")
    print("-" * 70)
    print(results.summary().to_string())

    print("\nSign tests")
    print("-" * 70)
    print(f"  {'P(beta > 0)  Okun holds':<34} {results.prob_beta_positive():6.1%}")
    if results.has_phillips:
        print(f"  {'P(gamma < 0) slack disinflates':<34} {results.prob_gamma_negative():6.1%}")

    print("\nIs u* placed by the data or by the prior?")
    print("-" * 70)
    for key, value in results.ustar_variation().items():
        print(f"  {key:<34} {value:6.2f}")
    print("  A 'sd of du*' at the imposed sigma_ustar means u* wanders as freely")
    print("  as the prior allows and the data are not holding it.")

    # Positional, not .loc[Period]: the pandas stubs do not accept a Period as
    # a .loc key, and the three series share one index by construction.
    recent = pd.DataFrame({
        "u": results.unemployment(),
        "ustar": results.ustar_median(),
        "ugap": results.ugap_median(),
    }).tail(8)
    print("\nRecent estimates")
    print("-" * 70)
    print(f"  {'':<10}{'u':>8}{'u*':>8}{'u - u*':>10}")
    for period, row in recent.iterrows():
        print(f"  {period!s:<10}{row['u']:8.2f}{row['ustar']:8.2f}{row['ugap']:10.2f}")


# The RBA's target band. Shading marks quarters where inflation sat outside it
# altogether, which is the bank's own definition of departing from the central
# tendency rather than a threshold chosen here.
_INFLATION_HIGH = 3.0
_INFLATION_LOW = 2.0

# Shading intensity ramps with the size of the breach, from just visible at the
# band edge to full at `_SHADE_FULL` points beyond it.
_SHADE_ALPHA_MIN = 0.03
_SHADE_ALPHA_MAX = 0.25
_SHADE_FULL = 2.0


def _inflation_regime_spans(index: pd.PeriodIndex) -> list[dict[str, Any]]:
    """Return axvspan dicts shading quarters where the trimmed mean left the band.

    Red above, blue below, nothing inside 2-3%.

    The series is the **quarterly trimmed mean, annualised**, which is the
    same basis the model's Phillips curve is written on, so the shading marks
    the quarters the equation is actually reading rather than a four-quarter
    average of them. It discriminates: scored against the sign of the
    unemployment gap over the 64 quarters outside the band, the models
    separate by up to 11 points on this measure and by 4 on the year-ended
    one. The cost is more, shorter blocks, since the quarterly rate crosses
    the thresholds several times a year.

    One caveat the label has to carry: this is the **trimmed mean**, not
    headline CPI, while the RBA's 2-3% band is a headline-CPI target, so it
    marks where the core measure sat outside the band rather than where the
    target was missed.

    **Shaded by size, not by a threshold.** Each quarter gets its own span with
    an alpha proportional to how far outside the band it sat, because the
    binary version gave a quarter at 3.1 the same weight as one at 7.4 and the
    chart became a picket fence. Half the out-of-band quarters are within 0.5
    of an edge and a quarter of them within 0.25, against a maximum deviation
    of 4.40, so most of that fence was inflation grazing the boundary.

    Ramped from `_SHADE_ALPHA_MIN` at the edge to `_SHADE_ALPHA_MAX` at
    `_SHADE_FULL`, flat above. 94% of out-of-band quarters sit inside the ramp,
    so the cap only holds back the 2022-23 peak from drowning everything else.
    """
    quarterly_rate = get_trimmed_mean_qrtly().data.astype(float)
    inflation = (((1 + quarterly_rate / 100) ** 4 - 1) * 100).reindex(index)

    spans: list[dict[str, Any]] = []
    step = 1  # one quarter, so each span covers the quarter it belongs to
    for period, raw in zip(index, inflation.to_numpy(), strict=True):
        value = float(raw)
        if pd.isna(value) or _INFLATION_LOW <= value <= _INFLATION_HIGH:
            continue
        high = value > _INFLATION_HIGH
        deviation = value - _INFLATION_HIGH if high else _INFLATION_LOW - value
        weight = min(deviation / _SHADE_FULL, 1.0)
        spans.append({
            "xmin": period,
            "xmax": period + step,
            "color": "tab:red" if high else "tab:blue",
            "alpha": _SHADE_ALPHA_MIN + weight * (_SHADE_ALPHA_MAX - _SHADE_ALPHA_MIN),
            "zorder": 0,
            "linewidth": 0,
        })

    return spans


def _rfooter(results: UStarResults) -> str:
    """Return the source line this run recorded, falling back for older runs."""
    return results.source_footer or chart_annotations.text(results, chart_annotations.RFOOTER_FALLBACK, _RFOOTER)


def _lfooter(results: UStarResults) -> str:
    """Return this run's left footer, or the bare model name."""
    return chart_annotations.text(results, chart_annotations.LFOOTER, _MODEL)


def _lfooter_band(results: UStarResults) -> str:
    """Return this run's left footer for a banded chart."""
    return chart_annotations.text(results, chart_annotations.LFOOTER_BAND, _MODEL + _BAND_NOTE)


def plot_ustar(results: UStarResults, shade_inflation: bool = False, tag: str = "") -> None:
    """u* against the observed unemployment rate, with a credible band.

    With `shade_inflation`, the background marks the quarters where quarterly
    annualised trimmed mean inflation sat outside the RBA's 2-3% band, so u*
    can be read against the quarters the Phillips curve is actually reading.

    Note when interpreting it that the Phillips curve fits inflation with
    `gamma x u_gap` and gamma is negative, so the estimation is not neutral
    about where the shading and the gap coincide.
    """
    ustar = results.ustar_posterior()

    ax = mg.fill_between_plot(_band(ustar), **_BAND_KWARGS)
    mg.line_plot(
        pd.DataFrame({
            "Unemployment rate": results.unemployment(),
            "u*": ustar.median(axis=1),
        }),
        ax=ax,
        color=["black", "darkorange"],
        width=[1.5, 2],
        style=["-", "--"],
        annotate=True,
        rounding=2,
    )
    finalise_kwargs: dict[str, Any] = {
        "title": "u* and the unemployment rate",
        "ylabel": "Per cent",
        "legend": {"loc": "best", "fontsize": "small"},
        "lheader": "u* is the unemployment rate consistent with output at potential",
        "rfooter": _rfooter(results),
        "lfooter": _lfooter_band(results),
        "show": False,
    }
    if tag:
        finalise_kwargs["tag"] = tag
    if shade_inflation:
        finalise_kwargs["axvspan"] = _inflation_regime_spans(results.obs_index)
        finalise_kwargs["lheader"] = (
            f"Shaded where quarterly annualised trimmed mean inflation sat outside "
            f"{_INFLATION_LOW:g}-{_INFLATION_HIGH:g}%: red above, blue below"
        )
    mg.finalise_plot(ax, **_with_excluded(finalise_kwargs, results, unidentified=not shade_inflation))


def plot_ugap(results: UStarResults) -> None:
    """Plot the unemployment gap, u - u*, with a credible band."""
    ugap = results.ugap_posterior()

    ax = mg.fill_between_plot(_band(ugap), **_BAND_KWARGS)
    mg.line_plot(
        ugap.median(axis=1).rename("u - u*"),
        ax=ax,
        color=["darkorange"],
        width=2,
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(ax, **_with_excluded({
        "title": "Unemployment gap",
        "ylabel": "Percentage points",
        "y0": True,
        "legend": {"loc": "best", "fontsize": "small"},
        "lheader": "Below zero is a tight labour market",
        "rfooter": _rfooter(results),
        "lfooter": _lfooter_band(results),
        "show": False,
    }, results))


def plot_inflation_decomposition(results: UStarResults) -> None:
    """Stack the Phillips curve's terms against observed inflation.

    The bars sum to observed inflation exactly, so the chart is the equation
    rather than an approximation of it. Annualised for display, because the
    model fits quarterly rates and nobody reads those.

    `Demand` is the only bar that contains u*, which is the point: it shows how
    much of Australian inflation this model is willing to attribute to the
    labour market, against expectations and supply. On the current calibration
    that share is small, and the chart is the honest way to say so.
    """
    # `annualize` takes an array, a Series, a frame or a scalar and returns the
    # same shape, which its signature can only express as a union. A frame went
    # in, so a frame comes out; checked rather than asserted, since everything
    # below indexes columns.
    decomp = annualize(results.inflation_decomposition())
    if not isinstance(decomp, pd.DataFrame):
        raise TypeError(f"expected the annualised decomposition to be a frame, got {type(decomp).__name__}")

    bars = pd.DataFrame({
        "Inflation target": decomp["anchor"],
        "Expectations above target": decomp["excess"],
        "Demand": decomp["demand"],
        "Supply": decomp["supply"],
        "Noise": decomp["residual"],
    })

    ax = mg.bar_plot(
        bars,
        stacked=True,
        color=["#cccccc", "mediumpurple", "orange", "darkblue", "lightblue"],
        annotate=False,
        width=1.0,
    )
    observed = decomp["observed"].rename("Observed inflation (quarterly annualised)")
    mg.line_plot(observed, ax=ax, color=["indigo"], width=1.5, annotate=False, zorder=10)
    ax.text(
        0.5, 0.02, _PHILLIPS_EQUATION, transform=ax.transAxes, fontsize=9,
        va="bottom", ha="center", usetex=True,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "grey"},
    )
    mg.finalise_plot(
        ax,
        title="Price inflation decomposition",
        ylabel="% p.a.",
        axhline={"y": 2.5, "color": "darkred", "linestyle": "--", "linewidth": 1, "label": "2.5% target"},
        legend={"loc": "best", "fontsize": "x-small"},
        y0=True,
        lheader="pi = target + expectations above target + demand + supply + noise",
        rfooter=_rfooter(results),
        lfooter=_lfooter(results),
        show=False,
    )


def plot_implied_ustar(results: UStarResults) -> None:
    """Show how much of the reported u* path is prior rather than likelihood.

    The grey line is the u* each quarter's inflation would imply on its own,
    from `results.implied_ustar()`. The state law's whole job is to turn that
    into something a NAIRU could plausibly be, so the question the chart
    answers is whether it does that by filtering the series or by ignoring it.

    Sharper here than in the joint model, because this model imposes
    `sigma_ustar` rather than estimating it, and `print_diagnostics` already
    reports that u* moves at the imposed value. The band is the posterior's,
    unwidened: widening it would blur exactly the comparison being made.
    """
    if not results.has_phillips:
        return

    ustar = results.ustar_posterior()
    implied = results.implied_ustar()
    fitted = ustar.median(axis=1)
    ratio = implied.diff().std() / fitted.diff().std()

    ax = mg.fill_between_plot(_band(ustar, widen=1.0), **_BAND_KWARGS)
    mg.line_plot(
        pd.DataFrame({"Implied by inflation alone": implied, "u*": fitted}),
        ax=ax,
        color=["grey", "darkorange"],
        width=[1.0, 2.5],
        alpha=[0.7, 1.0],
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(ax, **_with_excluded({
        "title": "What inflation alone says u* is, quarter by quarter",
        "ylabel": "Per cent",
        "legend": {"loc": "best", "fontsize": "small"},
        "lheader": (
            f"Implied series moves {ratio:.0f}x as much quarter to quarter; "
            f"correlation with u* {implied.corr(fitted):.2f}"
        ),
        "lfooter": _lfooter(results) + "Phillips inverted at posterior medians. ",
        "rfooter": _rfooter(results),
        "show": False,
    }, results))


def plot_ustar_components(results: UStarResults) -> None:
    """Show how much of u*'s path is the specification and how much is the data.

    The dashed line is where u* would have gone from the same 1993 starting
    point with every innovation set to zero, so it is the decay mechanism
    alone. The shaded distance between the two is the whole of what the data
    added. It is the honest answer to "is that narrow early credible band
    telling me the data placed u* at 10.8 in 1993": through the 1990s the two
    lines are nearly on top of each other, so they are not.
    """
    if not results.decays:
        return
    d = results.ustar_change_decomposition()

    ax = mg.fill_between_plot(
        pd.DataFrame({"lower": d[["ustar", "deterministic"]].min(axis=1),
                      "upper": d[["ustar", "deterministic"]].max(axis=1)}),
        color="cornflowerblue", alpha=0.25, label="Contribution of the data",
    )
    mg.line_plot(
        pd.DataFrame({
            "u*": d["ustar"],
            "Convergence alone, no innovations": d["deterministic"],
        }),
        ax=ax, color=["darkorange", "black"], width=[2, 1.5], style=["--", "-"],
        annotate=True, rounding=2,
    )
    total = d["ustar"].iloc[-1] - d["ustar"].iloc[0]
    det = d["deterministic"].iloc[-1] - d["deterministic"].iloc[0]
    mg.finalise_plot(ax, **_with_excluded({
        "title": "What moves u*: the specification or the data",
        "ylabel": "Per cent",
        "legend": {"loc": "best", "fontsize": "small"},
        "lheader": f"Of u*'s total fall of {abs(total):.2f}pp, "
                   f"{abs(det):.2f}pp is the decay mechanism alone",
        "rfooter": _rfooter(results),
        "lfooter": _lfooter(results),
        "show": False,
    }, results))



# The priors `estimate.py` sets, named here so the prior-posterior charts can
# draw them. Kept beside the charts rather than exported from `estimate`,
# because a chart needs (kind, mu, sd) and the model needs PyMC settings, and
# tying them together would make one serve the other badly.
_PRIORS: dict[str, tuple[str, float, float]] = {
    "sigma_okun": ("half", 0.0, 1.0),
    "gamma_pi": ("normal", -1.5, 1.0),
    "beta_pi": ("normal", 0.5, 0.3),
    "rho_pi": ("normal", 0.0, 0.1),
    "xi_gscpi": ("normal", 0.0, 0.1),
    "epsilon_pi": ("half", 0.0, 0.25),
}


def _prior_for(results: UStarResults, name: str) -> tuple[str, float, float] | None:
    """Return (kind, mu, sd) for a parameter's prior, or None if unknown.

    Two depend on run settings rather than being fixed in the source, so they
    are read from the constants the run recorded.
    """
    if name == "beta_okun":
        two_sided = bool(results.constants.get("two_sided_beta", True))
        return ("normal", 0.5, 0.5) if two_sided else ("half", 0.5, 0.5)
    if name == "sigma_ustar":
        prior = results.constants.get("sigma_ustar_prior")
        if prior is None:
            return None
        mu, sd = float(prior[0]), float(prior[1])
        return ("normal", mu, sd)
    return _PRIORS.get(name)


def plot_prior_posterior(results: UStarResults) -> int:
    """Draw one chart per estimated scalar, posterior against prior.

    The printed diagnostics report a mean and an interval, which cannot show
    a parameter whose chains peak in different places or one that has not
    moved off its prior. Both have mattered here.
    """
    return prior_posterior.plot_all(
        results.posterior,
        lambda name: _prior_for(results, name),
        footers={"lfooter": _lfooter(results), "rfooter": _rfooter(results)},
    )


def run_analysis(
    output_dir: Path | str | None = None,
    prefix: str = "ustar",
    chart_dir: Path | str | None = None,
) -> UStarResults:
    """Load a saved run, print the diagnostics, write the charts."""
    results = load_results(output_dir=output_dir, prefix=prefix)

    # Shade the quarters where u* is placed by the state law and by Okun rather
    # than by inflation. Measured on this model, not inherited: u* opens at
    # 10.75 against u of 10.93, the 90% band runs 2.79x its mid-sample width in
    # 1993 and does not settle until 1998, and Okun outweighs the Phillips curve
    # 3.2:1 per point of u* even with sigma_okun free at 0.485. Guarded on the
    # sample actually starting there, so a run over a different span is not
    # given a window that was never checked for it. See MODEL_NOTES.
    # The footers name the structure imposed on u*, so they are attached per run.
    lfooter = _MODEL + ustar_structure_note(results.constants)
    chart_annotations.attach(
        results,
        lfooter=lfooter,
        lfooter_band=lfooter + _BAND_NOTE,
        unidentified_window=(
            UNIDENTIFIED_WINDOW
            if results.obs_index[0] == pd.Period(UNIDENTIFIED_WINDOW[0], freq="Q")
            else None
        ),
    )

    print_diagnostics(results)

    # A variant run gets its own directory rather than overwriting the default
    # one. Charting clears its directory first, so without this a single
    # `--prefix ustar_conv` run would silently delete the headline charts, and
    # the person comparing them would not know why.
    if chart_dir is None:
        chart_dir = CHART_DIR if prefix == "ustar" else DEFAULT_CHART_BASE / f"UStar-{prefix}"
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()
    save_diagnostics(results.trace, chart_dir, prefix, model="ustar")

    plot_ustar(results)
    plot_ustar(results, shade_inflation=True, tag="inflation")
    plot_ugap(results)
    plot_ustar_components(results)
    print(f"Prior-posterior charts: {plot_prior_posterior(results)}")
    if results.has_phillips:
        plot_inflation_decomposition(results)
        plot_implied_ustar(results)

    print(f"\nCharts written to: {chart_dir if chart_dir is not None else CHART_DIR}")
    return results
