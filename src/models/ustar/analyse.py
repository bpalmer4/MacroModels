"""Charts and printed diagnostics for the ustar model.

One object is being estimated, so there are few charts: u* against the
unemployment rate, and the gap between them. The printed diagnostics carry the
weight, because the question that matters is not what u* is but whether the
data placed it there.
"""

from pathlib import Path  # noqa: TC003 — used at runtime in function signatures
from typing import Any

import mgplot as mg
import pandas as pd

from src.data.inflation import get_trimmed_mean_annual
from src.models.ustar.results import DEFAULT_CHART_BASE, UStarResults, load_results
from src.models.ystar.analyse import excluded_span_style
from src.utilities.rate_conversion import annualize

CHART_DIR = DEFAULT_CHART_BASE / "UStar"

# Used only for runs saved before `build_observations` began recording where its
# series came from. A current run carries its own records and `_rfooter` reads
# those instead, which is also how a chart drawn from the joint y*/u* run names
# that model's sources rather than this one's. The GSCPI is a Phillips curve
# input here and was missing from this line entirely.
_RFOOTER = "Built using: ABS 1364.0.15.003, 5206.0, 6401.0, 6457.0; NY Fed"
_LFOOTER = "Australia. u* model. "
# Only for charts that actually draw a band. The decomposition chart is bars
# and a line built from median parameters, with no interval on it to widen.
_LFOOTER_BAND = _LFOOTER + "Band widened x2 for the imposed drift; see notes. "

# Quarters that carried no likelihood, as ("2020Q2", "2021Q3"), or None.
#
# `ustar` never excludes anything, so this is None for its own runs. It exists
# because the joint y*/u* model reuses these plotting functions and *does*
# exclude the pandemic quarters from all three of its equations, which makes u*
# there a prior extrapolation rather than an estimate. Set by the caller before
# plotting, the same pattern `ystar.analyse` uses for the same reason. Without
# it the u* chart draws a confident line through six quarters nothing was
# fitted to.
_EXCLUDED_WINDOW: tuple[str, str] | None = None

# The early quarters where u* is placed by the state law rather than by
# inflation, shaded so the chart does not read as a confident estimate there.
# Set by whichever model has the evidence for a window; None draws nothing.
_UNIDENTIFIED_WINDOW: tuple[str, str] | None = None

# The window this model's own diagnostics support, applied in `run_analysis`.
# The joint model sets its own, to the same dates and on its own evidence.
#
# Ends 1995Q4, where the band criterion points rather than where the Phillips
# residuals and the expectations date do. The 90% band runs 2.79x its
# mid-sample width in 1993 and 1.96x in 1994, is 1.53x by 1995 and 1.36x by
# 1996, which is close to the 1.1-1.3 it holds until 2002. Shading to 1998
# would assert that 1997 is as doubtful as 1993, and it is not. That u* is not
# fully settled until 1998 is left to MODEL_NOTES, which can say it in degrees.
UNIDENTIFIED_WINDOW = ("1993Q1", "1995Q4")

# Orange rather than the excluded window's yellow, so the two are told apart at
# a glance where both appear. Low alpha: it sits under the u* line, which is
# darkorange itself, and must not compete with it.
_UNIDENTIFIED_SPAN: dict[str, Any] = {
    "color": "darkorange",
    "alpha": 0.12,
    "label": "u* not well identified",
}


def _unidentified_span() -> list[dict[str, Any]]:
    """Return an axvspan dict for the weakly identified early window, or nothing.

    It carries its own legend label and no footer note, for the reason
    `_excluded_span` gives: a footer would be a quieter second statement of what
    the legend already says.
    """
    if _UNIDENTIFIED_WINDOW is None:
        return []
    lo, hi = _UNIDENTIFIED_WINDOW
    return [{
        "xmin": pd.Period(lo, freq="Q"),
        "xmax": pd.Period(hi, freq="Q"),
        **_UNIDENTIFIED_SPAN,
        "label": f"{_UNIDENTIFIED_SPAN['label']}, {lo}-{hi}",
    }]


def _excluded_span() -> list[dict[str, Any]]:
    """Return an axvspan dict marking the unfitted window, or nothing.

    Styled from `ystar.analyse.excluded_span_style` rather than restyled here,
    so the pandemic window looks identical on every chart in the package.
    Copying the styling into each package is how it would drift.

    It carries its own legend label, which is why no footer note is added: a
    footer would be a second, quieter statement of the same thing, and on the
    inflation-shaded chart the reader has to tell this span apart from the
    band-breach spans by looking at it.
    """
    if _EXCLUDED_WINDOW is None:
        return []
    lo, hi = _EXCLUDED_WINDOW
    style = excluded_span_style()
    return [{
        "xmin": pd.Period(lo, freq="Q"),
        "xmax": pd.Period(hi, freq="Q"),
        **style,
        "label": f"{style['label']}, {lo}-{hi}",
    }]


def _with_excluded(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Add the unfitted-window and weakly-identified markers to finalise kwargs."""
    spans = _unidentified_span() + _excluded_span()
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
# to stand in for the uncertainty the model cannot express: sigma_ustar is
# imposed, so the posterior answers "where is u* given that it drifts at
# exactly this rate" and carries no uncertainty about the rate itself.
#
# Two is not a round number picked for convenience. The conditional band is
# 0.48pp wide at the endpoint and, measured across sigma_ustar from 0.024 to
# 0.05, u* itself moves 0.39pp; the union of the conditional bands over that
# range runs 4.44 to 5.33, about 0.89pp, which is what doubling reproduces.
# The band width is also near-invariant to the setting (0.48 to 0.50 across the
# whole range), so scaling rather than re-deriving is defensible.
#
# It is an approximation to a sweep, not a posterior, and every chart drawn
# with it says so in its left footer.
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
    in for uncertainty about the imposed drift; see `_BAND_WIDEN`.
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


def _inflation_regime_spans(index: pd.PeriodIndex) -> list[dict[str, Any]]:
    """Return axvspan dicts shading quarters where the trimmed mean left the band.

    Red above, blue below, nothing inside 2-3%. Two things the label has to be
    careful about. The series is the annual **trimmed mean**, not headline CPI,
    while the RBA's 2-3% band is a headline-CPI target, so this marks where the
    core measure sat outside the band rather than where the target was missed.
    And it is the four-quarter rate rather than the model's quarterly series,
    because "outside the band" is a four-quarter notion and the
    quarterly-annualised rate crosses the thresholds several times a year.

    Contiguous quarters are merged into single spans, so the chart gets a few
    readable blocks instead of 134 abutting rectangles with seamed edges.
    """
    inflation = get_trimmed_mean_annual().data.astype(float).reindex(index)

    spans: list[dict[str, Any]] = []
    run_state: str | None = None
    run_start: pd.Period | None = None

    def close(end: pd.Period) -> None:
        if run_state is None or run_start is None:
            return
        color = "tab:red" if run_state == "high" else "tab:blue"
        spans.append({
            "xmin": run_start, "xmax": end, "color": color, "alpha": 0.10, "zorder": 0,
        })

    # Paired with `index` rather than read off `inflation.items()`: the series
    # was just reindexed onto it, so the quarters are the same ones, and this
    # way each is a Period rather than the Hashable a Series yields.
    for period, value in zip(index, inflation.to_numpy(), strict=True):
        if pd.isna(value):
            state = None
        elif value > _INFLATION_HIGH:
            state = "high"
        elif value < _INFLATION_LOW:
            state = "low"
        else:
            state = None
        if state != run_state:
            close(period)
            run_state, run_start = state, period
    close(index[-1])

    return spans


def _rfooter(results: UStarResults) -> str:
    """Return the source line this run recorded, falling back for older runs."""
    return results.source_footer or _RFOOTER


def plot_ustar(results: UStarResults, shade_inflation: bool = False, tag: str = "") -> None:
    """u* against the observed unemployment rate, with a credible band.

    With `shade_inflation`, the background marks the quarters where annual
    inflation sat outside the RBA's 2-3% band, so u* can be read against the
    episodes when inflation actually left the central tendency.

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
        "lfooter": _LFOOTER_BAND,
        "show": False,
    }
    if tag:
        finalise_kwargs["tag"] = tag
    if shade_inflation:
        finalise_kwargs["axvspan"] = _inflation_regime_spans(results.obs_index)
        finalise_kwargs["lheader"] = (
            f"Shaded where the annual trimmed mean sat outside "
            f"{_INFLATION_LOW:g}-{_INFLATION_HIGH:g}%: red above, blue below"
        )
    mg.finalise_plot(ax, **_with_excluded(finalise_kwargs))


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
        "lfooter": _LFOOTER_BAND,
        "show": False,
    }))


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
        lfooter=_LFOOTER,
        show=False,
    )


def plot_ustar_components(results: UStarResults) -> None:
    """Show how much of u*'s path is the specification and how much is the data.

    The dashed line is where u* would have gone from the same 1993 starting
    point with every innovation set to zero, so it is the convergence mechanism
    alone. The shaded distance between the two is the whole of what the data
    added. It is the honest answer to "is that narrow early credible band
    telling me the data placed u* at 10.8 in 1993": through the 1990s the two
    lines are nearly on top of each other, so they are not.
    """
    if not results.converges:
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
                   f"{abs(det):.2f}pp is the convergence mechanism alone",
        "rfooter": _rfooter(results),
        "lfooter": _LFOOTER,
        "show": False,
    }))


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
    global _UNIDENTIFIED_WINDOW  # noqa: PLW0603 — the module-level marker these charts read
    _UNIDENTIFIED_WINDOW = (
        UNIDENTIFIED_WINDOW
        if results.obs_index[0] == pd.Period(UNIDENTIFIED_WINDOW[0], freq="Q")
        else None
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

    plot_ustar(results)
    plot_ustar(results, shade_inflation=True, tag="inflation")
    plot_ugap(results)
    plot_ustar_components(results)
    if results.has_phillips:
        plot_inflation_decomposition(results)

    print(f"\nCharts written to: {chart_dir if chart_dir is not None else CHART_DIR}")
    return results
