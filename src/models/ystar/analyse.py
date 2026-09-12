"""Diagnostics and charts for the ystar model."""

from typing import TYPE_CHECKING, Any, Unpack

import mgplot as mg
import numpy as np
import pandas as pd
from mgplot.finalisers import DataT, LPFKwargs

from src.data.henderson import hma
from src.models.ystar.decompose import (
    GrowthDecomposition,
    decompose_potential_growth,
    print_decomposition,
)
from src.models.ystar.results import (
    DEFAULT_CHART_BASE,
    PotentialResults,
    load_results,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from matplotlib.axes import Axes

CHART_DIR = DEFAULT_CHART_BASE / "YStar"

# Each specification writes to its own directory. `run_analysis` clears the
# directory before writing, so sharing one would mean whichever spec ran last
# silently deleted the other's charts.
SPEC_CHART_DIRS = {"production": DEFAULT_CHART_BASE / "YStar-production"}

# Used only for runs saved before `build_observations` began recording where its
# series came from. A current run carries its own records and `_rfooter` reads
# those instead, which is also how a chart drawn from the joint y*/u* run names
# that model's sources rather than this one's.
_RFOOTER = "Built using: ABS 5206.0, 6202.0, 6401.0"
_RFOOTER_CORE = "Built using: ABS 5206.0, 6401.0"
# The production spec adds the capital stock and the capital share. The stock
# comes from the Modellers' Database, not 5204.0, which this constant used to
# claim: see `src/data/capital.py`.
_RFOOTER_PRODUCTION = "Built using: ABS 1364.0.15.003, 5206.0, 6202.0, 6401.0"

# Shown on the inflation-defined gap chart. Not "a positive output gap is
# consistent with inflation": that series is a positive multiple of the
# inflation deviation, so it cannot be evidence for the claim. What the chart
# shows is the definition being applied, and the sign is the thing to read.
_GAP_HEADER = "The gap is defined by inflation's deviation from the target"
# Shown on the actual gap chart, where the point is what the definition leaves out.
_ACTUAL_GAP_HEADER = "GDP's full deviation from potential: the defined gap plus the residual"
_LFOOTER = "Australia. y* model. "
# Points per quarter used when shading the growth-versus-potential chart.
_FILL_SUBDIVISIONS = 20

# Quarters excluded from the output gap composition chart's vertical scale.
# The 2020 lockdown deviations are around -7 and -5 against a range of roughly
# +/-2 for every other quarter in the sample, so leaving them in the scale
# hides the target-period story the chart exists to show. 2021Q3 (the Delta
# lockdown, -2.1) is deliberately not here: it fits.
_OFF_SCALE = ("2020Q2", "2020Q3")

_BAND_KWARGS: dict[str, Any] = {
    "color": "cornflowerblue",
    "alpha": 0.25,
    "label": "90% credible interval",
}


# Set once per run by `run_analysis` and read by the two finalise wrappers
# below. Module-level rather than a parameter because roughly twenty chart
# functions call finalise, several of them from a `GrowthDecomposition` that has
# no access to the run's settings, and threading a window through every one of
# those signatures would be a worse trade than one piece of run-scoped state.
_EXCLUDED_WINDOW: tuple[str, str] | None = None

# Deliberately plain: the shading marks quarters that carry no likelihood, so it
# should read as an absence rather than as a highlighted episode. Behind the
# lines and the credible-interval band.
#
# `label` puts it in the legend, which is where a reader looks to find out what
# a shaded band means. "excluded from fit" rather than "pandemic" alone: the
# claim being made is about the estimation, not about the epidemiology, and a
# reader who sees only "pandemic" will take the shading for an episode marker.
# Yellow rather than orange: several of these charts draw their headline series
# in darkorange, and an orange wash behind an orange line costs contrast where
# it is needed most. Gold at low alpha reads as a warm highlight against both
# the orange lines and the cornflower credible-interval band.
_EXCLUDED_SPAN: dict[str, Any] = {
    "color": "gold",
    "alpha": 0.20,
    "zorder": -1,
    "label": "Pandemic: excluded from fit",
}


def excluded_span_style() -> dict[str, Any]:
    """Return the shared styling for the excluded-window span.

    Public because `ustar` and the joint y*/u* model draw the same window on
    their own charts, and the whole point is that it looks identical wherever
    it appears. Copying the dict into each package is how it would drift.
    """
    return dict(_EXCLUDED_SPAN)


def _excluded_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Add the excluded-window shading and its footer note to finalise kwargs.

    The states run through an excluded window under their priors, so every
    series still has values there and the charts would otherwise show a fitted
    trend across quarters the model was never shown. Shading says so on the
    chart rather than in a caption someone will read separately.
    """
    if _EXCLUDED_WINDOW is None:
        return dict(kwargs)

    lo, hi = _EXCLUDED_WINDOW
    kwargs = dict(kwargs)
    if "axvspan" in kwargs:
        raise ValueError("axvspan is set by the excluded-window shading; do not pass it too")
    kwargs["axvspan"] = {
        "xmin": pd.Period(lo, freq="Q"),
        "xmax": pd.Period(hi, freq="Q"),
        **_EXCLUDED_SPAN,
    }
    # The dates go in the legend label, not the left footer. Some of these
    # footers are already long (the gap chart names its off-scale quarters), and
    # appending to them overran the source line on the right.
    kwargs["axvspan"]["label"] = f"{_EXCLUDED_SPAN['label']}, {lo}-{hi}"
    return kwargs


def _on_quarterly_axis(axes: Axes) -> bool:
    """Whether these axes are plotted against quarters, so a span belongs on them.

    mgplot draws a PeriodIndex at its period ordinals, so a quarterly chart's
    xlim brackets the ordinal of every quarter it covers (2020Q2 is 201, against
    an xlim of roughly 85 to 232 on the full sample). Charts with any other
    x-axis are nowhere near those numbers: `plot_growth_contributions` draws
    period *blocks* on a RangeIndex with an xlim of -0.59 to 3.59, and shading it
    with a quarter stretched the axis to take in a coordinate 200 units away,
    squashing every bar against the left edge.

    Checked rather than made a caller's flag, since a flag is something the next
    chart added here would have to remember.
    """
    if _EXCLUDED_WINDOW is None:
        return False
    lo, hi = _EXCLUDED_WINDOW
    left, right = axes.get_xlim()
    return left <= pd.Period(hi, freq="Q").ordinal and pd.Period(lo, freq="Q").ordinal <= right


def _finalise(axes: Axes, **kwargs: Unpack[mg.FinaliseKwargs]) -> None:
    """`mg.finalise_plot` with any excluded window shaded."""
    if not _on_quarterly_axis(axes):
        mg.finalise_plot(axes, **kwargs)
        return
    mg.finalise_plot(axes, **_excluded_kwargs(kwargs))


def _line_plot_finalise(
    data: DataT,
    # LPFKwargs, not LineKwargs: the *_finalise entry points take the plot
    # kwargs and the finalise kwargs together, and `mg.LineKwargs` is only the
    # first half.
    **kwargs: Unpack[LPFKwargs],
) -> None:
    """`mg.line_plot_finalise` with any excluded window shaded.

    These entry points build their own axes, so the quarterly test is made on
    the data instead. Same reason as `_on_quarterly_axis`.
    """
    if not isinstance(data.index, pd.PeriodIndex):
        mg.line_plot_finalise(data, **kwargs)
        return
    mg.line_plot_finalise(data, **_excluded_kwargs(kwargs))


def _excluded_window(results: PotentialResults) -> tuple[str, str] | None:
    """Return the run's excluded window, read from its own recorded settings."""
    window = results.constants.get("exclude_window")
    return window if isinstance(window, tuple) else None


def _fitted_mask(results: PotentialResults) -> pd.Series:
    """Boolean over the sample: True where the run carried a likelihood term.

    Read from `results` rather than the module-level `_EXCLUDED_WINDOW`, since
    `print_diagnostics` can be called on a results object directly without
    going through `run_analysis`, and a statistic must not depend on whether
    some earlier call happened to set that global.
    """
    index = results.obs_index
    window = _excluded_window(results)
    if window is None:
        return pd.Series(data=True, index=index)

    lo, hi = window
    excluded = (index >= pd.Period(lo, freq="Q")) & (index <= pd.Period(hi, freq="Q"))
    return pd.Series(data=~np.asarray(excluded), index=index)


def _band(posterior: pd.DataFrame) -> pd.DataFrame:
    return PotentialResults.band(posterior)


def _rfooter(results: PotentialResults) -> str:
    """Source line naming only the catalogues this specification actually uses.

    Read from the run's own records where it has them, so the line describes
    what was loaded rather than what this module believes about a spec name.
    """
    recorded = results.source_footer
    if recorded is not None:
        return recorded
    return _spec_rfooter(results)


def _decomposition_rfooter(decomposition: GrowthDecomposition) -> str:
    """Source line for the growth-accounting charts.

    The decomposition loads hours, population and participation itself, so it
    names 6202.0 whether or not the run that produced `y*` did.
    """
    return decomposition.sources.footer() or _RFOOTER


def _spec_rfooter(results: PotentialResults) -> str:
    """Return the pre-recording fallback: what each specification used to load."""
    if results.spec == "labour":
        return _RFOOTER
    if results.spec == "production":
        return _RFOOTER_PRODUCTION
    return _RFOOTER_CORE


def print_diagnostics(results: PotentialResults) -> None:
    """Print the parameter summary and the model's headline numbers."""
    print("\nPosterior summary")
    print("-" * 70)
    print(results.summary().to_string())

    pot = results.potential_growth_posterior()
    gap = results.output_gap_posterior()
    last = pot.index[-1]

    gap_label = "Inflation-defined gap" if results.spec in ("inflation", "production") else "Output gap"
    headline = [("Potential growth", pot), (gap_label, gap)]
    if results.spec in ("inflation", "production", "core", "target"):
        headline.insert(0, ("Trend growth (g state)", results.trend_growth_posterior()))
    else:
        headline.insert(0, ("Trend productivity growth", results.trend_prod_growth_posterior()))

    print(f"\nHeadline estimates ({last})")
    print("-" * 70)
    for label, posterior in headline:
        row = posterior.loc[last]
        print(
            f"  {label:<28} {row.median():6.2f}"
            f"  [{row.quantile(0.05):5.2f}, {row.quantile(0.95):5.2f}]",
        )

    posterior_group = results.trace.get("posterior")
    if posterior_group is not None and "phi_1" in posterior_group:
        roots = results.ar_root_posterior()
        print("\nCycle stationarity (largest AR root modulus)")
        print("-" * 70)
        print(
            f"  {'|root|':<28} {np.median(roots):6.2f}"
            f"  [{np.quantile(roots, 0.05):5.2f}, {np.quantile(roots, 0.95):5.2f}]",
        )
        print(f"  {'draws stationary':<28} {(roots < 1.0).mean():6.1%}")
    elif results.spec in ("inflation", "production"):
        # The gap is defined by inflation, so there is no cycle to be
        # stationary. What matters instead is how much of output's deviation
        # from potential that definition actually accounts for.
        gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)
        # Not `gap`: that name is already bound to the time x draw DataFrame above.
        gap_median = results.output_gap_median()
        deviation = gdp - results.potential_median()

        # Over the fitted quarters only. Computed on the full index, the six
        # excluded quarters put deviations of -8.5 and -5.6 into the
        # denominator, which is a lockdown rather than a cycle: sd of GDP less
        # potential goes 0.51 -> 1.08 and the share reads 3.0% instead of 13.1%.
        # The share is meant to say how much of the *cycle* the definition
        # accounts for, so it can only be taken over quarters the model was
        # asked to explain.
        fitted = _fitted_mask(results)
        gap_median, deviation = gap_median[fitted], deviation[fitted]

        scope = "" if _excluded_window(results) is None else "  (fitted quarters only)"
        print(f"\nHow much of the cycle the inflation-defined gap explains{scope}")
        print("-" * 70)
        print(f"  {'sd of gap':<28} {gap_median.std():6.2f}")
        print(f"  {'sd of GDP less potential':<28} {deviation.std():6.2f}")
        print(f"  {'variance share':<28} {gap_median.var() / deviation.var():6.1%}")
    else:
        # cycle_ar off: the gap is the bare identity, so there is no AR root.
        print("\nNo AR(2) cycle restriction in this run: gap = log_gdp - y*.")

    constants = results.constants
    if constants:
        print("\nImposed (not estimated)")
        print("-" * 70)
        for key, value in constants.items():
            print(f"  {key:<28} {value}")


def plot_potential(
    results: PotentialResults,
    plot_from: str | None = None,
    tag: str = "full",
) -> None:
    """Actual vs potential output.

    Drawn on two windows. Over the full sample the two lines are nearly
    indistinguishable, because a gap of a per cent or two is invisible against
    thirty years of cumulative growth; the recent window is where the level
    difference can actually be read.
    """
    potential = results.potential_posterior()
    log_gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)

    if plot_from:
        start = pd.Period(plot_from, "Q")
        potential = potential.loc[potential.index >= start]
        log_gdp = log_gdp.loc[log_gdp.index >= start]

    ax = mg.fill_between_plot(_band(potential), **_BAND_KWARGS)
    mg.line_plot(
        pd.DataFrame({"Actual GDP": log_gdp, "Potential output": potential.median(axis=1)}),
        ax=ax,
        color=["black", "darkorange"],
        width=[1.5, 2],
        style=["-", "--"],
        annotate=True,
        rounding=1,
    )
    _finalise(
        ax,
        tag=tag,
        title="GDP and potential output",
        ylabel="log level x 100",
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def plot_actual_output_gap(
    results: PotentialResults,
    plot_from: str | None = None,
    tag: str = "full",
) -> None:
    """GDP's actual deviation from potential, log_gdp - y*, with a credible band.

    This is the series "output gap" is normally taken to mean, and it is the
    wider one: under the `inflation` and `production` specifications it is the
    inflation-defined gap plus `e_c`, whose sd is around 0.98 against a defined
    gap that mostly sits inside +/-0.5. The band is correspondingly wider too,
    since potential's own uncertainty is in it.

    Drawn on two windows. The 2020 lockdown quarters run to roughly -7, which
    compresses everything else, so the recent window is where the current
    position can be read.
    """
    gap = results.actual_output_gap_posterior()

    if plot_from:
        gap = gap.loc[gap.index >= pd.Period(plot_from, "Q")]

    band = _band(gap)
    median = gap.median(axis=1)

    # Same treatment the composition chart gives them, and for the same reason:
    # 2020Q2 reaches about -7.5 against a range of roughly +/-2 for every other
    # quarter, so leaving it in the scale flattens the whole chart. The line is
    # still drawn and runs off the top and bottom; the footnote says so.
    off_scale = [q for q in (pd.Period(p, "Q") for p in _OFF_SCALE) if q in gap.index]
    kept = band.drop(index=off_scale)
    ylim = (
        float(np.floor(kept["lower"].min() * 2 - 0.5) / 2),
        float(np.ceil(kept["upper"].max() * 2 + 0.5) / 2),
    )
    # Not `_off_scale_note`: that sentence is written for the composition
    # chart's header, and in a footer beside the source line it overruns.
    note = _off_scale_memo(median, off_scale)

    ax = mg.fill_between_plot(band, **_BAND_KWARGS)
    mg.line_plot(
        median.rename("Output gap"),
        ax=ax,
        color=["darkorange"],
        width=2,
        annotate=True,
        rounding=1,
    )
    _finalise(
        ax,
        tag=tag,
        title="Output gap",
        ylabel="Per cent of potential",
        ylim=ylim,
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=_ACTUAL_GAP_HEADER,
        rfooter=_rfooter(results),
        lfooter=_LFOOTER + (f"{note}. " if note else ""),
        show=False,
    )


def plot_inflation_defined_gap(results: PotentialResults) -> None:
    """Plot the inflation-defined gap, c x (pi - anchor), with a credible band.

    Deliberately not titled "output gap". Under the `inflation` and `production`
    specifications this series carries no GDP data at all: it is the inflation
    deviation rescaled by `c`, and it accounts for about a fifth of GDP's
    deviation from potential. `plot_actual_output_gap` draws the whole
    deviation, which is the wider series a reader expects from that phrase.

    The band is narrow because the only uncertainty in it is uncertainty in `c`;
    the inflation deviation is observed data.
    """
    gap = results.output_gap_posterior()

    ax = mg.fill_between_plot(_band(gap), **_BAND_KWARGS)
    median = gap.median(axis=1)
    mg.line_plot(
        median.rename("Inflation-defined gap"),
        ax=ax,
        color=["darkorange"],
        width=2,
        annotate=True,
        rounding=1,
    )
    _finalise(
        ax,
        title="Inflation-defined output gap",
        ylabel="Per cent of potential",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=_GAP_HEADER,
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def plot_gap_composition(results: PotentialResults) -> None:
    """Split GDP's deviation from potential into the explained gap and the noise.

    The `inflation` specification writes output as

        log_gdp_t = y*_t + gap_t + e_c,   gap_t = c · (pi_t - anchor)

    so the deviation of GDP from potential is exactly the inflation-defined gap
    plus the residual, with nothing else in it, since `e_c` is white noise by
    construction. Stacked bars make that additivity visible; the line is the
    total they sum to. It holds for `production` as well: that spec changes
    where potential's growth comes from, not the GDP observation equation.

    The residual bar is the arithmetic residual of the two plotted medians
    (median deviation less median gap) rather than the median of the residual
    posterior, so the bars sum to the line exactly. The two differ by at most
    0.01 percentage points on the current trace, medians not being additive.
    """
    if results.spec not in ("inflation", "production"):
        raise ValueError(
            f"gap composition is only meaningful for the 'inflation' and 'production' specifications, "
            f"not {results.spec!r}: elsewhere the gap is the identity log_gdp - y*, so the residual is "
            "identically zero",
        )

    gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)
    deviation = results.potential_posterior().rsub(gdp, axis=0).median(axis=1)
    gap = results.output_gap_median()

    components = pd.DataFrame({
        "Explained gap": gap,
        "Noise": deviation - gap,
    })

    # The 2020 lockdown quarters are an order of magnitude larger than anything
    # else and would compress the rest of the sample into a fifth of the chart.
    # They are excluded from the scale rather than from the data: the bars are
    # still drawn, they run off the top and bottom, and the footnote says so.
    off_scale = [q for q in (pd.Period(p, "Q") for p in _OFF_SCALE) if q in components.index]
    ax = _plot_gap_composition_axes(components, deviation)
    _finalise(
        ax,
        title="Output gap composition",
        ylabel="Per cent of potential",
        ylim=_scale_excluding(components, deviation, off_scale),
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER + "Bars sum to the line by construction. ",
        lheader=_off_scale_note(deviation, off_scale),
        show=False,
    )


def _off_scale_values(deviation: pd.Series, off_scale: list[pd.Period]) -> list[str]:
    """Return "quarter value" strings for the quarters left off the scale.

    Positional lookup: the pandas stubs do not accept a Period as a `.loc` key,
    and `get_loc` returns a slice or mask when the index has duplicates, so the
    integer case is narrowed explicitly rather than assumed.
    """
    values = []
    for q in off_scale:
        position = deviation.index.get_loc(q)
        if not isinstance(position, int):
            continue
        values.append(f"{q} {float(deviation.iloc[position]):.1f}")
    return values


def _off_scale_memo(deviation: pd.Series, off_scale: list[pd.Period]) -> str:
    """Name the off-scale quarters compactly, for use in a crowded footer."""
    values = _off_scale_values(deviation, off_scale)
    return f"Lockdown quarters off scale ({', '.join(values)})" if values else ""


def _off_scale_note(deviation: pd.Series, off_scale: list[pd.Period]) -> str:
    """Name the quarters the vertical scale leaves out, with their values."""
    if not off_scale:
        return ""
    values = _off_scale_values(deviation, off_scale)
    if not values:
        return ""
    quarters = ", ".join(values)
    return f"Scaled to exclude the lockdown quarters ({quarters}), which run off the chart"


def _scale_excluding(
    components: pd.DataFrame,
    deviation: pd.Series,
    off_scale: list[pd.Period],
) -> tuple[float, float]:
    """Return y limits covering the bars and the line, ignoring `off_scale`.

    A stacked bar reaches the sum of its positive parts above zero and the sum
    of its negative parts below, which is wider than the total the line shows
    whenever the two components have opposite signs. Both are measured.
    """
    kept = components.drop(index=off_scale)
    top = max(kept.clip(lower=0).sum(axis=1).max(), deviation.drop(index=off_scale).max())
    bottom = min(kept.clip(upper=0).sum(axis=1).min(), deviation.drop(index=off_scale).min())
    # Round outward to a half point so the ticks land on round numbers.
    return float(np.floor(bottom * 2 - 0.5) / 2), float(np.ceil(top * 2 + 0.5) / 2)


def _plot_gap_composition_axes(components: pd.DataFrame, deviation: pd.Series) -> Axes:
    """Draw the stacked component bars with the total overlaid as a line."""
    ax = mg.bar_plot(
        components,
        stacked=True,
        color=["darkorange", "slategrey"],
        annotate=False,
        width=1.0,
    )
    mg.line_plot(
        deviation.rename("GDP less potential"),
        ax=ax,
        color=["black"],
        width=1.5,
        annotate=True,
        rounding=1,
    )
    return ax


def plot_growth_vs_potential(
    results: PotentialResults,
    plot_from: str | None = None,
    tag: str = "",
) -> None:
    """Actual growth against potential, shaded by which side it is on.

    The shading is the point of the chart: inflation responds to the *level*
    of the output gap, but whether actual growth sits above or below potential
    says whether that gap is opening or closing. A positive gap with growth at
    potential means the gap is not closing — which is where Australia is.

    Actual growth is smoothed by Henderson on the log level and then
    differenced, not the reverse: differencing first applies a high-pass filter
    to exactly the noise being removed.
    """
    gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)
    actual = hma(gdp, 7).diff(4)
    potential = results.potential_growth_posterior().median(axis=1)
    # Classify against the gap averaged over the same four quarters the growth
    # differential spans. Against a single-quarter gap the shading flickers
    # wherever the gap crosses zero (2021 crosses three times), which is a
    # mismatch of horizons rather than anything economic.
    gap = results.output_gap_median().rolling(4, min_periods=1).mean()

    data = pd.DataFrame({
        "Actual growth (smoothed)": actual,
        "Potential growth": potential,
        "_gap": gap,
    }).dropna()
    if plot_from:
        data = data.loc[data.index >= pd.Period(plot_from, "Q")]
    gap_values = data.pop("_gap").to_numpy()

    ax = mg.line_plot(
        data,
        color=["black", "darkorange"],
        width=[2, 2],
        style=["-", "--"],
        annotate=True,
        rounding=1,
    )

    # Read the x-coordinates back from the drawn lines, so the fill lands on
    # whatever internal index mgplot mapped the PeriodIndex onto.
    x = ax.get_lines()[0].get_xdata()
    a = data["Actual growth (smoothed)"].to_numpy()
    p = data["Potential growth"].to_numpy()
    # Whether the gap is widening depends on its sign as well as the growth
    # differential: growth below potential closes a positive gap but deepens a
    # negative one. 2020 is the second case — a deeply negative gap widening
    # fast — so shading purely by which side of the line growth sits would
    # colour it as though it were benign. Red therefore means |gap| growing,
    # which can appear on either side of the potential line.
    #
    # Resample onto a dense grid before shading. `interpolate=True` would
    # interpolate each region's edge to where the two *curves* cross, but the
    # mask flips where the *gap* crosses zero — a different point — which draws
    # spurious wedges (2021Q3, where the mask flips while actual is at 6% and
    # potential at 1.8%). On a dense grid the untinted boundary interval is
    # sub-pixel, so no interpolation is needed and none is guessed.
    xd = np.asarray(x, dtype=float)
    xf = np.linspace(xd[0], xd[-1], (len(xd) - 1) * _FILL_SUBDIVISIONS + 1)
    af = np.interp(xf, xd, a)
    pf = np.interp(xf, xd, p)
    gf = np.interp(xf, xd, gap_values)
    widening = (gf >= 0) == (af >= pf)

    ax.fill_between(
        xf, af, pf, where=widening,
        color="firebrick", alpha=0.30, label="Gap widening (away from zero)",
    )
    ax.fill_between(
        xf, af, pf, where=~widening,
        color="steelblue", alpha=0.34, label="Gap narrowing (toward zero)",
    )

    _finalise(
        ax,
        title="Actual growth versus potential",
        ylabel="Year-ended per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER + "Actual smoothed, 7-term Henderson MA. ",
        tag=tag,
        show=False,
    )


def _growth_against_potential(
    results: PotentialResults,
    comparison: pd.Series,
    *,
    label: str,
    colour: str,
    title: str,
    lfooter: str,
    plot_from: str | None = None,
    tag: str = "",
) -> None:
    """Plot a year-ended growth rate against potential growth, with g*'s band.

    Distinct from `plot_growth_vs_potential`, which smooths GDP with a Henderson
    filter before differencing and shades by which side of potential it falls.
    This is the raw year-ended series: noisier, and the noise is the point when
    the question is how far the actual series swings around a trend that barely
    moves.
    """
    potential = results.potential_growth_posterior()
    data = pd.DataFrame({
        label: comparison,
        "Potential growth (g*)": potential.median(axis=1),
    }).dropna()
    band = _band(potential).reindex(data.index)
    if plot_from:
        start = pd.Period(plot_from, "Q")
        data, band = data.loc[data.index >= start], band.loc[band.index >= start]

    ax = mg.fill_between_plot(band, **_BAND_KWARGS)
    mg.line_plot(
        data,
        ax=ax,
        color=[colour, "darkorange"],
        width=[1.5, 2.5],
        style=["-", "--"],
        annotate=True,
        rounding=1,
    )
    _finalise(
        ax,
        title=title,
        ylabel="Year-ended per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER + lfooter,
        tag=tag,
        show=False,
    )


def plot_gdp_growth_against_potential(
    results: PotentialResults,
    plot_from: str | None = None,
    tag: str = "",
) -> None:
    """Year-ended GDP growth against potential growth."""
    log_gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)
    _growth_against_potential(
        results,
        log_gdp.diff(4),
        label="GDP growth (g)",
        colour="black",
        title="GDP growth and potential growth",
        # Kept short: a long left footer collides with the source line on the
        # right. "Year-ended" is on the y-axis already.
        lfooter="Unsmoothed. ",
        plot_from=plot_from,
        tag=tag,
    )


def plot_gov_growth_against_potential(
    results: PotentialResults,
    plot_from: str | None = None,
    tag: str = "",
) -> None:
    """Year-ended government consumption growth against potential growth.

    Government consumption is a component of the GDP whose trend the model is
    estimating, so this is not an independent check on g*. It is a question
    about composition: whether the public component has been running above or
    below the pace the economy's supply side can sustain.
    """
    from src.data.gov_spending import get_gov_consumption_qrtly  # noqa: PLC0415

    gov = get_gov_consumption_qrtly()
    log_gov = np.log(gov.data) * 100
    _growth_against_potential(
        results,
        log_gov.diff(4),
        label="Government consumption growth",
        colour="seagreen",
        title="Government spending growth and potential growth",
        lfooter="Govt final consumption, chain volume. ",
        plot_from=plot_from,
        tag=tag,
    )


def plot_trend_growth(results: PotentialResults) -> None:
    """Trend potential growth — the core specification's headline.

    This is the economy's speed limit: how fast output can grow without
    opening an output gap. One line only, deliberately.

    Which series is the right one depends on the specification, and getting it
    wrong is not a labelling quibble.

    - Where potential is *generated* from the drift (`core`, `target`), the `g`
      state is potential growth. Differencing the `y*` level gives the same
      quantity plus the level shocks `e_y`, so it oscillates around the state
      for no economic reason. Plot the state.
    - Where potential is a *residual* (`inflation`: y* = log_gdp - c·d), `g` is
      only a smooth curve the random walk prior fits alongside potential, and
      potential's actual growth is the differenced level. On this data the two
      diverge sharply (sd 0.72 against 1.57, and -5.95 against +2.02 in 2020Q2).
      Plot the differenced level, and do not quote the state as potential growth.
    """
    # `production` generates potential from the factor trends, so its `g` state
    # *is* potential growth and the differenced level would add the cumulation
    # noise for nothing. It belongs with `core`, not with `inflation`.
    residual_potential = results.spec == "inflation"
    trend = (
        results.potential_growth_posterior()
        if residual_potential
        else results.trend_growth_posterior()
    )

    ax = mg.fill_between_plot(_band(trend), **_BAND_KWARGS)
    mg.line_plot(
        trend.median(axis=1).rename("Potential growth"),
        ax=ax,
        color=["darkorange"],
        width=2,
        annotate=True,
        # Two decimals: the whole point of the chart is a speed limit that has
        # fallen about two points, and 2.1 hides the difference between this
        # model's 2.14, the RBA's ~2.0 and Cobb-Douglas at 1.86.
        rounding=2,
    )
    # The production spec builds this from capital, hours and MFP, so neither
    # the "output and inflation alone" claim nor the two-catalogue source line
    # is true there.
    production = results.spec == "production"
    provenance = (
        "g_Y* = a·g_K* + (1-a)·g_L* + g_M*, with a the observed capital share. "
        if production
        else "Identified from output and inflation alone. "
    )
    _finalise(
        ax,
        title="Potential growth from the production function" if production else "Potential growth",
        ylabel="Year-ended per cent" if residual_potential else "Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER + provenance,
        show=False,
    )


def plot_trend_productivity_growth(results: PotentialResults) -> None:
    """Trend labour productivity growth — the model's headline output."""
    prod = results.trend_prod_growth_posterior()

    ax = mg.fill_between_plot(_band(prod), **_BAND_KWARGS)
    mg.line_plot(
        prod.median(axis=1).rename("Trend productivity growth"),
        ax=ax,
        color=["darkorange"],
        width=2,
        annotate=True,
        rounding=1,
    )
    _finalise(
        ax,
        title="Trend labour productivity growth",
        ylabel="Annualised per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER + "GDP per LFS hour worked. ",
        show=False,
    )


def plot_potential_growth(results: PotentialResults) -> None:
    """Potential growth, split into its trend hours and productivity parts."""
    pot = results.potential_growth_posterior()
    prod = results.trend_prod_growth_level_posterior()
    hours = results.trend_hours_growth_posterior()

    ax = mg.fill_between_plot(_band(pot), **_BAND_KWARGS)
    mg.line_plot(
        pd.DataFrame({
            "Potential growth": pot.median(axis=1),
            "of which: trend hours": hours.median(axis=1),
            "of which: trend productivity": prod.median(axis=1),
        }),
        ax=ax,
        color=["black", "darkorange", "seagreen"],
        width=[2, 1.5, 1.5],
        style=["-", "--", "--"],
        annotate=True,
        rounding=1,
    )
    _finalise(
        ax,
        title="Potential output growth and its components",
        ylabel="Year-ended per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER + "Components add to potential growth exactly. ",
        show=False,
    )


def plot_growth_accounting(decomposition: GrowthDecomposition) -> None:
    """Potential growth split into trend hours and trend productivity.

    A post-modelling accounting split, not a re-estimation: `y*` is exactly the
    model's own path. Trend hours is a Henderson trend of measured labour
    input, so it carries no band; trend productivity is the residual `y* - h*`
    taken draw by draw, so it inherits the whole of the model's uncertainty
    about potential. Showing the band on productivity alone is the honest
    presentation of where the uncertainty actually sits.
    """
    prod = decomposition.productivity_posterior

    ax = mg.fill_between_plot(_band(prod), **_BAND_KWARGS)
    mg.line_plot(
        pd.DataFrame({
            "Potential growth": decomposition.potential_growth,
            "Trend hours": decomposition.hours_growth,
            "Trend productivity": decomposition.productivity_growth,
        }).dropna(),
        ax=ax,
        color=["black", "darkorange", "seagreen"],
        width=[2, 1.5, 1.5],
        style=["-", "--", "--"],
        annotate=True,
        rounding=1,
    )
    _finalise(
        ax,
        title="Potential growth: hours and productivity",
        ylabel="Year-ended per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_decomposition_rfooter(decomposition),
        lfooter="Australia. Accounting split. Productivity is the residual. ",
        show=False,
    )


def plot_growth_wedge(decomposition: GrowthDecomposition) -> None:
    """Potential growth against trend hours, with productivity as the wedge.

    The same decomposition as `plot_growth_accounting`, presented so that the
    residual is not mistaken for a measurement.

    Drawing trend productivity as its own line invites the reader to treat it
    as an estimate of trend productivity growth. It is not: it is
    `potential growth − trend hours`, and potential growth is nearly a straight
    slow drift (sd 0.68 against 0.96 for the residual). So wherever trend hours
    moves faster than the speed limit does, the productivity line is the hours
    line upside down. Over 1994-2007 the two correlate −0.86, over 2020-2026
    −0.99. That is a property of residuals, not an artefact of any one episode,
    which is why it cannot be fixed by shading a period or truncating the
    sample.

    Shown as a wedge, the identity does the explaining. The 2023 migration
    surge reads as trend hours crossing above potential growth, so the residual
    turns negative — which is the true statement — rather than as a productivity
    line plunging to −1.5, which is not.

    `interpolate=True` is correct here, unlike in `plot_growth_vs_potential`:
    the mask flips exactly where the two drawn curves cross, so there is no
    third variable whose zero sits somewhere else.
    """
    data = pd.DataFrame({
        "Potential growth": decomposition.potential_growth,
        "Trend hours": decomposition.hours_growth,
    }).dropna()

    ax = mg.line_plot(
        data,
        color=["black", "darkorange"],
        width=[2, 2],
        style=["-", "--"],
        annotate=True,
        rounding=1,
    )

    # Read the x-coordinates back from the drawn line, so the fill lands on
    # whatever internal index mgplot mapped the PeriodIndex onto.
    x = np.asarray(ax.get_lines()[0].get_xdata(), dtype=float)
    potential = data["Potential growth"].to_numpy()
    hours = data["Trend hours"].to_numpy()

    ax.fill_between(
        x, potential, hours, where=potential >= hours, interpolate=True,
        color="seagreen", alpha=0.30, label="Trend productivity (positive)",
    )
    ax.fill_between(
        x, potential, hours, where=potential < hours, interpolate=True,
        color="indianred", alpha=0.30, label="Trend productivity (negative)",
    )

    _finalise(
        ax,
        title="Potential growth and labour input",
        ylabel="Year-ended per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_decomposition_rfooter(decomposition),
        lfooter="Australia. The wedge is trend productivity, the residual. ",
        show=False,
    )


def plot_growth_contributions(decomposition: GrowthDecomposition) -> None:
    """Period-average contributions to potential growth, stacked.

    The four components add to potential growth exactly, so the composition is
    why the speed limit moved. Blocks rather than quarters, because the split
    is only informative at low frequency: at quarterly frequency the hours
    trend and the productivity residual are near mirror images of each other.
    """
    blocks = decomposition.block_means()

    ax = mg.bar_plot(
        blocks,
        stacked=True,
        annotate=False,
        color=["cornflowerblue", "darkorange", "seagreen", "indianred"],
    )
    # Negative contributions stack below the axis, so the top of the bar is not
    # the total. Mark the total explicitly rather than let it be misread. The
    # block labels are strings, so the marker series is put on a RangeIndex,
    # which is where bar_plot places the bars.
    totals = blocks.sum(axis=1).rename("Potential growth (total)")
    totals.index = pd.RangeIndex(len(totals))
    mg.line_plot(
        totals,
        ax=ax,
        color=["black"],
        style="None",
        marker="_",
        markersize=40,
        # mgplot annotates the last point only, which here collides with the
        # final bar for no gain: the numbers are in the printed table.
        annotate=False,
    )
    _finalise(
        ax,
        title="Contributions to potential growth",
        ylabel="Year-ended per cent, period average",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_decomposition_rfooter(decomposition),
        lfooter="Australia. Components add to potential growth exactly. ",
        show=False,
    )


def plot_trend_hours_components(results: PotentialResults) -> None:
    """Trend participation and trend hours per labour-force participant."""
    data = pd.DataFrame({
        "Trend participation": results.trend_participation_posterior().median(axis=1),
        "Observed participation": pd.Series(results.obs["log_pr"], index=results.obs_index),
    })

    _line_plot_finalise(
        data,
        color=["darkorange", "black"],
        width=[2, 1],
        style=["-", "-"],
        alpha=[1.0, 0.45],
        annotate=False,
        title="Participation rate: trend and observed",
        ylabel="log x 100",
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER + "Cycle removed via the estimated gap loading. ",
        show=False,
    )


def plot_gap_attribution(results: PotentialResults) -> None:
    """Split the output gap into its hours and productivity components."""
    data = pd.DataFrame({
        "Output gap": results.output_gap_median(),
        "Hours component": results.hours_gap_posterior().median(axis=1),
        "Productivity component": results.productivity_gap_posterior().median(axis=1),
    })

    _line_plot_finalise(
        data,
        color=["black", "darkorange", "seagreen"],
        width=[2, 1.5, 1.5],
        style=["-", "-", "-"],
        annotate=True,
        rounding=1,
        title="Output gap attribution",
        ylabel="Percentage points",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER + "Split by the estimated hours loading. ",
        show=False,
    )


def plot_factor_trends(results: PotentialResults) -> None:
    """Chart the three factor trends, which is what the production spec adds.

    Growth-rate states rather than differenced levels, so they are smooth by
    construction. Bands are omitted because three overlapping fills are
    unreadable; MFP gets its own banded chart, being the one with real
    uncertainty and the one nothing else in the package can put a band on.
    """
    trends = pd.DataFrame({
        "Trend capital growth": results.factor_trend_posterior("gk").median(axis=1),
        "Trend hours growth": results.factor_trend_posterior("gl").median(axis=1),
        "Trend MFP growth": results.factor_trend_posterior("gm").median(axis=1),
    })

    _line_plot_finalise(
        trends,
        title="Trend growth of the factors of production",
        ylabel="Year-ended growth (%)",
        color=["darkorange", "navy", "seagreen"],
        width=2,
        annotate=False,
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def plot_capital_share(results: PotentialResults) -> None:
    """Chart the published capital share against the model's latent trend.

    Worth charting because the two look nothing alike, and without the reason
    on the chart a near-flat trend through a volatile series reads as a failure
    to fit. It is not: almost all of the published movement is discarded on
    purpose. The share correlates +0.85 with the terms of trade and -0.08 with
    potential growth in changes, so it moves when ore prices move and not when
    capacity does. Cobb-Douglas also implies constant factor shares outright.

    The header and footer carry that explanation, which is why they are longer
    than elsewhere in this module. See `ModelConfig.ratio_a`.
    """
    published = pd.Series(results.obs["alpha"], index=results.obs_index)
    trend = results.factor_trend_posterior("a", annualised=False)

    ax = mg.fill_between_plot(_band(trend), **_BAND_KWARGS)
    mg.line_plot(
        published.rename("As published"),
        ax=ax,
        color=["darkgrey"],
        width=1.4,
        annotate=False,
    )
    mg.line_plot(
        trend.median(axis=1).rename("Smoothed by the model"),
        ax=ax,
        color=["indianred"],
        width=2.2,
        annotate=True,
        rounding=3,
    )
    _finalise(
        ax,
        title="Capital share used in the production function",
        ylabel="Share of income",
        legend={"loc": "best", "fontsize": "small"},
        lheader="Near-constant by design: most published movement is the terms of trade, not technology",
        rfooter=_rfooter(results),
        lfooter=(
            "Australia. alpha = GOS/(GOS+COE). Corr +0.85 with terms of trade, -0.08 with potential growth. "
        ),
        show=False,
    )


def plot_trend_mfp(results: PotentialResults) -> None:
    """Trend MFP growth with its credible interval.

    This is the specification's real addition. In `decompose.py` productivity
    is a residual: it absorbs every error in the hours trend and carries no
    uncertainty of its own. Here it is a state, so the band is meaningful, and
    on this data it spans zero — which is the honest reading of Australian
    productivity growth and one the deterministic split cannot express.
    """
    mfp = results.factor_trend_posterior("gm")

    ax = mg.fill_between_plot(_band(mfp), **_BAND_KWARGS)
    mg.line_plot(
        mfp.median(axis=1).rename("Trend MFP growth"),
        ax=ax,
        color=["seagreen"],
        width=2,
        annotate=True,
        rounding=2,
    )
    _finalise(
        ax,
        title="Trend multifactor productivity growth",
        ylabel="Year-ended growth (%)",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def plot_factor_contributions(results: PotentialResults) -> None:
    """Contributions to potential growth, which add to it exactly.

    alpha·g_K*, (1-alpha)·g_L* and g_M*, with alpha the observed smoothed
    capital share. The three sum to potential growth at every quarter, so the
    stack is the composition of the speed limit rather than an approximation
    to it.
    """
    contributions = results.factor_contributions()

    ax = mg.line_plot(
        contributions,
        color=["darkorange", "navy", "seagreen"],
        width=2,
        annotate=False,
    )
    mg.line_plot(
        contributions.sum(axis=1).rename("Potential growth"),
        ax=ax,
        color=["black"],
        width=2.4,
        style="--",
        annotate=False,
    )
    _finalise(
        ax,
        title="Contributions to potential growth",
        ylabel="Percentage points, year-ended",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER + "Capital share is observed and smoothed, not estimated. ",
        show=False,
    )


def run_analysis(
    output_dir: Path | str | None = None,
    prefix: str = "ystar",
    chart_dir: Path | str | None = None,
    decompose: bool = True,
) -> PotentialResults:
    """Load results, print diagnostics, and write every chart.

    `decompose` adds the post-modelling accounting split of potential growth
    into hours and productivity (see `decompose.py`). It loads labour force
    data, so it is the only part of the analysis that touches ABS sources; pass
    False to chart from the trace alone. It is skipped for `labour` and
    `production`, both of which estimate a split internally.
    """
    results = load_results(output_dir=output_dir, prefix=prefix)

    # Read from the run's own recorded settings rather than passed in, so a
    # chart can never disagree with the trace it was drawn from. Reset each
    # call: a session that analyses an excluded-window run and then a normal one
    # would otherwise carry the shading over to the second.
    global _EXCLUDED_WINDOW  # noqa: PLW0603 — run-scoped state, see the definition
    _EXCLUDED_WINDOW = _excluded_window(results)

    print_diagnostics(results)

    # Default per specification, so one spec's run cannot clear another's
    # charts. An explicit chart_dir still wins.
    if chart_dir is None:
        chart_dir = SPEC_CHART_DIRS.get(results.spec, CHART_DIR)

    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    plot_potential(results, tag="full")
    plot_potential(results, plot_from="2015Q1", tag="recent")
    plot_actual_output_gap(results, tag="full")
    plot_actual_output_gap(results, plot_from="2015Q1", tag="recent")
    # Only where the gap is defined by inflation. Under the identity
    # specifications `output_gap` is log_gdp - y* itself, so this chart would
    # duplicate the one above.
    if results.spec in ("inflation", "production"):
        plot_inflation_defined_gap(results)
        plot_gap_composition(results)

    # Two windows: the full sample, and one that excludes the COVID swing,
    # which otherwise dominates the scale and hides the recent story.
    plot_growth_vs_potential(results, tag="full")
    plot_growth_vs_potential(results, plot_from="2015Q1", tag="recent")
    plot_gdp_growth_against_potential(results, tag="full")
    plot_gdp_growth_against_potential(results, plot_from="2015Q1", tag="recent")
    plot_gov_growth_against_potential(results, tag="full")
    plot_gov_growth_against_potential(results, plot_from="2015Q1", tag="recent")

    if results.spec == "production":
        plot_factor_trends(results)
        plot_trend_mfp(results)
        plot_factor_contributions(results)
        plot_capital_share(results)

    if results.spec in ("inflation", "production", "core", "target"):
        plot_trend_growth(results)
        # `production` splits potential growth internally, into capital, hours
        # and MFP with credible intervals, so the post-modelling accounting
        # split would be a second, weaker answer to the same question: it
        # re-derives productivity as a residual from a path that already has
        # it as a state. Skipped there rather than shown alongside.
        if decompose and results.spec != "production":
            decomposition = decompose_potential_growth(results)
            print_decomposition(decomposition)
            plot_growth_accounting(decomposition)
            plot_growth_wedge(decomposition)
            plot_growth_contributions(decomposition)
    else:
        plot_trend_productivity_growth(results)
        plot_potential_growth(results)
        plot_trend_hours_components(results)
        plot_gap_attribution(results)

    print(f"\nCharts written to: {chart_dir}")
    return results


if __name__ == "__main__":
    run_analysis()
