"""Diagnostics and charts for the potential_uc model."""

from typing import TYPE_CHECKING, Any

import mgplot as mg
import numpy as np
import pandas as pd

from src.data.henderson import hma
from src.models.potential_uc.decompose import (
    GrowthDecomposition,
    decompose_potential_growth,
    print_decomposition,
)
from src.models.potential_uc.results import (
    DEFAULT_CHART_BASE,
    PotentialResults,
    load_results,
)

if TYPE_CHECKING:
    from pathlib import Path

CHART_DIR = DEFAULT_CHART_BASE / "PotentialUC"

_RFOOTER = "Source: ABS 5206.0, 6202.0, 6401.0"
_RFOOTER_CORE = "Source: ABS 5206.0, 6401.0"

# Shown on the output gap chart, where the sign is the thing to read.
_GAP_HEADER = "A positive output gap is consistent with inflation"
_LFOOTER = "Australia. Unobserved-components model. "
# Points per quarter used when shading the growth-versus-potential chart.
_FILL_SUBDIVISIONS = 20

_BAND_KWARGS: dict[str, Any] = {
    "color": "cornflowerblue",
    "alpha": 0.25,
    "label": "90% credible interval",
}


def _band(posterior: pd.DataFrame) -> pd.DataFrame:
    return PotentialResults.band(posterior)


def _rfooter(results: PotentialResults) -> str:
    """Source line naming only the catalogues this specification actually uses."""
    return _RFOOTER if results.spec == "labour" else _RFOOTER_CORE


def print_diagnostics(results: PotentialResults) -> None:
    """Print the parameter summary and the model's headline numbers."""
    print("\nPosterior summary")
    print("-" * 70)
    print(results.summary().to_string())

    pot = results.potential_growth_posterior()
    gap = results.output_gap_posterior()
    last = pot.index[-1]

    headline = [("Potential growth", pot), ("Output gap", gap)]
    if results.spec in ("inflation", "core", "target"):
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

    if "phi_1" in results.trace.posterior:
        roots = results.ar_root_posterior()
        print("\nCycle stationarity (largest AR root modulus)")
        print("-" * 70)
        print(
            f"  {'|root|':<28} {np.median(roots):6.2f}"
            f"  [{np.quantile(roots, 0.05):5.2f}, {np.quantile(roots, 0.95):5.2f}]",
        )
        print(f"  {'draws stationary':<28} {(roots < 1.0).mean():6.1%}")
    elif results.spec == "inflation":
        # The gap is defined by inflation, so there is no cycle to be
        # stationary. What matters instead is how much of output's deviation
        # from potential that definition actually accounts for.
        gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)
        # Not `gap`: that name is already bound to the time x draw DataFrame above.
        gap_median = results.output_gap_median()
        deviation = gdp - results.potential_median()
        print("\nHow much of the cycle the inflation-defined gap explains")
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
    mg.finalise_plot(
        ax,
        tag=tag,
        title="GDP and potential output",
        ylabel="log level x 100",
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def plot_output_gap(results: PotentialResults) -> None:
    """Output gap with a credible band."""
    gap = results.output_gap_posterior()

    ax = mg.fill_between_plot(_band(gap), **_BAND_KWARGS)
    median = gap.median(axis=1)
    mg.line_plot(
        median.rename("Output gap"),
        ax=ax,
        color=["darkorange"],
        width=2,
        annotate=True,
        rounding=1,
    )
    mg.finalise_plot(
        ax,
        title="Output gap",
        ylabel="Per cent of potential",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=_GAP_HEADER,
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


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

    mg.finalise_plot(
        ax,
        title="Actual growth versus potential",
        ylabel="Year-ended per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter="Source: ABS 5206.0, 6401.0",
        lfooter=_LFOOTER + "Actual smoothed, 7-term Henderson MA. ",
        tag=tag,
        show=False,
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
        rounding=1,
    )
    mg.finalise_plot(
        ax,
        title="Potential growth",
        ylabel="Year-ended per cent" if residual_potential else "Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter="Source: ABS 5206.0, 6401.0",
        lfooter=_LFOOTER + "Identified from output and inflation alone. ",
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
    mg.finalise_plot(
        ax,
        title="Trend labour productivity growth",
        ylabel="Annualised per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_RFOOTER,
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
    mg.finalise_plot(
        ax,
        title="Potential output growth and its components",
        ylabel="Year-ended per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_RFOOTER,
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
    mg.finalise_plot(
        ax,
        title="Potential growth: hours and productivity",
        ylabel="Year-ended per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_RFOOTER,
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

    mg.finalise_plot(
        ax,
        title="Potential growth and labour input",
        ylabel="Year-ended per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_RFOOTER,
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
    mg.finalise_plot(
        ax,
        title="Contributions to potential growth",
        ylabel="Year-ended per cent, period average",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_RFOOTER,
        lfooter="Australia. Components add to potential growth exactly. ",
        show=False,
    )


def plot_trend_hours_components(results: PotentialResults) -> None:
    """Trend participation and trend hours per labour-force participant."""
    data = pd.DataFrame({
        "Trend participation": results.trend_participation_posterior().median(axis=1),
        "Observed participation": pd.Series(results.obs["log_pr"], index=results.obs_index),
    })

    mg.line_plot_finalise(
        data,
        color=["darkorange", "black"],
        width=[2, 1],
        style=["-", "-"],
        alpha=[1.0, 0.45],
        annotate=False,
        title="Participation rate: trend and observed",
        ylabel="log x 100",
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_RFOOTER,
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

    mg.line_plot_finalise(
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
        rfooter=_RFOOTER,
        lfooter=_LFOOTER + "Split by the estimated hours loading. ",
        show=False,
    )


def run_analysis(
    output_dir: Path | str | None = None,
    prefix: str = "potential_uc",
    chart_dir: Path | str = CHART_DIR,
    decompose: bool = True,
) -> PotentialResults:
    """Load results, print diagnostics, and write every chart.

    `decompose` adds the post-modelling accounting split of potential growth
    into hours and productivity (see `decompose.py`). It loads labour force
    data, so it is the only part of the analysis that touches ABS sources; pass
    False to chart from the trace alone. It is skipped for the `labour`
    specification, which estimates that split internally.
    """
    results = load_results(output_dir=output_dir, prefix=prefix)

    print_diagnostics(results)

    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    plot_potential(results, tag="full")
    plot_potential(results, plot_from="2015Q1", tag="recent")
    plot_output_gap(results)

    # Two windows: the full sample, and one that excludes the COVID swing,
    # which otherwise dominates the scale and hides the recent story.
    plot_growth_vs_potential(results, tag="full")
    plot_growth_vs_potential(results, plot_from="2015Q1", tag="recent")

    if results.spec in ("inflation", "core", "target"):
        plot_trend_growth(results)
        if decompose:
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
