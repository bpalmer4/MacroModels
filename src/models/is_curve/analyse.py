"""Charts and printed diagnostics for the IS-curve scatter.

One chart per r* variant, each the same points against a different x-axis.
mgplot has no scatter function, so the points and the fitted line are drawn
with matplotlib and `finalise_plot` closes the chart out, which is the
documented pattern for composite charts.

Points are coloured by date. A scatter of forty years of quarters can show a
downward slope purely because the 1990s had high real rates and a different
average gap from the 2010s, which is a comparison across eras rather than the
within-era response an IS curve claims. The colouring makes that visible
instead of leaving it to be assumed.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd

from src.models.is_curve.fit import DEFAULT_LAG, FitResult, fit, lag_sweep
from src.models.is_curve.observations import (
    DEFAULT_START,
    DEFAULT_WINDOWS,
    IsCurveData,
    blocks,
    build_observations,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "ISCurve"

_LFOOTER = "Australia. Output gap from the joint y*/u* model. "

# What each variant puts on the x-axis, for the axis label and the title.
_X_LABEL = {
    "none": "Real cash rate (cash - expectations), %",
    "rstar": "Real rate gap using the bond-market r*, ppt",
    "rule": "Real rate gap against the reaction-function neutral b_t, ppt",
    "constant": "Real rate gap using a flat r*, ppt",
}
_TITLE = {
    "none": "IS curve: no r* imposed",
    "rstar": "IS curve: bond-market r*",
    "rule": "IS curve: reaction-function neutral",
    "constant": "IS curve: flat r*",
}

# How to read the zero-gap crossing, which differs by variant: under `none` it
# is r* itself, under the others it is the correction to the r* subtracted.
_CROSSING_LABEL = {
    "none": "implied r*",
    "rstar": "r* too low by",
    "rule": "r* too low by",
    "constant": "r* too low by",
}


def _colour_by_date(index: pd.PeriodIndex) -> np.ndarray:
    """Return a 0-1 position for each quarter, for the colour map."""
    years = np.array([period.year + (period.quarter - 1) / 4 for period in index], dtype=float)
    span = years.max() - years.min()
    return (years - years.min()) / span if span > 0 else np.zeros_like(years)


def _exclusion_note(data: IsCurveData) -> str:
    """Return the footer clause naming the quarters left out, if any."""
    if not len(data.excluded):
        return "All quarters kept. "
    return f"{data.excluded[0]}-{data.excluded[-1]} excluded. "


def _fit_header(result: FitResult) -> str:
    """Return the fit summary, for the chart's right header.

    In the header rather than a box inside the axes: an in-axes annotation
    competes with the legend for whichever corner is emptiest, and the two
    collided as soon as the sample changed.
    """
    return (
        f"slope {result.slope:+.3f} (t {result.t_slope:+.2f}) | "
        f"intercept {result.intercept:+.3f} | R2 {result.r_squared:.3f} | n {result.n}"
    )


def _reading_header(result: FitResult, variant: str) -> str:
    """Return the one-line reading of what the fit does or does not support."""
    if result.crossing is not None:
        return f"{_CROSSING_LABEL[variant]} {result.crossing:+.2f}"
    if result.slope > 0:
        return "Slope has the wrong sign for an IS curve: no r* can be read off it"
    return "Slope too flat to locate r*"


def plot_variant(data: IsCurveData, variant: str, lag: int = DEFAULT_LAG) -> FitResult:
    """Draw the scatter, the fitted line and the intercept for one variant.

    Args:
        data: the assembled observations
        variant: which r* treatment to plot
        lag: quarters by which the rate is lagged

    Returns:
        The FitResult that was drawn

    """
    x_series, y_series = data.xy(variant)
    frame = pd.DataFrame({"x": x_series.shift(lag), "y": y_series}).dropna()
    frame = frame.drop(index=data.excluded, errors="ignore")
    result = fit(x_series, y_series, variant, lag, data.excluded)

    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError(f"expected a PeriodIndex, got {type(index).__name__}")

    _, ax = plt.subplots(figsize=(9, 6))
    points = ax.scatter(
        frame["x"], frame["y"],
        c=_colour_by_date(index), cmap="viridis",
        s=34, alpha=0.85, edgecolors="none", zorder=3,
    )

    grid = np.linspace(frame["x"].min(), frame["x"].max(), 100)
    ax.plot(grid, result.line(grid), color="crimson", linewidth=2, zorder=4,
            label=f"OLS: gap = {result.intercept:+.2f} {result.slope:+.3f} x rate")

    ax.axhline(0.0, color="darkgrey", linewidth=0.9, zorder=1)
    ax.axvline(0.0, color="darkgrey", linewidth=0.9, zorder=1)
    if result.crossing is not None:
        ax.plot(
            [result.crossing], [0.0], marker="o", markersize=11,
            markerfacecolor="none", markeredgecolor="crimson", markeredgewidth=2, zorder=5,
            label=f"{_CROSSING_LABEL[variant]} {result.crossing:+.2f}",
        )

    colourbar = ax.figure.colorbar(points, ax=ax, pad=0.02)
    colourbar.set_ticks([0.0, 1.0])
    colourbar.set_ticklabels([str(index[0]), str(index[-1])])

    mg.finalise_plot(
        ax,
        title=f"{_TITLE[variant]} (rate lagged {lag}q)",
        xlabel=_X_LABEL[variant],
        ylabel="Output gap, % of potential",
        legend={"loc": "best", "fontsize": "small"},
        lheader=_reading_header(result, variant),
        rheader=_fit_header(result),
        lfooter=_LFOOTER + _exclusion_note(data),
        rfooter=data.sources.footer(),
        show=False,
    )
    return result


def print_diagnostics(data: IsCurveData, lag: int = DEFAULT_LAG) -> None:
    """Print the fits and the lag sweeps behind the charts."""
    print("\nFitted lines")
    print("-" * 96)
    for variant in data.variants:
        x_series, y_series = data.xy(variant)
        print("  " + fit(x_series, y_series, variant, lag, data.excluded).summary_line())

    kept_blocks = blocks(data.index, data.excluded)
    if len(kept_blocks) > 1:
        print("\nEach block on its own: is the pooled slope a comparison between eras?")
        print("-" * 96)
        for variant in data.variants:
            x_series, y_series = data.xy(variant)
            for block in kept_blocks:
                outside = data.index.difference(block)
                label = f"{variant} {block[0]}-{block[-1]}"
                try:
                    result = fit(x_series, y_series, label, lag, outside)
                except ValueError as exc:  # a short block is reported, not fatal
                    print(f"  {label}: {exc}")
                    continue
                print("  " + result.summary_line())

    print("\nSlope by lag: a weak slope should not be a timing mistake")
    print("-" * 96)
    for variant in data.variants:
        x_series, y_series = data.xy(variant)
        print(f"\n  {variant}")
        print(lag_sweep(x_series, y_series, variant, drop=data.excluded).round(3).to_string())


def run_analysis(
    start: str | None = None,
    end: str | None = None,
    *,
    lag: int = DEFAULT_LAG,
    joint_prefix: str = "ystar_ustar",
    rstar_prefix: str = "rstar_bonds",
    rule_prefix: str = "rstar_rba",
    exclude_windows: Sequence[tuple[str, str]] | None = DEFAULT_WINDOWS,
) -> dict[str, FitResult]:
    """Assemble the data, chart each variant and print the diagnostics.

    Args:
        start: first quarter, or None for the default
        end: last quarter, or None for the latest available
        lag: quarters by which the rate is lagged
        joint_prefix: prefix of the saved joint y*/u* run
        rstar_prefix: prefix of the saved rstar run
        rule_prefix: prefix of the saved rstar_rba run, whose nominal r*
            is converted to real before use
        exclude_windows: windows left out of the fits, or None to keep all

    Returns:
        The fitted line for each variant, keyed by variant name

    """
    data = build_observations(
        start=DEFAULT_START if start is None else start,
        end=end,
        joint_prefix=joint_prefix,
        rstar_prefix=rstar_prefix,
        rule_prefix=rule_prefix,
        exclude_windows=exclude_windows,
    )

    mg.set_chart_dir(str(CHART_DIR))
    mg.clear_chart_dir()

    results = {variant: plot_variant(data, variant, lag) for variant in data.variants}
    print_diagnostics(data, lag)
    print(f"\nCharts written to {CHART_DIR}")
    return results
