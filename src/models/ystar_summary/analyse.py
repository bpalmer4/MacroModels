"""Chart for the output gap summary."""

from pathlib import Path
from typing import Any

import mgplot as mg
import pandas as pd

from src.models.common.charts import excluded_span_style
from src.models.common.pooling import BAND_50, BAND_90, COMBINED_MEDIAN
from src.models.ystar_summary.sources import Window
from src.paths import CHARTS

CHART_DIR = CHARTS / "ystar-summary"

# A range needs at least two models on the same quarters.
MIN_MODELS_FOR_RANGE = 2


def _shared_window(windows: dict[str, Window]) -> Window:
    """Return the excluded window if every run dropped the same one, else None.

    Shading a window some lines were fitted through would tell the reader those
    quarters were unobserved for all of them. When the runs disagree the chart
    carries no shading and the run log names each run's window instead.
    """
    distinct = set(windows.values())
    if len(distinct) == 1:
        return distinct.pop()
    print("  note: the runs exclude different windows, so none is shaded:")
    for label, window in windows.items():
        print(f"    {label:<32} {'-'.join(window) if window else 'none'}")
    return None


def _span(window: Window) -> dict[str, Any] | None:
    """Return axvspan kwargs for the shared excluded window, or None."""
    if window is None:
        return None
    lo, hi = window
    style = excluded_span_style()
    return {
        "xmin": pd.Period(lo, freq="Q"),
        "xmax": pd.Period(hi, freq="Q"),
        **style,
        "label": f"{style['label']}, {lo}-{hi}",
    }


def plot_summary(
    frame: pd.DataFrame,
    windows: dict[str, Window],
    start: str | None = "1993Q1",
) -> None:
    """Plot every model's output gap on one axis."""
    data = frame.loc[frame.index >= pd.Period(start, freq="Q")] if start else frame
    ax = mg.line_plot(
        data,
        width=2.0,
        annotate=True,
        rounding=2,
    )
    finalise: dict[str, Any] = {
        "title": "Australian output gap: every live model that estimates y*",
        "ylabel": "Per cent of potential",
        "y0": True,
        "legend": {"loc": "best", "fontsize": "x-small"},
        "lheader": "Each model's own definition of the gap: they are not the same object",
        "rfooter": "Built using: ABS",
        "lfooter": "Australia. Posterior medians. ",
        "show": False,
    }
    span = _span(_shared_window(windows))
    if span is not None:
        finalise["axvspan"] = span
    mg.finalise_plot(ax, **finalise)


def plot_spread(
    frame: pd.DataFrame,
    windows: dict[str, Window],
    start: str | None = "1993Q1",
) -> None:
    """Plot the range across the models' gaps, quarter by quarter.

    No mean is drawn: the gaps are different objects (a component of the cycle
    and the whole of it), so their average describes nothing. Much of the range
    is that difference in definition rather than disagreement about the economy.
    """
    data = frame.loc[frame.index >= pd.Period(start, freq="Q")] if start else frame
    usable = data.dropna(how="any")
    if usable.empty or len(usable.columns) < MIN_MODELS_FOR_RANGE:
        print("  note: fewer than two models overlap; skipping the spread chart")
        return

    band = pd.DataFrame({"lower": usable.min(axis=1), "upper": usable.max(axis=1)})
    ax = mg.fill_between_plot(band, color="darkblue", alpha=0.18, label="Range across models")
    # The band's edges as thin lines, so their latest values are annotated. The
    # leading underscore keeps them out of the legend.
    mg.line_plot(
        band.rename(columns={"upper": "_top", "lower": "_bottom"})[["_top", "_bottom"]],
        ax=ax, color=["darkblue", "darkblue"], width=[0.25, 0.25], annotate=True, rounding=2,
    )
    width = band["upper"] - band["lower"]
    finalise: dict[str, Any] = {
        "title": "How much the models disagree about the output gap",
        "ylabel": "Per cent of potential",
        "y0": True,
        "legend": {"loc": "best", "fontsize": "small"},
        "lheader": "Each model's own definition: much of the range is definitional",
        "rheader": f"Widest {width.max():.2f}pp in {width.idxmax()}; latest {width.iloc[-1]:.2f}pp",
        "rfooter": "Built using: ABS",
        "lfooter": f"Australia. Posterior medians, {len(usable.columns)} models on their common quarters. ",
        "show": False,
    }
    span = _span(_shared_window(windows))
    if span is not None:
        finalise["axvspan"] = span
    mg.finalise_plot(ax, **finalise)


def plot_combined(
    combined: pd.DataFrame,
    models: int,
    windows: dict[str, Window],
    start: str | None = "1993Q1",
) -> None:
    """Plot the pooled median and bands of every model's gap draws.

    The gaps are different objects, a component of the cycle and the whole of
    it, so the pool mixes definitions as well as models: its bands hold
    within-model uncertainty, between-model disagreement and the difference in
    definition at once.
    """
    data = combined.loc[combined.index >= pd.Period(start, freq="Q")] if start else combined
    ax = mg.fill_between_plot(data[list(BAND_90)], color="darkblue", alpha=0.12,
                              label=f"{round((BAND_90[1] - BAND_90[0]) * 100)}% band")
    mg.fill_between_plot(data[list(BAND_50)], ax=ax, color="darkblue", alpha=0.25,
                         label=f"{round((BAND_50[1] - BAND_50[0]) * 100)}% band")
    mg.line_plot(
        data[COMBINED_MEDIAN].rename("Combined median"),
        ax=ax, color=["darkblue"], width=2.0, annotate=True, rounding=2, fontsize="x-small",
    )
    finalise: dict[str, Any] = {
        "title": f"Australian output gap: combined estimate of {models} models",
        "ylabel": "Per cent of potential",
        "y0": True,
        "legend": {"loc": "best", "fontsize": "x-small"},
        "lheader": "Pools three different definitions of the gap, not one",
        "rfooter": "Built using: ABS",
        "lfooter": "Australia. Posterior draws pooled, equal weights. ",
        "show": False,
    }
    span = _span(_shared_window(windows))
    if span is not None:
        finalise["axvspan"] = span
    mg.finalise_plot(ax, **finalise)


def run_analyse(
    frame: pd.DataFrame,
    combined: pd.DataFrame,
    notes: dict[str, str],
    windows: dict[str, Window],
    *,
    start: str | None = "1993Q1",
    chart_dir: Path | str | None = None,
) -> None:
    """Write the three charts and print what each line is and its latest reading."""
    directory = Path(chart_dir) if chart_dir else CHART_DIR
    directory.mkdir(parents=True, exist_ok=True)
    mg.set_chart_dir(str(directory))
    mg.clear_chart_dir()

    plot_summary(frame, windows, start=start)
    plot_spread(frame, windows, start=start)
    plot_combined(combined, len(frame.columns), windows, start=start)

    print("\nWhat each line is:")
    for label, note in notes.items():
        print(f"  {label:<32} {note}")
    print("\nLatest output gap:")
    for column, series in frame.items():
        clean = series.dropna()
        if not clean.empty:
            print(f"  {column!s:<32} {clean.iloc[-1]:5.2f}   ({clean.index[-1]})")
    latest = combined.iloc[-1]
    print(f"  {'Combined median':<32} {latest[COMBINED_MEDIAN]:5.2f}   ({combined.index[-1]})")
    for band in (BAND_50, BAND_90):
        name = f"Combined {round((band[1] - band[0]) * 100)}% band"
        print(f"  {name:<32} {latest[band[0]]:5.2f} to {latest[band[1]]:.2f}")
    print(f"\nCharts written to: {directory}")
