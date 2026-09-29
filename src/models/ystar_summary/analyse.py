"""Chart for the output gap summary."""

from pathlib import Path
from typing import Any

import mgplot as mg
import pandas as pd

from src.models.common.charts import excluded_span_style
from src.models.ystar_summary.sources import Window
from src.paths import CHARTS

CHART_DIR = CHARTS / "ystar-summary"


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


def run_analyse(
    frame: pd.DataFrame,
    notes: dict[str, str],
    windows: dict[str, Window],
    start: str | None = "1993Q1",
    chart_dir: Path | str | None = None,
) -> None:
    """Write the chart and print what each line is and its latest reading."""
    directory = Path(chart_dir) if chart_dir else CHART_DIR
    directory.mkdir(parents=True, exist_ok=True)
    mg.set_chart_dir(str(directory))
    mg.clear_chart_dir()

    plot_summary(frame, windows, start=start)

    print("\nWhat each line is:")
    for label, note in notes.items():
        print(f"  {label:<32} {note}")
    print("\nLatest output gap:")
    for column, series in frame.items():
        clean = series.dropna()
        if not clean.empty:
            print(f"  {column!s:<32} {clean.iloc[-1]:5.2f}   ({clean.index[-1]})")
    print(f"\nCharts written to: {directory}")
