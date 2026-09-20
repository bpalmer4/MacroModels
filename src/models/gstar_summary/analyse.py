"""Charts for the potential-growth summary."""

from pathlib import Path

import mgplot as mg
import pandas as pd

from src.paths import CHARTS

CHART_DIR = CHARTS / "gstar-summary"

# The two y* specs are deliberately adjacent in hue, being one model run two
# ways; the joint model is clearly separate from both.
_COLOURS = ("navy", "steelblue", "darkorange", "seagreen")


def plot_summary(
    frame: pd.DataFrame,
    notes: dict[str, str],
    start: str | None = "1993Q1",
) -> None:
    """Plot every model's potential growth on one axis."""
    data = frame.loc[frame.index >= pd.Period(start, freq="Q")] if start else frame
    index = data.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    # Actual GDP growth is deliberately NOT drawn here. Its COVID swing runs
    # -6 to +10 and squeezes lines that differ by a tenth of a point into an
    # unreadable band, which defeats the one job this chart has. `ystar` pairs
    # actual against potential on its own charts, at a sensible scale.
    ax = mg.line_plot(
        data,
        color=list(_COLOURS[:len(data.columns)]),
        width=2.0,
        annotate=True,
        rounding=2,
    )
    latest = " | ".join(
        f"{str(label).split(' (')[0]} {series.dropna().iloc[-1]:.2f}"
        for label, series in data.items() if series.notna().any()
    )
    mg.finalise_plot(
        ax,
        title="Australian potential output growth: every model that estimates one",
        ylabel="Year-ended per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        lheader="All three share the y* state-space core: agreement is not independence",
        rheader=f"Latest: {latest}",
        rfooter="Built using: ABS",
        lfooter="Australia. ",
        show=False,
    )
    print("\nWhat each line is built from:")
    for label, note in notes.items():
        print(f"  {label:<36} {note}")


def plot_spread(frame: pd.DataFrame, start: str | None = "1993Q1") -> None:
    """Plot the range across models, which is the disagreement itself.

    Worth reading against `rstar_summary`'s equivalent chart. There the band is
    about a percentage point wide and no model identifies a level. Here it is a
    fifth of that, which is the substantive difference between what this repo
    knows about potential growth and what it knows about the neutral rate.
    """
    data = frame.loc[frame.index >= pd.Period(start, freq="Q")] if start else frame
    usable = data.dropna(how="any")
    if usable.empty or len(usable.columns) < 2:  # noqa: PLR2004 — a range needs two
        print("  note: fewer than two models overlap; skipping the spread chart")
        return

    band = pd.DataFrame({"lower": usable.min(axis=1), "upper": usable.max(axis=1)})
    ax = mg.fill_between_plot(band, color="crimson", alpha=0.18, label="Range across models")
    # A mean across structural assumptions is a value no model produces. It
    # describes where the models sit; it is not an estimate. Same caveat as
    # `rstar_summary`, and the same reason `rstar_hlw`'s notes object to its
    # own blended median.
    mg.line_plot(
        usable.mean(axis=1).rename("Mean across models"),
        ax=ax, color=["crimson"], width=2.5, annotate=True, rounding=2,
    )
    width = band["upper"] - band["lower"]
    mg.finalise_plot(
        ax,
        title="How much the models disagree about potential growth",
        ylabel="Year-ended per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"Across {len(usable.columns)} sources, on their common quarters",
        rheader=f"Widest {width.max():.2f}pp in {width.idxmax()}; "
                f"latest {width.iloc[-1]:.2f}pp",
        rfooter="Built using: ABS",
        lfooter="Australia. The mean describes where the models sit, not an estimate. ",
        show=False,
    )


def run_analyse(
    frame: pd.DataFrame,
    notes: dict[str, str],
    start: str | None = "1993Q1",
    chart_dir: Path | str | None = None,
) -> None:
    """Write both charts and print the latest readings."""
    directory = Path(chart_dir) if chart_dir else CHART_DIR
    directory.mkdir(parents=True, exist_ok=True)
    mg.set_chart_dir(str(directory))
    mg.clear_chart_dir()

    plot_summary(frame, notes, start=start)
    plot_spread(frame, start=start)

    print("\nLatest potential growth:")
    for label, series in frame.items():
        clean = series.dropna()
        if not clean.empty:
            print(f"  {label!s:<36} {clean.iloc[-1]:5.2f}   ({clean.index[-1]})")
    print(f"\nCharts written to: {directory}")
