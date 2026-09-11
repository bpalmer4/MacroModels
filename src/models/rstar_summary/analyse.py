"""The summary chart: every r* this repo produces, on one nominal scale.

The point is the DISAGREEMENT. Each model anchors r* to a different thing and
each returns roughly what its anchor implies, so the spread between these lines
is not sampling error, it is four different structural assumptions. Nothing
here picks a winner.

The nominal cash rate is drawn behind them because it is the comparison anyone
reading an r* chart is actually making, and because a line that tracks it is
reporting policy back rather than measuring neutral.
"""

from pathlib import Path

import mgplot as mg
import pandas as pd

from src.data.cash_rate import get_cash_rate_qrtly
from src.models.rstar_summary.sources import TARGET

CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "rstar-summary"

# Chosen for contrast against each other AND against the grey cash rate, which
# matters more than matching each model's own suite colours.
_COLOURS = ("darkblue", "crimson", "darkgreen", "darkorange", "purple", "teal")


def _cash_rate(index: pd.PeriodIndex) -> pd.Series:
    """Return the nominal cash rate on the chart's index."""
    cash = get_cash_rate_qrtly().data.astype(float)
    if isinstance(cash.index, pd.DatetimeIndex):
        cash.index = cash.index.to_period("Q")
    return cash.reindex(index)


def plot_summary(
    frame: pd.DataFrame,
    notes: dict[str, str],
    start: str | None = "1993Q1",
) -> None:
    """Plot every model's nominal r* together, with the cash rate behind them."""
    data = frame.loc[frame.index >= pd.Period(start, freq="Q")] if start else frame
    index = data.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    ax = mg.line_plot(
        _cash_rate(index).rename("Nominal cash rate"),
        color=["darkgrey"], width=1.5, style="--", annotate=False,
    )
    mg.line_plot(
        data,
        ax=ax,
        color=list(_COLOURS[:len(data.columns)]),
        width=2.0,
        annotate=True,
        rounding=2,
    )
    # Column labels are Hashable to the type checker, so the short name is
    # taken from a string built here rather than by splitting the label object.
    latest = " | ".join(
        f"{str(label).split(' (')[0]} {series.dropna().iloc[-1]:.2f}"
        for label, series in data.items() if series.notna().any()
    )
    mg.finalise_plot(
        ax,
        title="Australian nominal r*: the models that identify one",
        ylabel="Per cent, nominal",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        lheader=f"All converted to nominal: real estimates plus the {TARGET:g}% target",
        rheader=f"Latest: {latest}",
        rfooter="Built using: ABS; RBA; NY Fed",
        lfooter=(
            f"Australia. Spread between lines is {len(data.columns)} structural "
            "assumptions, not error. "
        ),
        show=False,
    )
    print("\nWhat each line is anchored to:")
    for label, note in notes.items():
        print(f"  {label:<40} {note}")


def plot_spread(frame: pd.DataFrame, start: str | None = "1993Q1") -> None:
    """Plot the range across models, which is the disagreement itself.

    A single number is only as good as the narrowest this gets. Where the band
    is wide the models are saying different things about the same quarter, and
    no amount of within-model precision closes it.
    """
    data = frame.loc[frame.index >= pd.Period(start, freq="Q")] if start else frame
    usable = data.dropna(how="any")
    if usable.empty or len(usable.columns) < 2:  # noqa: PLR2004 — a range needs two
        print("  note: fewer than two models overlap; skipping the spread chart")
        return

    band = pd.DataFrame({"lower": usable.min(axis=1), "upper": usable.max(axis=1)})
    ax = mg.fill_between_plot(band, color="crimson", alpha=0.18, label="Range across models")
    # MEAN, not median. With three series the median is whichever model happens
    # to sit in the middle that quarter, so it switches identity wherever the
    # lines cross (around 2001, 2010 and 2019) and picks up kinks that say
    # nothing about r*. The mean uses all three and moves smoothly.
    #
    # Neither is an estimate. An average across structural assumptions is a
    # value no model produces, which is the objection `rstar_hlw`'s notes make
    # to its own blended median. It describes where the models sit.
    mg.line_plot(
        usable.mean(axis=1).rename("Mean across models"),
        ax=ax, color=["crimson"], width=2.5, annotate=True, rounding=2,
    )
    widest = (band["upper"] - band["lower"])
    mg.finalise_plot(
        ax,
        title="How much the models disagree about nominal r*",
        ylabel="Per cent, nominal",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"Across {len(usable.columns)} models, on their common quarters",
        rheader=f"Widest {widest.max():.2f}pp in {widest.idxmax()}; "
                f"latest {widest.iloc[-1]:.2f}pp",
        rfooter="Built using: ABS; RBA; NY Fed",
        lfooter="Australia. The mean is a description of where the models sit, not an estimate. ",
        show=False,
    )


def run_analyse(
    frame: pd.DataFrame,
    notes: dict[str, str],
    chart_dir: Path | str | None = None,
    start: str | None = "1993Q1",
) -> None:
    """Produce the summary charts."""
    directory = Path(chart_dir) if chart_dir else CHART_DIR
    mg.set_chart_dir(str(directory))
    mg.clear_chart_dir()

    plot_summary(frame, notes, start=start)
    plot_spread(frame, start=start)
    print(f"\nCharts saved to: {directory}")
