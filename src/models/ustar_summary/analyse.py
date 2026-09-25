"""Charts for the u* summary."""

from pathlib import Path

import mgplot as mg
import pandas as pd

from src.models.ustar.analyse import UNIDENTIFIED_WINDOW as USTAR_WINDOW
from src.models.ystar_ustar.analyse import UNIDENTIFIED_WINDOW as JOINT_WINDOW
from src.paths import CHARTS

CHART_DIR = CHARTS / "ustar-summary"

# A range needs at least two specifications on the same quarters.
MIN_MODELS_FOR_RANGE = 2

# The window both models mark as poorly identified, read from each so neither
# can drift from this chart. Where they differ, the shading covers both.
_WINDOW = (min(USTAR_WINDOW[0], JOINT_WINDOW[0]), max(USTAR_WINDOW[1], JOINT_WINDOW[1]))
_UNIDENTIFIED = {
    "xmin": pd.Period(_WINDOW[0], freq="Q"),
    "xmax": pd.Period(_WINDOW[1], freq="Q"),
    "color": "darkorange",
    "alpha": 0.10,
    "zorder": 0,
    "label": f"u* not well identified, {_WINDOW[0]}-{_WINDOW[1]}",
}

# Not a model, and it loads nothing of its own beyond the saved runs, so the
# footer names the models rather than claiming their data providers.
_RFOOTER = "Built from saved runs of ustar and ystar_ustar"


def _from[T: (pd.DataFrame, pd.Series)](data: T, start: str | None) -> T:
    """Return `data` from `start` on, or all of it."""
    return data.loc[data.index >= pd.Period(start, freq="Q")] if start else data


def plot_summary(
    frame: pd.DataFrame,
    unemployment: pd.Series,
    start: str | None = "1993Q1",
) -> None:
    """Plot every specification's u* on one axis, against the unemployment rate.

    mgplot chooses the colours and line styles for the u* lines. The
    unemployment rate is drawn thin and black behind them, because every line
    is answering how far u* may depart from it.
    """
    data = _from(frame, start)
    ax = mg.line_plot(
        data,
        width=2.0,
        annotate=True,
        rounding=2,
        fontsize="x-small",
    )
    mg.line_plot(
        _from(unemployment, start),
        ax=ax,
        color="black",
        width=1.0,
        zorder=1,
        annotate=True,
        rounding=2,
        fontsize="x-small",
    )
    mg.finalise_plot(
        ax,
        title="Australian u*: two models, six specifications",
        ylabel="Per cent",
        axvspan=_UNIDENTIFIED,
        legend={"loc": "best", "fontsize": "x-small"},
        lheader="All fitted to the same data: agreement is not independence",
        rfooter=_RFOOTER,
        lfooter="Australia. Posterior medians. ",
        show=False,
    )


def plot_spread(frame: pd.DataFrame, start: str | None = "1993Q1") -> None:
    """Plot the range across specifications, which is the disagreement itself."""
    usable = _from(frame, start).dropna(how="any")
    if usable.empty or len(usable.columns) < MIN_MODELS_FOR_RANGE:
        print("  note: fewer than two specifications overlap; skipping the spread chart")
        return

    band = pd.DataFrame({"lower": usable.min(axis=1), "upper": usable.max(axis=1)})
    ax = mg.fill_between_plot(band, color="darkblue", alpha=0.18, label="Range across specifications")
    # The band's edges as thin lines, so their latest values are annotated. The
    # leading underscore keeps them out of the legend.
    mg.line_plot(
        band.rename(columns={"upper": "_top", "lower": "_bottom"})[["_top", "_bottom"]],
        ax=ax, color=["darkblue", "darkblue"], width=[0.25, 0.25], annotate=True, rounding=2,
    )
    # A mean across specifications is a value no specification produces. It
    # describes where they sit; it is not an estimate.
    mg.line_plot(
        usable.mean(axis=1).rename("Mean across specifications"),
        ax=ax, color=["darkblue"], width=2.5, annotate=True, rounding=2,
    )
    width = band["upper"] - band["lower"]
    mg.finalise_plot(
        ax,
        title="How much the specifications disagree about u*",
        ylabel="Per cent",
        axvspan=_UNIDENTIFIED,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"Across {len(usable.columns)} specifications, on their common quarters",
        rheader=f"Widest {width.max():.2f}pp in {width.idxmax()}; "
                f"latest {width.iloc[-1]:.2f}pp",
        rfooter=_RFOOTER,
        lfooter="Australia. The mean describes where they sit, not an estimate. ",
        show=False,
    )


def run_analyse(
    frame: pd.DataFrame,
    notes: dict[str, str],
    unemployment: pd.Series,
    start: str | None = "1993Q1",
    chart_dir: Path | str | None = None,
) -> None:
    """Write both charts and print the latest readings."""
    directory = Path(chart_dir) if chart_dir else CHART_DIR
    directory.mkdir(parents=True, exist_ok=True)
    mg.set_chart_dir(str(directory))
    mg.clear_chart_dir()

    plot_summary(frame, unemployment, start=start)
    plot_spread(frame, start=start)

    print("\nWhat each line is built from:")
    for label, note in notes.items():
        print(f"  {label:<52} {note}")

    print("\nLatest u*:")
    for column, series in frame.items():
        clean = series.dropna()
        if not clean.empty:
            print(f"  {column!s:<52} {clean.iloc[-1]:5.2f}   ({clean.index[-1]})")
    latest_u = unemployment.dropna()
    print(f"  {'Unemployment rate':<52} {latest_u.iloc[-1]:5.2f}   ({latest_u.index[-1]})")
    print(f"\nCharts written to: {directory}")
