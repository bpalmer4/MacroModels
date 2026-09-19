"""Chart and table for the u* specification sweep."""

from pathlib import Path

import mgplot as mg
import pandas as pd

from src.data.inflation import get_trimmed_mean_qrtly
from src.models.ustar.analyse import UNIDENTIFIED_WINDOW
from src.models.ustar_summary.sources import Loaded, without_okun

CHART_DIR = Path("charts") / "UStar-summary"

# The RBA's band, and the test built on it: over the quarters where quarterly
# annualised trimmed mean inflation sat outside 2-3%, does the sign of the
# unemployment gap point the right way? It is not independent evidence, since
# every specification is fitted to the same inflation series, but it does
# separate them where the in-sample fit statistics do not.
_HIGH, _LOW = 3.0, 2.0

# Names the source of the charts. These are the summary's own, so they say
# u* summary; the individual runs' charts say ustar.
_LFOOTER = "Australia. u* summary, three specifications. "

# The window `ustar` marks on its own charts, read from that model rather than
# restated here so the two cannot drift apart. It matters more on this chart
# than on any single model's: the specifications differ most inside it, and
# the reason is that the data place u* least well there.
_UNIDENTIFIED = {
    "xmin": pd.Period(UNIDENTIFIED_WINDOW[0], freq="Q"),
    "xmax": pd.Period(UNIDENTIFIED_WINDOW[1], freq="Q"),
    "color": "darkorange",
    "alpha": 0.10,
    "zorder": 0,
    "label": f"u* not well identified, {UNIDENTIFIED_WINDOW[0]}-{UNIDENTIFIED_WINDOW[1]}",
}
_RFOOTER = "Built using: ABS 1364.0.15.003, 5206.0, 6401.0, 6457.0; NY Fed"


def _annualised_trimmed(index: pd.PeriodIndex) -> pd.Series:
    """Return quarterly trimmed mean inflation, annualised, on `index`."""
    quarterly = get_trimmed_mean_qrtly().data.astype(float)
    return (((1 + quarterly / 100) ** 4 - 1) * 100).reindex(index)


def score(loaded: Loaded) -> dict[str, float]:
    """Return the diagnostics the table reports for one specification."""
    ustar, u = loaded.ustar, loaded.unemployment
    inflation = _annualised_trimmed(pd.PeriodIndex(ustar.index))
    above, below = inflation > _HIGH, inflation < _LOW
    gap = u - ustar
    hits = int((above & (gap < 0)).sum()) + int((below & (gap > 0)).sum())
    outside = int((above | below).sum())
    deviation = (loaded.implied - ustar).dropna()
    tail = ustar.loc["2015Q1":]
    return {
        "band test": hits / outside,
        "1993-98": float(ustar.loc["1993Q1":"1998Q4"].mean()),
        "post-2015 slope": float(tail.iloc[-1] - tail.iloc[0]),
        "latest": float(ustar.iloc[-1]),
        "90% band": loaded.band,
        "bias vs implied": float(deviation.mean()),
        "sd(du*)": float(ustar.diff().std()),
    }


def table(loaded: list[Loaded]) -> pd.DataFrame:
    """Return one row per specification."""
    return pd.DataFrame(
        [score(item) for item in loaded],
        index=[item.source.label for item in loaded],
    )


def plot_ustar(loaded: list[Loaded]) -> None:
    """Every specification's u* on one axis, against the unemployment rate.

    Colour is the knot count and dashing is the gap-form Okun equation, so the two
    dimensions can be read separately. The unemployment rate is drawn in black
    behind them because the question every one of these lines is answering is
    how far u* is allowed to depart from it.
    """
    frame = pd.DataFrame({item.source.label: item.ustar for item in loaded})
    frame["Unemployment rate"] = loaded[0].unemployment

    ax = mg.line_plot(
        frame,
        color=[item.source.colour for item in loaded] + ["black"],
        style=[item.source.style for item in loaded] + ["-"],
        width=[1.8] * len(loaded) + [1.0],
        annotate=True,
        rounding=2,
        fontsize="x-small",
    )
    mg.finalise_plot(
        ax,
        title="u*: one model, three specifications",
        axvspan=_UNIDENTIFIED,
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "x-small"},
        lheader="Colour is the knot count; dashed carries the gap-form Okun equation",
        lfooter=_LFOOTER,
        rfooter=_RFOOTER,
        show=False,
    )


def plot_gap(loaded: list[Loaded]) -> None:
    """Chart the unemployment gap each specification implies."""
    frame = pd.DataFrame({item.source.label: item.unemployment - item.ustar for item in loaded})
    ax = mg.line_plot(
        frame,
        color=[item.source.colour for item in loaded],
        style=[item.source.style for item in loaded],
        width=[1.8] * len(loaded),
        annotate=True,
        rounding=2,
        fontsize="x-small",
    )
    mg.finalise_plot(
        ax,
        title="The unemployment gap, by specification",
        axvspan=_UNIDENTIFIED,
        ylabel="Percentage points",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter=_LFOOTER,
        rfooter=_RFOOTER,
        show=False,
    )


def plot_range(loaded: list[Loaded]) -> None:
    """Shade the range across the specifications, with the mean through it.

    The band only. The individual lines belong on the levels chart; drawing
    them here as well left five series over a band that is itself built from
    them, and the band is the subject.

    2.87pp at 1993Q1 falling to 0.03pp today is a fair reflection of a period
    where pinning u* is harder, and it agrees with the shaded window and with
    the bias against the implied series. Three routes, same seven years.

    It is not a symmetric error band: the three differ in a structured way,
    whether Okun is in, rather than randomly. Among the three without Okun the
    spread is 0.21pp, against a mean 90% band within any one of them of 0.65,
    so after the early sample how u* may move matters less than the estimation
    uncertainty does.

    MEAN, not median. With three series the median is whichever specification
    sits in the middle that quarter, so it switches identity wherever the
    lines cross and picks up kinks that say nothing about u*.

    THE MEAN IS NOT AN ESTIMATE, and here it is weaker than the same line on
    `rstar_summary`. There the average is across models built on different
    data and different identifying assumptions. These share a sample, a
    Phillips curve and an expectations series, so their agreement is close to
    arithmetic. It describes where the specifications sit.
    """
    frame = pd.DataFrame({item.source.label: item.ustar for item in loaded})
    band = pd.DataFrame({"lower": frame.min(axis=1), "upper": frame.max(axis=1)})
    width = band["upper"] - band["lower"]
    ax = mg.fill_between_plot(band, color="darkorange", alpha=0.18,
                              label="Range across specifications")
    mg.line_plot(frame.mean(axis=1).rename("Mean across specifications"),
                 ax=ax, color=["darkorange"], width=2.5, annotate=True, rounding=2)
    mg.line_plot(loaded[0].unemployment.rename("Unemployment rate"),
                 ax=ax, color=["black"], width=1.0, annotate=True, rounding=2)
    mg.finalise_plot(
        ax,
        title="How much the specification matters for u*",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "x-small"},
        lheader="Widest where u* is least identified; see the levels chart for the lines",
        rheader=f"Widest {width.max():.2f}pp in {width.idxmax()}; latest {width.iloc[-1]:.2f}pp",
        lfooter="Australia. u* summary. The mean describes where the specifications sit. ",
        rfooter=_RFOOTER,
        show=False,
    )


def plot_spread(loaded: list[Loaded]) -> None:
    """How far apart the six are, quarter by quarter.

    The point of the summary: where the specifications agree, the answer is a
    property of the data; where they disagree, it is a property of the
    assumptions. This is that distinction as one series.
    """
    frame = pd.DataFrame({item.source.label: item.ustar for item in loaded})
    spread = (frame.max(axis=1) - frame.min(axis=1)).rename("Widest minus narrowest")
    ax = mg.line_plot(spread, color=["darkorange"], width=2.2, annotate=True, rounding=2)
    mg.finalise_plot(
        ax,
        title="How much the specification matters",
        axvspan=_UNIDENTIFIED,
        ylabel="Percentage points",
        lheader=f"Mean {spread.mean():.2f}pp, worst {spread.max():.2f}pp at {spread.idxmax()}",
        lfooter=_LFOOTER,
        rfooter=_RFOOTER,
        show=False,
    )


def run_analysis(loaded: list[Loaded], chart_dir: Path = CHART_DIR) -> pd.DataFrame:
    """Draw the charts, print the table, and return it."""
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    summary = table(loaded)
    print("\nThree specifications of one model")
    print("-" * 100)
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    frame = pd.DataFrame({item.source.label: item.ustar for item in loaded})
    spread = frame.max(axis=1) - frame.min(axis=1)
    core = pd.DataFrame({item.source.label: item.ustar for item in without_okun(loaded)})
    core_spread = core.max(axis=1) - core.min(axis=1)
    print(f"\nSpread, all {len(loaded)}: mean {spread.mean():.2f}pp, "
          f"worst {spread.max():.2f}pp at {spread.idxmax()}, latest {spread.iloc[-1]:.2f}pp")
    print(f"Spread, without Okun: mean {core_spread.mean():.2f}pp, "
          f"worst {core_spread.max():.2f}pp at {core_spread.idxmax()}, "
          f"latest {core_spread.iloc[-1]:.2f}pp")
    print(f"Latest u* ranges {frame.iloc[-1].min():.2f} to {frame.iloc[-1].max():.2f} "
          f"against unemployment of {loaded[0].unemployment.iloc[-1]:.2f}")

    plot_ustar(loaded)
    plot_gap(loaded)
    plot_range(loaded)
    plot_spread(loaded)
    print(f"\nCharts written to: {chart_dir}")
    return summary
