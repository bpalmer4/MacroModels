"""Charts and table for `--compare`: the comparison specifications side by side.

MODEL_NOTES, "Comparing specifications", says how to read them.
"""

from pathlib import Path

import mgplot as mg
import pandas as pd

from src.data.inflation import get_trimmed_mean_qrtly
from src.models.ustar.analyse import UNIDENTIFIED_WINDOW
from src.models.ustar.compare import Loaded, without_okun
from src.paths import CHARTS

CHART_DIR = CHARTS / "UStar-compare"

# The RBA's band, and the test built on it: over the quarters where quarterly
# annualised trimmed mean inflation sat outside 2-3%, does the sign of the
# unemployment gap point the right way? Not independent evidence, since every
# specification is fitted to the same inflation series, but it separates them
# where the in-sample fit statistics do not.
_HIGH, _LOW = 3.0, 2.0

_LFOOTER = "Australia. u*: comparing three specifications. "

# The window the model's own charts mark, read from there so the two cannot
# drift apart. It matters most here: the specifications differ most inside it,
# because the data place u* least well there.
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
        index=[item.spec.label for item in loaded],
    )


def plot_ustar(loaded: list[Loaded]) -> None:
    """Every specification's u* on one axis, against the unemployment rate.

    Colour is the knot count and dashing is the gap-form Okun equation, so the
    two dimensions read separately. The unemployment rate is drawn behind them
    because every line is answering how far u* may depart from it.
    """
    frame = pd.DataFrame({item.spec.label: item.ustar for item in loaded})
    frame["Unemployment rate"] = loaded[0].unemployment

    ax = mg.line_plot(
        frame,
        color=[item.spec.colour for item in loaded] + ["black"],
        style=[item.spec.style for item in loaded] + ["-"],
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
    frame = pd.DataFrame({item.spec.label: item.unemployment - item.ustar for item in loaded})
    ax = mg.line_plot(
        frame,
        color=[item.spec.colour for item in loaded],
        style=[item.spec.style for item in loaded],
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

    The band only: the individual lines are on the levels chart. The mean, not
    the median, because with three series the median switches identity
    wherever the lines cross. The mean describes where the specifications sit;
    it is not an estimate.
    """
    frame = pd.DataFrame({item.spec.label: item.ustar for item in loaded})
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
        lfooter="Australia. u* comparison. The mean describes where the specifications sit. ",
        rfooter=_RFOOTER,
        show=False,
    )


def plot_spread(loaded: list[Loaded]) -> None:
    """Chart how far apart the specifications are, quarter by quarter.

    Where they agree, the answer is a property of the data; where they
    disagree, of the assumptions. This is that distinction as one series.
    """
    frame = pd.DataFrame({item.spec.label: item.ustar for item in loaded})
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


def run_comparison(loaded: list[Loaded], chart_dir: Path = CHART_DIR) -> pd.DataFrame:
    """Draw the charts, print the table, and return it."""
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    summary = table(loaded)
    print("\nThree specifications of one model")
    print("-" * 100)
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    frame = pd.DataFrame({item.spec.label: item.ustar for item in loaded})
    spread = frame.max(axis=1) - frame.min(axis=1)
    core = pd.DataFrame({item.spec.label: item.ustar for item in without_okun(loaded)})
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
