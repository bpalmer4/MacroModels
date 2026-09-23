"""Charts and table for `--compare`: the comparison specifications side by side.

MODEL_NOTES, "Comparing specifications", says how to read them.
"""

from pathlib import Path

import mgplot as mg
import pandas as pd

from src.models.ystar_ustar.analyse import UNIDENTIFIED_WINDOW
from src.models.ystar_ustar.compare import Loaded
from src.paths import CHARTS

CHART_DIR = CHARTS / "YStarUStar-compare"

_MODEL = "Australia. y*/u*: comparing specifications"
_RFOOTER = "Built using: ABS 1364.0.15.003, 5206.0, 6401.0, 6457.0; NY Fed"

# The window the model marks on its own charts, read from it so the two cannot
# drift apart. The specifications differ most inside it, because that is where
# the data place the stars least well.
_UNIDENTIFIED = {
    "xmin": pd.Period(UNIDENTIFIED_WINDOW[0], freq="Q"),
    "xmax": pd.Period(UNIDENTIFIED_WINDOW[1], freq="Q"),
    "color": "darkorange",
    "alpha": 0.10,
    "zorder": 0,
    "label": f"Stars not well identified, {UNIDENTIFIED_WINDOW[0]}-{UNIDENTIFIED_WINDOW[1]}",
}


def _lfooter(loaded: list[Loaded], extra: str = "") -> str:
    """Return the left footer, counting the specifications actually charted."""
    return f"{_MODEL}, {len(loaded)} of them. {extra}"


def _unidentified(loaded: list[Loaded]) -> dict | None:
    """Return the shaded window, or nothing when the sample does not open on it.

    The window was measured on a sample starting at its first quarter, where
    the stars are placed by structure rather than by the data. On a sample that
    begins earlier the disinflation is inside the sample and the shading would
    mark a period as doubtful on evidence gathered about a different run.
    """
    if not loaded or loaded[0].ustar.index[0] != _UNIDENTIFIED["xmin"]:
        return None
    return _UNIDENTIFIED


def _frame(loaded: list[Loaded], attr: str) -> pd.DataFrame:
    """Collect one series from every specification into a single frame."""
    return pd.DataFrame({item.spec.label: getattr(item, attr) for item in loaded})


def _styling(loaded: list[Loaded]) -> dict[str, list]:
    """Return the per-series colour, dash and width, in the frame's column order."""
    return {
        "color": [item.spec.colour for item in loaded],
        "style": [item.spec.style for item in loaded],
        "width": [2.0] * len(loaded),
    }


def _plot(
    loaded: list[Loaded],
    attr: str,
    *,
    title: str,
    ylabel: str,
    extra: pd.Series | None = None,
    extra_label: str = "",
    y0: bool = False,
) -> None:
    """Draw one comparison chart: every specification's version of one series."""
    frame = _frame(loaded, attr)
    style = _styling(loaded)
    if extra is not None:
        frame[extra_label] = extra
        style["color"] = [*style["color"], "black"]
        style["style"] = [*style["style"], ":"]
        style["width"] = [*style["width"], 1.5]

    mg.line_plot_finalise(
        frame,
        title=title,
        ylabel=ylabel,
        color=style["color"],
        style=style["style"],
        width=style["width"],
        annotate=True,
        rounding=1,
        y0=y0,
        legend={"loc": "best", "fontsize": "x-small"},
        axvspan=_unidentified(loaded),
        lfooter=_lfooter(loaded, "Posterior medians. "),
        rfooter=_RFOOTER,
        show=False,
    )


def _plot_spread(loaded: list[Loaded], attr: str, *, title: str, ylabel: str) -> None:
    """Draw how far apart the specifications sit, quarter by quarter.

    The range, not a standard deviation: the lines differ in a structured way
    rather than randomly, so a dispersion statistic would imply a distribution
    around a centre that does not exist.
    """
    frame = _frame(loaded, attr)
    mg.line_plot_finalise(
        (frame.max(axis=1) - frame.min(axis=1)).rename("Range across specifications"),
        title=title,
        ylabel=ylabel,
        color=["firebrick"],
        width=2,
        annotate=True,
        rounding=2,
        legend={"loc": "best", "fontsize": "x-small"},
        axvspan=_unidentified(loaded),
        lfooter=_lfooter(loaded, "Highest less lowest, on medians. "),
        rfooter=_RFOOTER,
        show=False,
    )


def _residual_stats(item: Loaded) -> dict[str, float]:
    """Compare fitted u* with the u* the Phillips curve implies, quarter by quarter.

    What the residual catches is BIAS, a fitted path sitting systematically to
    one side of where inflation puts u*. Its sd is expected: the fitted path is
    a smoothed version of the implied one.
    """
    deviation = (item.implied_ustar - item.ustar).dropna()
    early = deviation.loc[:"1999Q4"]
    return {
        "resid mean": float(deviation.mean()),
        "resid sd": float(deviation.std()),
        "resid 93-99": float(early.mean()) if len(early) else float("nan"),
    }


def table(loaded: list[Loaded]) -> pd.DataFrame:
    """Return the headline numbers, one row per specification."""
    best = max(item.elpd for item in loaded)
    rows = {}
    for item in loaded:
        tail = item.ustar.loc["2015Q1":]
        rows[item.spec.label] = {
            "elpd diff": item.elpd - best,
            "elpd se": item.elpd_se,
            "bad k": item.pareto_bad,
            "R-hat": item.max_rhat,
            "min ESS": item.min_ess,
            "div": item.divergences,
            **_residual_stats(item),
            "u* latest": float(item.ustar.iloc[-1]),
            "u* 1993Q1": float(item.ustar.iloc[0]),
            "u* post-2015": float(tail.iloc[-1] - tail.iloc[0]),
            "u* band": item.ustar_band,
            "gap 1993Q1": float(item.output_gap.iloc[0]),
            "gap latest": float(item.output_gap.iloc[-1]),
            "gap sd": float(item.output_gap.std()),
            "g* latest": float(item.potential_growth.iloc[-1]),
        }
    return pd.DataFrame(rows).T.sort_values("elpd diff", ascending=False)


def print_table(loaded: list[Loaded]) -> None:
    """Print the headline comparison."""
    print("\n" + "=" * 78)
    print(f"{len(loaded)} SPECIFICATIONS OF THE JOINT MODEL")
    print("=" * 78)
    print(table(loaded).round(2).to_string())
    print(
        "\n  elpd diff is leave-one-out predictive accuracy against the best row, scored\n"
        "  ONLY on the two equations every specification observes, unemployment and\n"
        "  inflation. The GDP equation is excluded because under the identity gap it is\n"
        "  a definition carrying no likelihood. Read a difference against elpd se.\n"
        "  bad k counts observations where the LOO estimate is unreliable.\n"
        "\n  R-hat, min ESS and div are the gate, not the score: a specification that did\n"
        "  not sample is out however well it fits.\n"
        "\n  resid is the Phillips-implied u* less the fitted u*. Its MEAN is the test,\n"
        "  not its sd: a smoothed path should scatter, but it should not sit to one side.\n"
        "\n  u* band is the mean 90% interval width, estimation uncertainty within a run.\n"
        "  The spread across rows is specification uncertainty. The two are not comparable.",
    )


def run_comparison(loaded: list[Loaded], chart_dir: Path | str | None = None) -> None:
    """Write every chart and print the table."""
    print_table(loaded)

    chart_dir = Path(chart_dir) if chart_dir is not None else CHART_DIR
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    _plot(loaded, "ustar", title="u* by specification", ylabel="Per cent",
          extra=loaded[0].unemployment, extra_label="Unemployment rate")
    _plot(loaded, "output_gap", title="The output gap by specification",
          ylabel="Per cent of potential", y0=True)
    _plot(loaded, "potential_growth", title="Potential growth by specification",
          ylabel="Year-ended, per cent")
    _plot(loaded, "potential", title="Potential output by specification",
          ylabel="Log level x 100")
    _plot_spread(loaded, "ustar", title="How much the specification matters for u*",
                 ylabel="Percentage points")
    _plot_spread(loaded, "output_gap", title="How much the specification matters for the output gap",
                 ylabel="Percentage points")
    print(f"Charts written to: {chart_dir}")
