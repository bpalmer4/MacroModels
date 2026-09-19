"""Charts and table for the y* specification comparison."""

from pathlib import Path

import mgplot as mg
import pandas as pd

from src.models.ystar_summary.sources import Loaded, common_quarters

CHART_DIR = Path("charts") / "YStar-summary"

_MODEL = "Australia. y* summary"
_RFOOTER = "Built using: ABS 1364.0.15.003, 5206.0, 6401.0, 6457.0"


def _lfooter(loaded: list[Loaded], extra: str = "") -> str:
    """Return the left footer, counting the specifications actually charted."""
    return f"{_MODEL}, {len(loaded)} specifications. {extra}"


def _frame(loaded: list[Loaded], attr: str) -> pd.DataFrame:
    """Collect one series from every specification into a single frame."""
    return pd.DataFrame({item.source.label: getattr(item, attr) for item in loaded})


def _plot(loaded: list[Loaded], attr: str, *, title: str, ylabel: str, y0: bool = False) -> None:
    """Draw one comparison chart: every specification's version of one series."""
    mg.line_plot_finalise(
        _frame(loaded, attr),
        title=title,
        ylabel=ylabel,
        color=[item.source.colour for item in loaded],
        style=[item.source.style for item in loaded],
        width=[2.0] * len(loaded),
        annotate=True,
        rounding=1,
        y0=y0,
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter=_lfooter(loaded, "Posterior medians. "),
        rfooter=_RFOOTER,
        show=False,
    )


def _plot_spread(loaded: list[Loaded], attr: str, *, title: str, ylabel: str) -> None:
    """Draw how far apart the specifications sit, quarter by quarter."""
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
        lfooter=_lfooter(loaded, "Highest less lowest, on medians. "),
        rfooter=_RFOOTER,
        show=False,
    )


def table(loaded: list[Loaded]) -> pd.DataFrame:
    """Return the headline numbers, one row per specification.

    The fit column is restricted to the quarters every specification's GDP
    likelihood covers, so the rows are summed over the same observations.
    """
    shared = common_quarters(loaded)
    scores = {item.source.label: float(item.elpd_by_quarter.loc[shared].sum()) for item in loaded}
    best = max(scores.values())

    rows = {}
    for item in loaded:
        growth = item.potential_growth
        rows[item.source.label] = {
            "elpd diff": scores[item.source.label] - best,
            "bad k": item.pareto_bad,
            "R-hat": item.max_rhat,
            "min ESS": item.min_ess,
            "div": item.divergences,
            "g* latest": float(growth.iloc[-1]),
            "g* 1990": float(growth.loc["1990Q1":"1992Q4"].mean()),
            "g* 2015-19": float(growth.loc["2015Q1":"2019Q4"].mean()),
            "gap 1992Q4": float(item.output_gap.loc[pd.Period("1992Q4")]),
            "gap latest": float(item.output_gap.iloc[-1]),
            "gap sd": float(item.output_gap.std()),
        }
    return pd.DataFrame(rows).T.sort_values("elpd diff", ascending=False)


def print_table(loaded: list[Loaded]) -> None:
    """Print the headline comparison."""
    shared = common_quarters(loaded)
    print("\n" + "=" * 78)
    print(f"{len(loaded)} SPECIFICATIONS OF THE y* MODEL, FROM {shared.min()}")
    print("=" * 78)
    print(table(loaded).round(2).to_string())
    print(
        f"\n  elpd diff scores GDP only, over the {len(shared)} quarters every specification\n"
        "  fitted, because GDP is the one series all five observe. It therefore favours a\n"
        "  specification that spends everything on GDP: `inflation` and `target` observe\n"
        "  nothing else, while `labour` answers for hours and participation with the same\n"
        "  trends and `production` for four factor series. One input, not a ranking.\n"
        "\n  R-hat, min ESS and div are the gate: a specification that did not sample is\n"
        "  out however well it fits.\n"
        "\n  The gap is log GDP less potential in every row, not the inflation-defined\n"
        "  series, so the column compares like with like.",
    )


def run_analysis(loaded: list[Loaded], chart_dir: Path | str | None = None) -> None:
    """Write every chart and print the table."""
    print_table(loaded)

    chart_dir = Path(chart_dir) if chart_dir is not None else CHART_DIR
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    _plot(
        loaded, "potential_growth",
        title="Potential growth by specification",
        ylabel="Year-ended, per cent",
    )
    _plot(
        loaded, "output_gap",
        title="The output gap by specification",
        ylabel="Per cent of potential",
        y0=True,
    )
    _plot(
        loaded, "potential",
        title="Potential output by specification",
        ylabel="Log level x 100",
    )
    _plot_spread(
        loaded, "potential_growth",
        title="How much the specification matters for potential growth",
        ylabel="Percentage points",
    )
    _plot_spread(
        loaded, "output_gap",
        title="How much the specification matters for the output gap",
        ylabel="Percentage points",
    )
    print(f"Charts written to: {chart_dir}")
