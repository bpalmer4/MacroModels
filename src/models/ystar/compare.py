"""Compare saved ystar runs on one set of axes.

The model's headline numbers are conditional on assumptions that are imposed
rather than estimated: the inflation anchor, the trend/cycle variance ratios,
and the form of the Phillips curve. `sigma_sweep.py` measures that
conditionality as a table. This draws it.

The comparison that matters most is the Phillips curve's, because that is where
the identification comes from. Three runs answer it:

- `ystar`          quarterly (non-overlapping) inflation, no supply term
- `ystar_supply`   the same, plus demeaned import price growth
- `ystar_piannual` the original four-quarter (overlapping) inflation

If the three agree, inflation is not carrying the answer on its own. If they do
not, the spread is the honest uncertainty, and it belongs with the headline
rather than in an appendix.

Usage::

    uv run python -m src.models.ystar.compare
    uv run python -m src.models.ystar.compare --runs ystar ystar_supply
"""

import argparse
from pathlib import Path  # noqa: TC003 — used at runtime in function signatures

import mgplot as mg
import pandas as pd

from src.models.ystar.config import DEFAULT_OUTPUT_DIR
from src.models.ystar.results import DEFAULT_CHART_BASE, load_results

# Own directory, for the same reason as `realtime.CHART_DIR`: `run_analysis`
# clears YStar/ before writing.
CHART_DIR = DEFAULT_CHART_BASE / "YStar" / "Compare"

DEFAULT_RUNS: dict[str, str] = {
    "ystar": "Quarterly inflation",
    "ystar_supply": "+ import prices",
    "ystar_piannual": "Annual inflation (overlapping)",
}

# Distinguishable against each other and against a pale band fill.
_COLORS = ["darkorange", "darkgreen", "darkblue", "crimson", "purple"]


def collect(runs: dict[str, str], output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load each run and return the median path of each compared quantity.

    Runs may cover different samples, so the frames are aligned on the union of
    their indexes and left with NaN where a run has nothing to say.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    gathered: dict[str, dict[str, pd.Series]] = {
        "output_gap": {},
        "trend_growth": {},
        "potential_growth": {},
    }
    for prefix, label in runs.items():
        results = load_results(output_dir=output_dir, prefix=prefix)
        gathered["output_gap"][label] = results.output_gap_median()
        gathered["potential_growth"][label] = results.potential_growth_median()
        # Keyed off the state's presence, not the spec name: `target` carries
        # the same g state as `core`, and a spec-name test silently dropped it.
        if "trend_growth" in results.trace.posterior:
            gathered["trend_growth"][label] = results.trend_growth_posterior().median(axis=1)

    return {name: pd.DataFrame(columns) for name, columns in gathered.items() if columns}


def plot_comparison(
    frame: pd.DataFrame,
    title: str,
    ylabel: str,
    *,
    plot_from: str | None = None,
    tag: str = "full",
    note: str = "Same model, different Phillips curve. ",
    sources: str = "Source: ABS 5206.0, 6401.0, 6457.0",
) -> None:
    """Draw one comparison chart. `tag` keeps the two windows in separate files."""
    data = frame if plot_from is None else frame.loc[frame.index >= pd.Period(plot_from, "Q")]

    mg.line_plot_finalise(
        data,
        tag=tag,
        color=_COLORS[: data.shape[1]],
        width=2,
        annotate=True,
        rounding=1,
        title=title,
        ylabel=ylabel,
        legend={"loc": "best", "fontsize": "small"},
        y0=True,
        rfooter=sources,
        lfooter=f"Australia. {note}",
        show=False,
    )


def run_comparison(
    runs: dict[str, str] | None = None,
    output_dir: Path | None = None,
    chart_dir: Path | str = CHART_DIR,
    *,
    note: str = "Same model, different Phillips curve. ",
    by: str = "Phillips curve specification",
    sources: str = "Source: ABS 5206.0, 6401.0, 6457.0",
) -> dict[str, pd.DataFrame]:
    """Load the runs, print the endpoint spread, and write the charts."""
    if runs is None:
        runs = DEFAULT_RUNS

    frames = collect(runs, output_dir)

    titles = {
        "output_gap": (f"Output gap by {by}", "Per cent of potential"),
        "trend_growth": (f"Trend potential growth by {by}", "Annualised %"),
        "potential_growth": (f"Potential growth by {by}", "Year-ended %"),
    }

    print(f"\n{'=' * 70}\nENDPOINT SPREAD ACROSS SPECIFICATIONS\n{'=' * 70}")
    for name, frame in frames.items():
        last = frame.dropna(how="all").index[-1]
        row = frame.loc[last]
        print(f"\n{name}  ({last})")
        for label, value in row.items():
            print(f"  {label:<32} {value:6.2f}")
        print(f"  {'spread':<32} {row.max() - row.min():6.2f}")

    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()
    for name, frame in frames.items():
        title, ylabel = titles[name]
        plot_comparison(frame, title, ylabel, tag="full", note=note, sources=sources)
        plot_comparison(
            frame, title, ylabel, plot_from="2015Q1", tag="recent", note=note, sources=sources,
        )

    print(f"\nCharts written to: {chart_dir}")
    return frames


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compare saved ystar runs")
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Output prefixes to compare (default: the three Phillips curve variants)",
    )
    args = parser.parse_args()

    runs = None if args.runs is None else {p: DEFAULT_RUNS.get(p, p) for p in args.runs}
    run_comparison(runs=runs)


if __name__ == "__main__":
    main()
