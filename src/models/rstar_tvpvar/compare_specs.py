"""Chart several saved TVP-VAR runs against each other.

WHAT THIS IS FOR. The model's answer moves with the sample window, and the
question "is COVID the problem, or 1994-95, or the start date" can only be
settled by estimating the variants and putting them side by side. This module
does the side-by-side. It estimates nothing: every run must already be saved
under its own prefix.

IT EXISTS SO THE COMPARISON IS REPRODUCIBLE. `rstar_invert`'s notes carry a
table from a parameterisation that was never implemented, so its central claim
cannot be regenerated from the repo. That is the failure this module avoids:
whatever the charts below show, `uv run python -m src.models.rstar_tvpvar.compare_specs`
redraws it from the saved traces.

THE SECOND CHART IS THE IMPORTANT ONE. r* paths differ by less than the
diagnostics do, and the spectral radius is what decides whether any of them
mean anything: near one, the H-quarter projection still carries most of today's
state, so r* is part nowcast rather than an estimate of neutral.

    uv run python -m src.models.rstar_tvpvar.compare_specs
    uv run python -m src.models.rstar_tvpvar.compare_specs --specs a=1993 b=1999
"""

import argparse
from pathlib import Path

import mgplot as mg
import pandas as pd

from src.models.rstar_tvpvar.results import DEFAULT_CHART_BASE, load_results

CHART_DIR = DEFAULT_CHART_BASE / "RStarTVPVAR_specs"

# The four runs made on 2026-09-17 to test whether the sample window is the
# lever. Order matters: it sets the colours, and the two 1993 runs should read
# as a pair against the two 1999 runs.
DEFAULT_SPECS: tuple[tuple[str, str], ...] = (
    ("rstar_tvpvar", "1993 start, COVID kept (shipped)"),
    ("rstar_tvpvar_nocovid", "1993 start, COVID excluded"),
    ("rstar_tvpvar_99", "1999 start, COVID kept"),
    ("rstar_tvpvar_99nc", "1999 start, COVID excluded"),
)

# Paired by start date rather than by suite convention: the two 1993 runs are
# blues and the two 1999 runs are warm, so the eye groups them the way the
# finding does. Solid is COVID kept, dashed is COVID excluded.
_COLOURS = ["darkblue", "cornflowerblue", "darkred", "darkorange"]
_STYLES = ["-", "--", "-", "--"]

_RFOOTER = "Built using: ABS 5206.0, 6401.0; RBA F1"
_LFOOTER = "Australia. TVP-VAR, canonical spec, varying the sample only. "
# Above this the projection carries more of today's state than of the VAR's own
# resting point, which is the line between an estimate and a nowcast.
_NOWCAST_RADIUS = 0.97

# A comparison of one run is not a comparison.
_MIN_RUNS = 2


def _collect(specs: tuple[tuple[str, str], ...]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return r* paths, median spectral-radius paths, and the labels that loaded.

    A missing trace is skipped with a note rather than raised: the point of the
    module is comparing whatever has been run, and demanding all four would make
    it useless after a single prefix is cleaned up.
    """
    rstar: dict[str, pd.Series] = {}
    radius: dict[str, pd.Series] = {}
    found: list[str] = []
    for prefix, label in specs:
        try:
            results = load_results(prefix=prefix)
        except FileNotFoundError:
            print(f"  skipping {label}: no saved trace for prefix {prefix!r}")
            continue
        rstar[label] = results.rstar_median()
        radius[label] = results.spectral_radius().median(axis=1)
        found.append(label)
        stability = results.stability_report()
        print(f"  {label:<34} r* latest {rstar[label].iloc[-1]:5.2f}   "
              f"radius {stability['median spectral radius']:.3f}   "
              f"explosive {stability['share of draw-quarters explosive']:5.1%}")
    return pd.DataFrame(rstar), pd.DataFrame(radius), found


def plot_rstar(rstar: pd.DataFrame, found: list[str]) -> None:
    """Plot the r* paths. No bands: four overlapping bands would be unreadable."""
    n = len(found)
    ax = mg.line_plot(
        rstar,
        color=_COLOURS[:n],
        style=_STYLES[:n],
        width=[2.0] * n,
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="r-star across TVP-VAR sample choices",
        ylabel="Per cent, real",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        lheader="Bands omitted: they average 4.1 to 5.1pp and would cover the chart",
        rfooter=_RFOOTER,
        lfooter=_LFOOTER,
        show=False,
    )


def plot_radius(radius: pd.DataFrame, found: list[str]) -> None:
    """Plot the median spectral radius, which is what actually separates the runs."""
    n = len(found)
    ax = mg.line_plot(
        radius,
        color=_COLOURS[:n],
        style=_STYLES[:n],
        width=[2.0] * n,
        annotate=True,
        rounding=3,
    )
    mg.finalise_plot(
        ax,
        title="How close to a unit root each sample choice gets",
        ylabel="Median spectral radius",
        legend={"loc": "best", "fontsize": "x-small"},
        axhline=[
            {"y": 1.0, "color": "black", "linestyle": "-", "linewidth": 1.0},
            {"y": _NOWCAST_RADIUS, "color": "grey", "linestyle": ":", "linewidth": 1.0},
        ],
        lheader="Above 1.0 the projection runs away; near it, r* is mostly a nowcast",
        rfooter=_RFOOTER,
        lfooter=_LFOOTER,
        show=False,
    )


def run(specs: tuple[tuple[str, str], ...] = DEFAULT_SPECS, chart_dir: Path | None = None) -> None:
    """Load the saved runs and draw the two comparison charts."""
    print("Loading saved runs:")
    rstar, radius, found = _collect(specs)
    if len(found) < _MIN_RUNS:
        print(f"\nFewer than {_MIN_RUNS} runs available; nothing to compare.")
        return

    mg.set_chart_dir(str(chart_dir or CHART_DIR))
    mg.clear_chart_dir()
    plot_rstar(rstar, found)
    plot_radius(radius, found)

    common = rstar.dropna()
    print(f"\nOn the {len(common)} quarters all runs share:")
    print(f"  spread at the last common quarter  {common.iloc[-1].max() - common.iloc[-1].min():.2f}pp")
    print(f"  pairwise correlations:\n{common.corr().round(3).to_string()}")
    print(f"\nCharts written to: {chart_dir or CHART_DIR}")


def main() -> None:
    """Chart saved TVP-VAR runs against each other."""
    parser = argparse.ArgumentParser(description="Compare saved TVP-VAR runs")
    parser.add_argument(
        "--specs", nargs="*", default=None, metavar="PREFIX=LABEL",
        help="Runs to compare, as prefix=label pairs. Defaults to the four sample-window "
             "variants estimated on 2026-09-17",
    )
    args = parser.parse_args()
    specs = DEFAULT_SPECS
    if args.specs:
        specs = tuple(
            (item.split("=", 1)[0], item.split("=", 1)[1] if "=" in item else item)
            for item in args.specs
        )
    run(specs)


if __name__ == "__main__":
    main()
