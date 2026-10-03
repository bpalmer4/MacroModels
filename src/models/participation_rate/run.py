"""Command-line entry point: participation rate after RBA rate moves."""

import argparse

import mgplot as mg
import pandas as pd

from src.data.labour_force_age import get_labour_force_status_by_age
from src.models.participation_rate import CHART_DIR
from src.models.participation_rate.analysis import (
    LP_HORIZONS,
    age_comparison,
    cycle_table,
    dose_test,
    era_split,
    local_projection,
    pre_trend,
    raw_means,
)
from src.models.participation_rate.charts import age_charts, latest_cycle_by_sex_chart, make_charts
from src.models.participation_rate.data import build_frame, sample_months

DECIMALS = 3
LP_COLUMNS = ["hike_b", "hike_se", "hike_t", "cut_b", "cut_se", "cut_t", "n"]


def _show(title: str, table: pd.DataFrame | pd.Series) -> None:
    print(f"\n=== {title} ===")
    print(table.round(DECIMALS).to_string())


def main() -> None:
    """Print the tests and write the charts."""
    parser = argparse.ArgumentParser(description="Does participation rise after RBA rate rises?")
    parser.add_argument("--chart-dir", default=None, help="Override the chart directory")
    args = parser.parse_args()

    d = build_frame()
    months = sample_months(d)
    print(f"Sample {months[0]} to {months[-1]}")
    for ev in ["hike", "first_hike", "cut", "first_cut"]:
        print(f"  {ev}: {int(d.loc[months, ev].sum())} months")
    print("First hikes:", ", ".join(str(p) for p in months[d.loc[months, "first_hike"] == 1]))

    _show("Raw mean PR change, t-1 to t+h (pp)", raw_means(d, months, "pr"))

    projections = {
        "every decision": ("pr", ("hike", "cut"), False),
        "first move of a cycle": ("pr", ("first_hike", "first_cut"), False),
        "every decision, given the unemployment path": ("pr", ("hike", "cut"), True),
        "first move of a cycle, given the unemployment path": ("pr", ("first_hike", "first_cut"), True),
        "females, every decision": ("pr_f", ("hike", "cut"), False),
        "males, every decision": ("pr_m", ("hike", "cut"), False),
    }
    results = {}
    for label, (col, events, ur_path) in projections.items():
        results[label] = local_projection(d, months, col, events, ur_path=ur_path)
        _show(f"Local projection: {label}", results[label].loc[LP_HORIZONS, LP_COLUMNS])

    cycles = cycle_table(d, months)
    _show("First hikes, cycle by cycle (pp)", cycles)
    _show("First hikes, mean of completed cycles (pp)", cycles.mean())
    _show("Dose: PR change per 1pp of rate rises / falls", dose_test(d, months))
    _show("Era split: hike effect early vs late", era_split(d, months))
    _show("Pre-trend: PR change in the year before a first move", pre_trend(d, months, "pr"))

    # The unemployment rate on the same events: the labour market cycle the
    # participation response is tested against. No unemployment-path variant:
    # it would put the outcome on both sides.
    print("\n\n######## Unemployment rate ########")
    _show("Raw mean UR change, t-1 to t+h (pp)", raw_means(d, months, "ur"))
    ur_results = {}
    for label, events in [("every decision", ("hike", "cut")), ("first move of a cycle", ("first_hike", "first_cut"))]:
        ur_results[label] = local_projection(d, months, "ur", events, ur_path=False)
        _show(f"Local projection, UR: {label}", ur_results[label].loc[LP_HORIZONS, LP_COLUMNS])
    _show("Pre-trend: UR change in the year before a first move", pre_trend(d, months, "ur"))

    chart_dir = args.chart_dir or str(CHART_DIR)
    mg.set_chart_dir(chart_dir)
    mg.clear_chart_dir()
    make_charts(d, months, results["first move of a cycle"], "pr", "Participation")
    make_charts(d, months, ur_results["first move of a cycle"], "ur", "Unemployment")
    latest_cycle_by_sex_chart(d, months)

    # The latest cycle by age: from the month before its first hike to the
    # latest month of the age data.
    status = get_labour_force_status_by_age()
    base = months[d.loc[months, "first_hike"] == 1][-1] - 1
    end = status.index.get_level_values("month").max()
    diff, contribution, normal_years = age_comparison(status, base, end)
    _show(f"Participation change {base} to {end} minus normal-year change, by age (pp)", diff)
    _show("Contribution to each sex's change: change x population share (pp)", contribution)
    age_charts(diff, contribution, base, end, normal_years)
    print(f"\nCharts written to: {chart_dir}")


if __name__ == "__main__":
    main()
