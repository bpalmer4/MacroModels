"""Pseudo-real-time endpoint revisions.

The model estimates an I(2) trend, and the quantity people want from it is the
value at the right-hand endpoint. There is no future data at the endpoint to
distinguish temporary weakness in GDP from a level shock to potential from a
change in potential growth, so the endpoint estimate is the fragile one.
In-sample fit says nothing about that. This module measures it directly:
re-estimate on a truncated sample, keep the estimate of *today* that a user
would have had at that date, then ask how much later data moved it.

    revision(t, h) = estimate of x_t from the sample ending t+h
                   - estimate of x_t from the sample ending t

The classic reference for the exercise is Orphanides & van Norden (2002),
*The Unreliability of Output Gap Estimates in Real Time*, which found revisions
to US output gaps of the same order as the gaps themselves.

**This is pseudo-real time, not real time.** The data are the current vintage,
truncated. ABS back-revises GDP, and the LFS population series is rebased on
Census, so a genuine real-time exercise would need historical vintages the
project does not hold. What this measures is the filter's endpoint problem
alone, with data revision set aside. That is the larger of the two terms for a
trend/cycle model, but it is not the whole of it, and the numbers here are
therefore a lower bound on real-time unreliability.

Usage::

    uv run python -m src.models.ystar.realtime
    uv run python -m src.models.ystar.realtime --first 2005Q4 --step 4
"""

import argparse
import pickle
from dataclasses import replace
from pathlib import Path

import mgplot as mg
import numpy as np
import pandas as pd

from src.models.ystar.base import SamplerConfig, sample_model
from src.models.ystar.config import ModelConfig
from src.models.ystar.estimate import build_model
from src.models.ystar.observations import build_observations
from src.models.ystar.results import DEFAULT_CHART_BASE, PotentialResults

# Its own directory: `analyse.run_analysis` clears YStar/ before writing,
# so charts left there by a separate command would not survive the next run.
CHART_DIR = DEFAULT_CHART_BASE / "YStar" / "RealTime"

# Horizons reported, in quarters. Every one must be a multiple of the vintage
# step, or no pair of vintages is that far apart and the row is empty.
DEFAULT_HORIZONS = (4, 8, 12, 20)

# Quantities tracked. Each is a (label, accessor) pair returning a time-indexed
# posterior; only the median path is kept, since revisions are about the point
# estimate a user would have quoted.
_QUANTITIES = ("trend_growth", "potential_growth", "output_gap")


def _paths(config: ModelConfig, sampler_config: SamplerConfig) -> dict[str, pd.Series]:
    """Estimate once and return the median path of each tracked quantity.

    Sampling happens here rather than through `run_estimate` because a real-time
    exercise is dozens of runs and each full trace is tens of megabytes on disk.
    Only the medians are needed.
    """
    obs, obs_index, _, _ = build_observations(
        start=config.start, end=config.end, verbose=False,
        smooth_pop=config.smooth_pop, spec=config.spec,
        pi_basis=config.pi_basis, supply_control=config.supply_control,
    )
    model = build_model(obs, config=config, verbose=False, obs_index=obs_index)
    trace = sample_model(model, sampler_config)
    results = PotentialResults(trace=trace, obs=obs, obs_index=obs_index)

    return {
        "trend_growth": results.trend_growth_posterior().median(axis=1),
        "potential_growth": results.potential_growth_posterior().median(axis=1),
        "output_gap": results.output_gap_posterior().median(axis=1),
    }


def run_vintages(
    first: str = "2008Q4",
    step: int = 4,
    base_config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Re-estimate on samples ending at a grid of vintage dates.

    Args:
        first: Earliest vintage end-date.
        step: Quarters between vintages. Revision horizons must be multiples
            of this, so 4 gives the 4/8/12/20-quarter horizons directly.
        base_config: Config to truncate (defaults to ModelConfig()).
        sampler_config: Sampler settings (defaults to a shorter run).

    Returns:
        One DataFrame per tracked quantity, indexed by quarter, with one column
        per vintage. Column `v` holds what the model said about every quarter
        up to `v`, using data up to `v` only.

    """
    if base_config is None:
        base_config = ModelConfig()
    if base_config.spec != "core":
        # `trend_growth` is a core-only state, and the labour spec additionally
        # needs vintage-consistent population smoothing, which this does not do.
        raise ValueError(f"the real-time exercise is core-only, got spec={base_config.spec!r}")
    if sampler_config is None:
        sampler_config = SamplerConfig(draws=1_000, tune=1_000, chains=4, cores=4)

    # The full sample defines the last vintage and hence the "final" estimate
    # every revision is measured against.
    _, full_index, _, _ = build_observations(
        start=base_config.start, end=base_config.end, verbose=False,
        smooth_pop=base_config.smooth_pop, spec=base_config.spec,
        pi_basis=base_config.pi_basis, supply_control=base_config.supply_control,
    )
    last = full_index[-1]

    vintages = list(pd.period_range(pd.Period(first, "Q"), last, freq="Q")[::step])
    if vintages[-1] != last:
        vintages.append(last)

    # Columns are labelled with the vintage as a string, not as a Period: a
    # Period is not an accepted column key for `.loc`, and every lookup below
    # is column-then-row anyway.
    collected: dict[str, dict[str, pd.Series]] = {q: {} for q in _QUANTITIES}
    for i, vintage in enumerate(vintages, start=1):
        print(f"\n[{i}/{len(vintages)}] vintage {vintage} ...", flush=True)
        config = replace(base_config, end=str(vintage))
        for name, series in _paths(config, sampler_config).items():
            collected[name][str(vintage)] = series

    return {name: pd.DataFrame(columns) for name, columns in collected.items()}


def revisions(path: pd.DataFrame, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """Summarise how much later data moved each vintage's endpoint estimate.

    For every vintage `v`, the real-time estimate is `path.loc[v, v]`: what the
    model said about quarter `v` using data to `v`. The revised estimate at
    horizon `h` is `path.loc[v, v + h]`, the same quarter seen from `h` quarters
    of extra data. `final` is the same quarter seen from the full sample.

    Args:
        path: Vintage DataFrame from `run_vintages` (rows quarters, columns
            vintages).
        horizons: Revision horizons in quarters.

    Returns:
        DataFrame indexed by horizon, plus a "final" row, with the mean
        revision (bias), mean absolute revision, and the ratio of the mean
        absolute revision to the sd of the final-vintage series.

    """
    vintages = pd.PeriodIndex(list(path.columns), freq="Q")
    final_vintage = vintages[-1]
    scale = float(path[str(final_vintage)].std())

    rows = []
    for horizon in (*horizons, None):
        deltas: list[float] = []
        for vintage in vintages:
            if vintage not in path.index:
                continue
            realtime = path[str(vintage)].loc[str(vintage)]
            if pd.isna(realtime):
                continue
            later = final_vintage if horizon is None else vintage + horizon
            if str(later) not in path.columns:
                continue
            revised = path[str(later)].loc[str(vintage)]
            if pd.isna(revised):
                continue
            deltas.append(float(revised) - float(realtime))

        if not deltas:
            continue
        values = np.array(deltas)
        rows.append({
            "horizon": "final" if horizon is None else f"{horizon}q",
            "n": len(values),
            "mean_revision": values.mean(),
            "mean_abs_revision": np.abs(values).mean(),
            "max_abs_revision": np.abs(values).max(),
            "vs_sd_of_series": np.abs(values).mean() / scale if scale else np.nan,
        })

    return pd.DataFrame(rows).set_index("horizon")


def realtime_series(path: pd.DataFrame) -> pd.Series:
    """Return the endpoint estimate from each vintage: the real-time series."""
    vintages = pd.PeriodIndex(list(path.columns), freq="Q")
    points = {
        vintage: float(path[str(vintage)].loc[str(vintage)])
        for vintage in vintages
        if vintage in path.index and not pd.isna(path[str(vintage)].loc[str(vintage)])
    }
    return pd.Series(points).sort_index()


def plot_realtime(path: pd.DataFrame, label: str, ylabel: str) -> None:
    """Chart the real-time endpoint estimates against the full-sample path."""
    mg.set_chart_dir(str(CHART_DIR))

    final = path[path.columns[-1]].rename("Full-sample estimate")
    live = realtime_series(path).rename("Real-time (endpoint) estimate")

    ax = mg.line_plot(
        final,
        color=["darkorange"],
        width=2,
        annotate=False,
    )
    mg.line_plot(
        live,
        ax=ax,
        color=["darkblue"],
        width=1.5,
        style=["--"],
        marker="o",
        markersize=4,
        annotate=False,
    )
    mg.finalise_plot(
        ax,
        title=f"Real-time vs full-sample {label}",
        ylabel=ylabel,
        legend={"loc": "best", "fontsize": "small"},
        rfooter="Source: ABS 5206.0, 6401.0",
        lfooter="Australia. Pseudo-real time: current vintage, truncated. ",
        y0=True,
        show=False,
    )


def save_paths(paths: dict[str, pd.DataFrame], output_dir: Path, prefix: str) -> Path:
    """Persist the vintage paths so the report can be redrawn without resampling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{prefix}_realtime.pkl"
    with file_path.open("wb") as f:
        pickle.dump(paths, f)
    print(f"\nSaved vintage paths to: {file_path}")
    return file_path


def load_paths(output_dir: Path, prefix: str) -> dict[str, pd.DataFrame]:
    """Load previously saved vintage paths."""
    file_path = output_dir / f"{prefix}_realtime.pkl"
    with file_path.open("rb") as f:
        return pickle.load(f)


def report(paths: dict[str, pd.DataFrame]) -> None:
    """Print the revision tables and draw the real-time charts."""
    charts = {
        "trend_growth": ("trend growth", "Annualised %"),
        "potential_growth": ("potential growth", "Year-ended %"),
        "output_gap": ("output gap", "Per cent of potential"),
    }

    for name, path in paths.items():
        label, ylabel = charts[name]
        print(f"\n{'=' * 70}\nREVISIONS TO {label.upper()}\n{'=' * 70}")
        print(revisions(path).round(3).to_string())
        print("\nReal-time endpoint estimates:")
        print(realtime_series(path).round(2).to_string())
        plot_realtime(path, label, ylabel)

    print(
        "\n`vs_sd_of_series` is the mean absolute revision as a fraction of the\n"
        "full-sample series' own standard deviation. At 1.0 the revision is as\n"
        "large as the variation the estimate is meant to describe, which is the\n"
        "Orphanides-van Norden finding for US real-time output gaps.",
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Pseudo-real-time revisions for ystar")
    parser.add_argument("--first", default="2008Q4", help="Earliest vintage end-date")
    parser.add_argument("--step", type=int, default=4, help="Quarters between vintages")
    parser.add_argument("--draws", type=int, default=1_000)
    parser.add_argument("--tune", type=int, default=1_000)
    parser.add_argument("--prefix", default="ystar")
    parser.add_argument(
        "--report-only", action="store_true",
        help="Redraw from saved vintage paths without re-estimating",
    )
    args = parser.parse_args()

    config = ModelConfig()

    if args.report_only:
        paths = load_paths(config.output_dir, args.prefix)
    else:
        sampler_config = SamplerConfig(draws=args.draws, tune=args.tune, chains=4, cores=4)
        paths = run_vintages(
            first=args.first, step=args.step,
            base_config=config, sampler_config=sampler_config,
        )
        save_paths(paths, config.output_dir, args.prefix)

    report(paths)


if __name__ == "__main__":
    main()
