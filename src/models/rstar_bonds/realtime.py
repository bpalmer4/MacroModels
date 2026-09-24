"""Pseudo-real-time endpoint revisions for the bond-market r*.

Refinement 1 in MODEL_NOTES, and the one the notes call the fragility that
matters most: r* is a random walk, so its last value is the least constrained
point in the sample, and the headline *is* that last value. In-sample fit says
nothing about it. This module measures it directly: re-estimate on a truncated
sample, keep the estimate of *today* a user would have had at that date, then
ask how much later data moved it.

    revision(t, h) = estimate of x_t from the sample ending t+h
                   - estimate of x_t from the sample ending t

The reference for the exercise is Orphanides and van Norden (2002), *The
Unreliability of Output Gap Estimates in Real Time*, which found US output gap
revisions of the same order as the gaps themselves.

**Why this test is cleaner here than in `ystar`.** The inputs are mostly market
prices, and a bond yield is never revised: the indexed 10-year, the cash rate
and the world anchor are what they were. So the data-revision term that
contaminates a GDP-based exercise is largely absent, and what is left is closer
to the filter's endpoint problem on its own.

**Two honest exceptions, and they run opposite ways.**

1. The default deflator is `expectations`, which is a *model* output estimated
   on the full sample (`get_model_expectations_unanchored`). Truncating `end`
   shortens that series but does not un-learn it, so the real cash rate carries
   genuine look-ahead. Run `--deflator trimmed` for the comparator with no
   model in the deflator: it is ABS data, revised only as the ABS revises it.
   If the two disagree, the difference is the look-ahead.
2. The Cleveland Fed anchor (`REAINTRATREARAT10Y`) is also a model estimate and
   is re-fitted through history, so the same caveat applies to `w`, and there
   is no data-only substitute for it that covers the sample. `--world-source
   tips` is a pure price but starts in 2003.

So this is pseudo-real time, and these numbers are a **lower bound** on
real-time unreliability.

**What it is for.** The notes leave the choice between the default
specification and the `--us-premium` pin open, and the two are hard to separate
on in-sample grounds: their r* paths correlate at 0.966 and each sits inside
the other's band. Endpoint stability is a criterion that does not reduce to
taste, so `--spec default` and `--spec pin` are the intended pair.

Usage::

    uv run python -m src.models.rstar_bonds.realtime
    uv run python -m src.models.rstar_bonds.realtime --spec pin
    uv run python -m src.models.rstar_bonds.realtime --deflator trimmed
    uv run python -m src.models.rstar_bonds.realtime --report-only
"""

import argparse
import pickle
from dataclasses import replace
from pathlib import Path

import mgplot as mg
import numpy as np
import pandas as pd

from src.models.rstar_bonds.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.rstar_bonds.estimate import build_model
from src.models.rstar_bonds.observations import build_observations
from src.models.rstar_bonds.results import DEFAULT_CHART_BASE
from src.models.ystar.base import SamplerConfig, sample_model

# Its own directory: `analyse.run_analysis` clears RStarBonds/ before writing,
# so charts left there by a separate command would not survive the next run.
CHART_DIR = DEFAULT_CHART_BASE / "RStarBonds" / "RealTime"

# Horizons reported, in quarters. Every one must be a multiple of the vintage
# step, or no pair of vintages is that far apart and the row is empty.
DEFAULT_HORIZONS = (4, 8, 12, 20)

# Latent paths tracked. r* is the headline, `wedge` is what the notes say is
# the model's actual output, and `g` is the stance, which is the quantity most
# often quoted off this model and the one a sign change would embarrass.
_QUANTITIES = ("r_star", "wedge", "g")

_LABELS = {
    "r_star": ("real r*", "Per cent"),
    "wedge": ("the Australian wedge", "Percentage points"),
    "g": ("the policy stance", "Percentage points, + restrictive"),
}

# The two specifications the exercise exists to separate. `pin` is the
# `rstar_pin` combination named in MODEL_NOTES: market anchor, imposed loading,
# term premium pinned to the published US one.
SPECS = ("default", "pin")


def spec_config(spec: str, base: ModelConfig | None = None) -> ModelConfig:
    """Return the ModelConfig for a named specification."""
    config = base if base is not None else ModelConfig()
    if spec == "default":
        return config
    if spec == "pin":
        return replace(
            config,
            world_source="market",
            free_world_loading=False,
            us_premium_anchor=True,
        )
    raise ValueError(f"spec must be one of {', '.join(SPECS)}, got {spec!r}")


def _paths(config: ModelConfig, sampler_config: SamplerConfig) -> dict[str, pd.Series]:
    """Estimate once and return the median path of each tracked quantity.

    Sampling happens here rather than through `run_estimate` because a
    real-time exercise is dozens of runs and each full trace is around 50MB on
    disk. Only the medians are needed, since revisions are about the point
    estimate a user would have quoted.

    The Taylor rule is not built: it lives outside the likelihood, cannot move
    the state, and reads a completed `ystar_ustar` run that this exercise does
    not truncate. Including it would add look-ahead for no gain.
    """
    obs, obs_index, _, _ = build_observations(
        start=config.start,
        end=config.end,
        world_source=config.world_source,
        deflator=config.deflator,
        short_rate=config.short_rate,
        us_premium_anchor=config.us_premium_anchor,
        us_premium_source=config.us_premium_source,
        use_curve=config.use_curve,
        curve_maturity=config.curve_maturity,
        verbose=False,
    )
    model = build_model(obs, obs_index, config=config, verbose=False)
    trace = sample_model(model, sampler_config)

    posterior = trace.posterior
    out: dict[str, pd.Series] = {}
    for name in _QUANTITIES:
        # The package idiom: xarray's `stack`, not pandas', so PD013 is a
        # false positive. See `results.RStarResults._vector`.
        stacked = posterior[name].stack(sample=("chain", "draw"))
        draws = np.asarray(stacked.values)
        out[name] = pd.Series(np.median(draws, axis=1), index=obs_index)
    return out


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
    if sampler_config is None:
        # Production settings, not a shortened run. Draw count is not the
        # binding constraint here (r* MCSE is 0.010 at 2,000 draws against
        # revisions of 0.1 to 0.5), but TUNING is: at tune=1,000 the short
        # early vintages threw divergences, and divergences are biased
        # exploration rather than noise, which no MCSE calculation catches.
        # It also keeps the final-vintage column comparable to the published
        # trace, which is what every revision here is measured against.
        sampler_config = SamplerConfig()

    # The full sample defines the last vintage and hence the "final" estimate
    # every revision is measured against.
    _, full_index, _, _ = build_observations(
        start=base_config.start,
        end=base_config.end,
        world_source=base_config.world_source,
        deflator=base_config.deflator,
        short_rate=base_config.short_rate,
        us_premium_anchor=base_config.us_premium_anchor,
        us_premium_source=base_config.us_premium_source,
        use_curve=base_config.use_curve,
        curve_maturity=base_config.curve_maturity,
        verbose=False,
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
    horizon `h` is `path.loc[v, v + h]`, the same quarter seen from `h`
    quarters of extra data. `final` is the same quarter from the full sample.

    Args:
        path: Vintage DataFrame from `run_vintages` (rows quarters, columns
            vintages).
        horizons: Revision horizons in quarters.

    Returns:
        DataFrame indexed by horizon, plus a "final" row, with the mean
        revision (bias), mean and max absolute revision, the ratio of the mean
        absolute revision to the sd of the final-vintage series, and the share
        of revisions that flip the estimate's sign.

    """
    vintages = pd.PeriodIndex(list(path.columns), freq="Q")
    final_vintage = vintages[-1]
    scale = float(path[str(final_vintage)].std())

    rows = []
    for horizon in (*horizons, None):
        deltas: list[float] = []
        flips = 0
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
            if np.sign(float(revised)) != np.sign(float(realtime)):
                flips += 1

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
            "sign_flips": flips / len(values),
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


def plot_realtime(path: pd.DataFrame, label: str, ylabel: str, spec: str) -> None:
    """Chart the real-time endpoint estimates against the full-sample path."""
    mg.set_chart_dir(str(CHART_DIR))

    final = path[path.columns[-1]].rename("Full-sample estimate")
    live = realtime_series(path).rename("Real-time (endpoint) estimate")

    ax = mg.line_plot(final, color=["darkorange"], width=2, annotate=False)
    mg.line_plot(
        live, ax=ax, color=["darkblue"], width=1.5, style=["--"],
        marker="o", markersize=4, annotate=False,
    )
    mg.finalise_plot(
        ax,
        title=f"Real-time vs full-sample {label}",
        ylabel=ylabel,
        legend={"loc": "best", "fontsize": "small"},
        rfooter="FRED; RBA; ABS",
        lfooter=f"Australia. Pseudo-real time, {spec} spec. ",
        y0=True,
        tag=spec,
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


def report(paths: dict[str, pd.DataFrame], spec: str) -> None:
    """Print the revision tables and draw the real-time charts."""
    for name, path in paths.items():
        label, ylabel = _LABELS[name]
        print(f"\n{'=' * 70}\nREVISIONS TO {label.upper()} ({spec})\n{'=' * 70}")
        print(revisions(path).round(3).to_string())
        print("\nReal-time endpoint estimates:")
        print(realtime_series(path).round(2).to_string())
        plot_realtime(path, label, ylabel, spec)

    print(
        "\n`vs_sd_of_series` is the mean absolute revision as a fraction of the\n"
        "full-sample series' own standard deviation. At 1.0 the revision is as\n"
        "large as the variation the estimate is meant to describe, which is the\n"
        "Orphanides-van Norden finding for US real-time output gaps.\n"
        "`sign_flips` is the share of vintages whose estimate later changed sign.\n"
        "For `g` that is the stance calling policy easy when it was tight.",
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pseudo-real-time endpoint revisions for rstar_bonds",
    )
    parser.add_argument("--first", default="2008Q4", help="Earliest vintage end-date")
    parser.add_argument("--step", type=int, default=4, help="Quarters between vintages")
    parser.add_argument(
        "--spec", default="default", choices=list(SPECS),
        help="Which specification to test: 'default', or 'pin' for the rstar_pin "
             "combination (market anchor, imposed loading, premium pinned)",
    )
    parser.add_argument(
        "--deflator", default=None, choices=["expectations", "trimmed"],
        help="Override the deflator. 'trimmed' removes the model-based look-ahead "
             "in the real cash rate (see the module docstring)",
    )
    # Matching production. Lowering `tune` is what produced divergences in the
    # short early vintages, so treat these as the floor rather than the default
    # to economise on.
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--tune", type=int, default=2_000)
    parser.add_argument("--prefix", default=None, help="Output prefix (default: rstar_bonds_<spec>)")
    parser.add_argument(
        "--report-only", action="store_true",
        help="Redraw from saved vintage paths without re-estimating",
    )
    args = parser.parse_args()

    config = spec_config(args.spec)
    if args.deflator is not None:
        config = replace(config, deflator=args.deflator)
    prefix = args.prefix or f"rstar_bonds_{args.spec}"

    if args.report_only:
        paths = load_paths(DEFAULT_OUTPUT_DIR, prefix)
    else:
        sampler_config = SamplerConfig(draws=args.draws, tune=args.tune, chains=4, cores=4)
        paths = run_vintages(
            first=args.first, step=args.step,
            base_config=config, sampler_config=sampler_config,
        )
        save_paths(paths, DEFAULT_OUTPUT_DIR, prefix)

    report(paths, args.spec)


if __name__ == "__main__":
    main()
