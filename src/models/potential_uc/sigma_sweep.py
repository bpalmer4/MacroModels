"""Sweep the fixed variance ratios and report how much the answer moves.

The model imposes its trend/cycle variance split rather than estimating it
(see `config.py`). That buys identification, but it means every headline the
model produces is conditional on the ratios chosen. This module measures how
conditional.

`ratio_g` is swept by default, being the core specification's counterpart of
HP's smoothing parameter: it governs how fast trend *growth* is allowed to
move, and trend growth is what the model exists to measure. (`ratio_g_lp` is
the labour specification's equivalent.) If the estimated path simply tracks the
ratio, the model is reporting an assumption back. If it survives the sweep, it
is in the data.

Two settings in the grids are not variances but belong here anyway:
`ratio_ystar = 0`, which strips the level innovation and leaves the integrated
random walk HP(1600) actually is, and `anchor`, which is the other thing the
Phillips curve is told rather than shown.

Usage::

    uv run python -m src.models.potential_uc.sigma_sweep
    uv run python -m src.models.potential_uc.sigma_sweep --param ratio_ystar
    uv run python -m src.models.potential_uc.sigma_sweep --param anchor
"""

import argparse
from dataclasses import replace

import pandas as pd

from src.models.potential_uc.base import SamplerConfig
from src.models.potential_uc.config import ModelConfig
from src.models.potential_uc.estimate import run_estimate
from src.models.potential_uc.results import load_results

DEFAULT_GRIDS: dict[str, list[float]] = {
    # core
    "ratio_g": [0.0125, 0.025, 0.05, 0.10, 0.20],
    # 0 is the informative end of this grid, not a corner case: it removes the
    # level innovation and leaves the integrated random walk that HP(1600)
    # actually is. See `equations/potential.py`.
    "ratio_ystar": [0.0, 0.05, 0.10, 0.13, 0.25, 0.40],
    # labour
    "ratio_g_lp": [0.0125, 0.025, 0.05, 0.10, 0.20],
    "ratio_pr_star": [0.05, 0.10, 0.20, 0.40],
    "ratio_hpp_star": [0.05, 0.10, 0.20, 0.40],
    "ratio_lp_star": [0.05, 0.10, 0.20, 0.40],
    # both
    "sigma_c": [0.40, 0.50, 0.60, 0.80, 1.00],
    # Not a variance setting, but the same kind of imposed assumption, and it
    # belongs in the same honesty check: the anchor is the other thing the
    # Phillips curve is told rather than shown. The grid is not a claim that
    # 2.0 and 2.75 are equally plausible targets — it measures how much of the
    # answer the anchor is supplying.
    "anchor": [2.0, 2.25, 2.5, 2.75],
}

# Settings that are imposed assumptions but not part of `scale_constants`.
_NON_SCALE_PARAMS = ("anchor",)

# Quarters used as fixed reference points in the report.
_EARLY = "1995Q4"
_PRE_COVID = "2019Q4"


def run_sweep(
    param: str = "ratio_g",
    values: list[float] | None = None,
    base_config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
) -> pd.DataFrame:
    """Re-estimate across a grid of one fixed variance setting.

    Args:
        param: Which ModelConfig field to vary.
        values: Grid of values (defaults to DEFAULT_GRIDS[param]).
        base_config: Config to vary from (defaults to ModelConfig()).
        sampler_config: Sampler settings (defaults to a shorter sweep run).

    Returns:
        DataFrame, one row per grid value, with the headline quantities.

    """
    if base_config is None:
        base_config = ModelConfig()
    if values is None:
        if param not in DEFAULT_GRIDS:
            raise ValueError(f"No default grid for {param!r}; pass values explicitly")
        values = DEFAULT_GRIDS[param]
    if sampler_config is None:
        sampler_config = SamplerConfig(draws=1_000, tune=1_000, chains=4, cores=4)

    if not hasattr(base_config, param):
        raise ValueError(f"ModelConfig has no field {param!r}")

    # A ratio the active spec does not use would vary silently, producing a
    # grid of identical runs and a spuriously flat sweep.
    if param not in base_config.scale_constants and param not in _NON_SCALE_PARAMS:
        raise ValueError(
            f"{param!r} is not used by the {base_config.spec!r} specification "
            f"(it uses {sorted(base_config.scale_constants)})",
        )

    rows = []
    for value in values:
        config = replace(base_config)
        setattr(config, param, value)
        prefix = f"potential_uc_sweep_{param}_{value:g}"
        print(f"\n{'=' * 70}\n{param} = {value:g}\n{'=' * 70}")

        run_estimate(config=config, sampler_config=sampler_config, prefix=prefix)
        results = load_results(output_dir=config.output_dir, prefix=prefix)

        # The core spec's trend-growth state, the labour spec's trend
        # productivity growth: in each case the drift the sweep is stressing.
        # Keyed off the state's presence rather than the spec name: `inflation`
        # and `target` carry the same g state as `core`, and a spec-name test
        # sent them into the labour branch and raised.
        trend = (
            results.trend_growth_posterior()
            if "trend_growth" in results.trace.posterior
            else results.trend_prod_growth_posterior()
        )
        pot = results.potential_growth_posterior()
        gap = results.output_gap_posterior()
        last = trend.index[-1]

        summary = results.summary()
        rhat = float(summary["r_hat"].max()) if "r_hat" in summary else float("nan")

        rows.append({
            param: value,
            "trend_g_early": trend.loc[_EARLY].median(),
            "trend_g_precovid": trend.loc[_PRE_COVID].median(),
            "trend_g_latest": trend.loc[last].median(),
            "trend_g_latest_lo": trend.loc[last].quantile(0.05),
            "trend_g_latest_hi": trend.loc[last].quantile(0.95),
            "trend_g_decline": trend.loc[_EARLY].median() - trend.loc[last].median(),
            "potential_g_latest": pot.loc[last].median(),
            "gap_latest": gap.loc[last].median(),
            "max_rhat": rhat,
        })

    return pd.DataFrame(rows)


def report(table: pd.DataFrame, param: str) -> None:
    """Print the sweep table and the spread it implies."""
    print(f"\n{'=' * 70}")
    print(f"SWEEP OVER {param}")
    print("=" * 70)
    print(table.round(3).to_string(index=False))

    latest = table["trend_g_latest"]
    decline = table["trend_g_decline"]
    print(
        f"\nTrend growth, latest: "
        f"{latest.min():.2f} to {latest.max():.2f} "
        f"(spread {latest.max() - latest.min():.2f}pp)",
    )
    print(
        f"Decline since {_EARLY}:              "
        f"{decline.min():.2f} to {decline.max():.2f} "
        f"(spread {decline.max() - decline.min():.2f}pp)",
    )
    print(
        "\nRead this as the honest band. A spread comparable to the estimate "
        "itself\nmeans the model is reporting the assumption back.",
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Sweep potential_uc variance settings")
    parser.add_argument(
        "--param", default="ratio_g", choices=sorted(DEFAULT_GRIDS),
        help="Which fixed variance setting to vary",
    )
    parser.add_argument(
        "--values", type=float, nargs="+", default=None,
        help="Explicit grid (overrides the default for --param)",
    )
    parser.add_argument("--draws", type=int, default=1_000)
    parser.add_argument("--tune", type=int, default=1_000)
    args = parser.parse_args()

    sampler_config = SamplerConfig(draws=args.draws, tune=args.tune, chains=4, cores=4)
    table = run_sweep(param=args.param, values=args.values, sampler_config=sampler_config)
    report(table, args.param)


if __name__ == "__main__":
    main()
