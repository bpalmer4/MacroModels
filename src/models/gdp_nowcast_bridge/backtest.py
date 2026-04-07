"""GDP nowcast backtesting.

Runs the nowcast model across historical quarters at four information set
timings (T-3m, T-2m, T-1m, T-0) and evaluates against actual GDP outcomes.

Note: This is a pseudo real-time backtest using latest-revised data for both
inputs and evaluation. Upgrading to true vintage data (via readabs history
parameter) is a future improvement.

Usage:
    uv run python -m src.models.gdp_nowcast_bridge.backtest
    uv run python -m src.models.gdp_nowcast_bridge.backtest --start 2022Q1 --end 2025Q4
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import mgplot as mg
import numpy as np
import pandas as pd

from src.data.gdp import get_gdp
from src.models.gdp_nowcast_bridge.model import (
    DataAvailability,
    _compute_gdp_growth,
    _load_monthly_indicators,
    _load_quarterly_indicators,
    nowcast,
)

logger = logging.getLogger(__name__)

BACKTEST_CHART_DIR = "./charts/GDP-Nowcast-Backtest/"
BACKTEST_OUTPUT_DIR = "./model_outputs/gdp_nowcast/"

INFO_SETS = {
    "T-3m": DataAvailability.at_t_minus_3m,
    "T-2m": DataAvailability.at_t_minus_2m,
    "T-1m": DataAvailability.at_t_minus_1m,
    "T-0": DataAvailability.at_t_minus_0,
}


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    start: str = "2022Q1"
    end: str | None = None  # None = latest published quarter


@dataclass
class BacktestResults:
    """Results from a backtest run."""

    config: BacktestConfig
    results: pd.DataFrame  # Quarter × info_set → nowcast_qoq, actual_qoq, error, etc.
    summary: pd.DataFrame  # Metrics by info_set


def _get_backtest_quarters(config: BacktestConfig, gdp: pd.Series) -> list[pd.Period]:
    """Generate list of quarters to backtest."""
    start = pd.Period(config.start, "Q-DEC")
    end = pd.Period(config.end, "Q-DEC") if config.end is not None else gdp.dropna().index[-1]

    # Only include quarters where we have actual GDP to evaluate against
    quarters = []
    current = start
    while current <= end:
        if current in gdp.index and pd.notna(gdp.loc[current]):
            quarters.append(current)
        current = current + 1

    return quarters


def run_backtest(config: BacktestConfig | None = None) -> BacktestResults:
    """Run the backtest across historical quarters.

    For each quarter, runs the nowcast at four information set timings
    and compares to the actual GDP outcome.

    """
    if config is None:
        config = BacktestConfig()

    # Load all data once (latest vintage)
    print("Loading data...")
    gdp_ds = get_gdp(gdp_type="CVM", seasonal="SA")
    gdp = gdp_ds.data.dropna()
    gdp_growth = _compute_gdp_growth(gdp)

    monthly_indicators = _load_monthly_indicators()
    quarterly_indicators = _load_quarterly_indicators()

    quarters = _get_backtest_quarters(config, gdp)
    print(f"Backtesting {len(quarters)} quarters: {quarters[0]} to {quarters[-1]}")

    # Run nowcasts
    rows = []
    for i, target_q in enumerate(quarters):
        actual_qoq = gdp_growth.loc[target_q] if target_q in gdp_growth.index else np.nan
        actual_tty = ((gdp.loc[target_q] / gdp.loc[target_q - 4] - 1) * 100
                      if target_q - 4 in gdp.index else np.nan)

        for info_name, info_factory in INFO_SETS.items():
            availability = info_factory(target_q)

            try:
                result = nowcast(
                    target_quarter=target_q,
                    availability=availability,
                    gdp=gdp.copy(),
                    monthly_indicators=monthly_indicators,
                    quarterly_indicators=quarterly_indicators,
                    quiet=True,
                )

                rows.append({
                    "quarter": target_q,
                    "info_set": info_name,
                    "nowcast_qoq": result.gdp_qoq,
                    "actual_qoq": actual_qoq,
                    "error_qoq": result.gdp_qoq - actual_qoq,
                    "nowcast_tty": result.gdp_tty,
                    "actual_tty": actual_tty,
                    "error_tty": result.gdp_tty - actual_tty,
                    "n_active_bridges": sum(1 for b in result.bridge_results if b.available),
                    "ci_70_lower": result.gdp_qoq_70[0],
                    "ci_70_upper": result.gdp_qoq_70[1],
                    "ci_90_lower": result.gdp_qoq_90[0],
                    "ci_90_upper": result.gdp_qoq_90[1],
                })

            except RuntimeError:
                logger.warning("Failed: %s at %s", target_q, info_name)
                rows.append({
                    "quarter": target_q,
                    "info_set": info_name,
                    "nowcast_qoq": np.nan,
                    "actual_qoq": actual_qoq,
                    "error_qoq": np.nan,
                    "nowcast_tty": np.nan,
                    "actual_tty": actual_tty,
                    "error_tty": np.nan,
                    "n_active_bridges": 0,
                    "ci_70_lower": np.nan,
                    "ci_70_upper": np.nan,
                    "ci_90_lower": np.nan,
                    "ci_90_upper": np.nan,
                })

        # Progress
        if (i + 1) % 5 == 0 or i == len(quarters) - 1:
            print(f"  {i + 1}/{len(quarters)} quarters completed")

    results_df = pd.DataFrame(rows)

    # Compute summary statistics by info set
    summary = _compute_summary(results_df)

    bt_results = BacktestResults(config=config, results=results_df, summary=summary)

    # Output
    _print_backtest_summary(bt_results)
    _save_results(bt_results)
    _plot_backtest(bt_results, gdp_growth)

    return bt_results


def _compute_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics by information set."""
    rows = []
    for info_set in INFO_SETS:
        subset = results[results["info_set"] == info_set].dropna(subset=["error_qoq"])

        if len(subset) == 0:
            continue

        errors = subset["error_qoq"]
        errors_tty = subset["error_tty"].dropna()

        # Direction accuracy: did we get the sign of Q/Q change right?
        direction_correct = (
            (subset["nowcast_qoq"] > 0) == (subset["actual_qoq"] > 0)
        ).mean()

        # CI coverage
        ci_70_covers = (
            (subset["actual_qoq"] >= subset["ci_70_lower"])
            & (subset["actual_qoq"] <= subset["ci_70_upper"])
        ).mean()

        ci_90_covers = (
            (subset["actual_qoq"] >= subset["ci_90_lower"])
            & (subset["actual_qoq"] <= subset["ci_90_upper"])
        ).mean()

        # Naive benchmark: GDP grows at its trailing 4-quarter average
        naive_errors = subset["actual_qoq"] - subset["actual_qoq"].rolling(4).mean().shift(1)
        naive_rmse = np.sqrt((naive_errors ** 2).mean()) if len(naive_errors.dropna()) > 0 else np.nan

        # Correlation between nowcast and actual — measures shape-tracking
        corr = subset[["nowcast_qoq", "actual_qoq"]].corr().iloc[0, 1]
        nowcast_std = subset["nowcast_qoq"].std()

        rows.append({
            "info_set": info_set,
            "n_quarters": len(subset),
            "rmse_qoq": np.sqrt((errors ** 2).mean()),
            "mae_qoq": errors.abs().mean(),
            "mean_bias_qoq": errors.mean(),
            "rmse_tty": np.sqrt((errors_tty ** 2).mean()) if len(errors_tty) > 0 else np.nan,
            "mae_tty": errors_tty.abs().mean() if len(errors_tty) > 0 else np.nan,
            "direction_accuracy": direction_correct,
            "ci_70_coverage": ci_70_covers,
            "ci_90_coverage": ci_90_covers,
            "naive_rmse_qoq": naive_rmse,
            "correlation": corr,
            "nowcast_std": nowcast_std,
            "avg_active_bridges": subset["n_active_bridges"].mean(),
        })

    return pd.DataFrame(rows).set_index("info_set")


def _print_backtest_summary(bt: BacktestResults) -> None:
    """Print backtest summary to terminal."""
    print("\n" + "=" * 80)
    print(f"  BACKTEST SUMMARY: {bt.config.start} to {bt.config.end or 'latest'}")
    print("=" * 80)

    s = bt.summary
    print(f"\n  {'Info Set':<8} {'RMSE':>7} {'MAE':>7} {'Bias':>8} {'Dir%':>6} "
          f"{'Corr':>7} {'NCstd':>7} {'70%CI':>6} {'90%CI':>6} {'Naive':>7} {'Br':>5}")
    print("  " + "-" * 82)

    for info_set in s.index:
        row = s.loc[info_set]
        print(f"  {info_set:<8} "
              f"{row['rmse_qoq']:>6.3f}% "
              f"{row['mae_qoq']:>6.3f}% "
              f"{row['mean_bias_qoq']:>+7.3f}% "
              f"{row['direction_accuracy']:>5.0%} "
              f"{row['correlation']:>+7.3f} "
              f"{row['nowcast_std']:>6.3f}% "
              f"{row['ci_70_coverage']:>5.0%} "
              f"{row['ci_90_coverage']:>5.0%} "
              f"{row['naive_rmse_qoq']:>6.3f}% "
              f"{row['avg_active_bridges']:>5.1f}")

    print("\n  RMSE = Root Mean Squared Error (Q/Q growth)")
    print("  MAE = Mean Absolute Error")
    print("  Dir% = Direction accuracy (positive/negative growth)")
    print("  Corr = Correlation between nowcast and actual (shape-tracking)")
    print("  NCstd = Std deviation of nowcasts (model variance — flat = ~naive)")
    print("  70%CI, 90%CI = Confidence interval coverage rates")
    print("  Naive = RMSE of trailing 4-quarter average benchmark")
    print("  Br = Average number of active bridge equations")
    print("\n  Note: Pseudo real-time backtest using latest-revised data.")
    print("=" * 80)


def _save_results(bt: BacktestResults) -> None:
    """Save backtest results to disk."""
    output_dir = Path(BACKTEST_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    bt.results.to_parquet(output_dir / "backtest_results.parquet", index=False)
    bt.summary.to_csv(output_dir / "backtest_summary.csv")

    with (output_dir / "backtest_summary.txt").open("w") as f:
        f.write(f"Backtest: {bt.config.start} to {bt.config.end or 'latest'}\n\n")
        f.write(bt.summary.to_string())
        f.write("\n\nNote: Pseudo real-time backtest using latest-revised data.\n")

    print(f"\n  Results saved to {output_dir}/")


# --- Backtest charts ---


def _plot_backtest(bt: BacktestResults, gdp_growth: pd.Series) -> None:
    """Generate backtest evaluation charts."""
    mg.set_chart_dir(BACKTEST_CHART_DIR)
    mg.clear_chart_dir()

    _plot_rmse_by_info_set(bt)
    _plot_actual_vs_nowcast(bt)
    _plot_error_distribution(bt)
    _plot_nowcast_evolution(bt, gdp_growth)


def _plot_rmse_by_info_set(bt: BacktestResults) -> None:
    """Bar chart of RMSE by information set."""
    s = bt.summary

    rmse_df = pd.DataFrame({
        "Nowcast": s["rmse_qoq"],
        "Naive": s["naive_rmse_qoq"],
    })

    mg.bar_plot_finalise(
        rmse_df,
        title="Bridge Model RMSE by Information Set (Q/Q Growth)",
        ylabel="RMSE (percentage points)",
        rfooter=f"{bt.config.start} to {bt.config.end or 'latest'}",
        lfooter="Pseudo real-time backtest, latest-revised data. ",
        legend={"loc": "upper right", "fontsize": "small"},
        show=False,
    )


def _plot_actual_vs_nowcast(bt: BacktestResults) -> None:
    """Actual vs nowcast at T-0 with 90% confidence band."""
    t0 = bt.results[bt.results["info_set"] == "T-0"].copy()
    t0 = t0.set_index("quarter").sort_index()

    # 90% CI band behind the nowcast
    ci_band = pd.DataFrame({
        "lower": t0["ci_90_lower"],
        "upper": t0["ci_90_upper"],
    })
    ax = mg.fill_between_plot(ci_band, color="red", alpha=0.12, label="90% CI")

    # Nowcast and actual lines on top
    df = pd.DataFrame({
        "Actual": t0["actual_qoq"],
        "Nowcast (T-0)": t0["nowcast_qoq"],
    })
    mg.line_plot(df, ax=ax, color=["navy", "red"], width=[2, 1.5], style=["-", "--"])

    mg.finalise_plot(
        ax,
        title="Bridge Model: Actual vs Nowcast GDP Growth (Q/Q, T-0)",
        ylabel="Per cent",
        rfooter="Source: ABS 5206.0",
        lfooter="Pseudo real-time backtest, latest-revised data. ",
        y0=True,
        legend={"loc": "lower left", "fontsize": "small"},
        show=False,
    )


def _plot_error_distribution(bt: BacktestResults) -> None:
    """Error distribution across information sets."""
    errors = {}
    for info_set in INFO_SETS:
        subset = bt.results[bt.results["info_set"] == info_set]
        errors[info_set] = subset["error_qoq"].dropna()

    # Plot as box-like: show errors over time for each info set
    error_df = pd.DataFrame({
        info_set: bt.results[bt.results["info_set"] == info_set].set_index("quarter")["error_qoq"]
        for info_set in INFO_SETS
    })

    mg.line_plot_finalise(
        error_df,
        title="Bridge Model Nowcast Errors by Information Set (Q/Q Growth)",
        ylabel="Error (percentage points)",
        rfooter=f"{bt.config.start} to {bt.config.end or 'latest'}",
        lfooter="Positive = overestimate. Pseudo real-time backtest. ",
        y0=True,
        width=1.5,
        legend={"loc": "lower left", "fontsize": "x-small"},
        show=False,
    )


def _plot_nowcast_evolution(
    bt: BacktestResults,
    _gdp_growth: pd.Series,  # retained for future overlay of actual GDP growth
) -> None:
    """Show how the nowcast converges to actual as data arrives."""
    evolution_data = {}
    for info_set in INFO_SETS:
        subset = bt.results[bt.results["info_set"] == info_set].set_index("quarter")
        evolution_data[info_set] = subset["error_qoq"].abs()

    mae_evolution = pd.DataFrame(evolution_data)

    min_rolling = 4
    rolling_mae = (mae_evolution.rolling(min_rolling, min_periods=2).mean()
                   if len(mae_evolution) > min_rolling else mae_evolution)

    mg.line_plot_finalise(
        rolling_mae,
        title="Bridge Model Nowcast Accuracy Evolution (4-quarter rolling MAE)",
        ylabel="MAE (percentage points)",
        rfooter=f"{bt.config.start} to {bt.config.end or 'latest'}",
        lfooter="Lower = better. Pseudo real-time backtest. ",
        width=2,
        legend={"loc": "upper right", "fontsize": "small"},
        show=False,
    )


# --- CLI ---


def main() -> None:
    """CLI entry point for backtesting."""
    parser = argparse.ArgumentParser(description="GDP Nowcast Backtest")
    parser.add_argument("--start", default="2022Q1", help="First quarter to backtest (default: 2022Q1)")
    parser.add_argument("--end", default=None, help="Last quarter to backtest (default: latest)")
    args = parser.parse_args()

    config = BacktestConfig(start=args.start, end=args.end)
    run_backtest(config)


if __name__ == "__main__":
    main()
