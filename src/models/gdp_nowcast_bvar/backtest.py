"""GDP nowcast BVAR backtesting (T-0 only).

The BVAR is a T-0 model — it requires all quarterly indicators to be published
before it can produce a nowcast. So unlike the bridge and DFM backtests, there
is only one information set (T-0).

Usage:
    uv run python -m src.models.gdp_nowcast_bvar.backtest
    uv run python -m src.models.gdp_nowcast_bvar.backtest --start 2022Q1 --end 2025Q4
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import mgplot as mg
import numpy as np
import pandas as pd

from src.data.gdp import get_gdp
from src.models.gdp_nowcast_bvar.model import _load_panel, nowcast

logger = logging.getLogger(__name__)

BACKTEST_CHART_DIR = "./charts/GDP-Nowcast-BVAR-Backtest/"
BACKTEST_OUTPUT_DIR = "./model_outputs/gdp_nowcast_bvar/"


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    start: str = "2022Q1"
    end: str | None = None


@dataclass
class BacktestResults:
    """Results from a backtest run."""

    config: BacktestConfig
    results: pd.DataFrame
    summary: pd.DataFrame


def _get_backtest_quarters(config: BacktestConfig, gdp: pd.Series) -> list[pd.Period]:
    """Generate list of quarters to backtest."""
    start = pd.Period(config.start, "Q-DEC")
    end = pd.Period(config.end, "Q-DEC") if config.end is not None else gdp.dropna().index[-1]
    quarters = []
    current = start
    while current <= end:
        if current in gdp.index and pd.notna(gdp.loc[current]):
            quarters.append(current)
        current = current + 1
    return quarters


def run_backtest(config: BacktestConfig | None = None) -> BacktestResults:
    """Run the backtest across historical quarters at T-0 only."""
    if config is None:
        config = BacktestConfig()

    print("Loading data...")
    gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()
    gdp_growth = (np.log(gdp).diff(1) * 100)
    panel = _load_panel()

    quarters = _get_backtest_quarters(config, gdp)
    print(f"Backtesting {len(quarters)} quarters: {quarters[0]} to {quarters[-1]}")

    rows = []
    for i, target_q in enumerate(quarters):
        actual_qoq = gdp_growth.loc[target_q] if target_q in gdp_growth.index else np.nan
        actual_tty = (
            (gdp.loc[target_q] / gdp.loc[target_q - 4] - 1) * 100
            if target_q - 4 in gdp.index
            else np.nan
        )

        try:
            result = nowcast(
                target_quarter=target_q,
                panel=panel.copy(),
                gdp=gdp.copy(),
                quiet=True,
            )
            rows.append({
                "quarter": target_q,
                "info_set": "T-0",
                "nowcast_qoq": result.gdp_qoq,
                "actual_qoq": actual_qoq,
                "error_qoq": result.gdp_qoq - actual_qoq,
                "nowcast_tty": result.gdp_tty,
                "actual_tty": actual_tty,
                "error_tty": result.gdp_tty - actual_tty,
                "ci_70_lower": result.gdp_qoq_70[0],
                "ci_70_upper": result.gdp_qoq_70[1],
                "ci_90_lower": result.gdp_qoq_90[0],
                "ci_90_upper": result.gdp_qoq_90[1],
            })
        except (RuntimeError, ValueError) as e:
            logger.warning("Failed: %s — %s", target_q, e)
            rows.append({
                "quarter": target_q,
                "info_set": "T-0",
                "nowcast_qoq": np.nan,
                "actual_qoq": actual_qoq,
                "error_qoq": np.nan,
                "nowcast_tty": np.nan,
                "actual_tty": actual_tty,
                "error_tty": np.nan,
                "ci_70_lower": np.nan,
                "ci_70_upper": np.nan,
                "ci_90_lower": np.nan,
                "ci_90_upper": np.nan,
            })

        if (i + 1) % 5 == 0 or i == len(quarters) - 1:
            print(f"  {i + 1}/{len(quarters)} quarters completed")

    results_df = pd.DataFrame(rows)
    summary = _compute_summary(results_df)
    bt_results = BacktestResults(config=config, results=results_df, summary=summary)

    _print_backtest_summary(bt_results)
    _save_results(bt_results)
    _plot_backtest(bt_results)

    return bt_results


def _compute_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for the T-0 backtest."""
    subset = results.dropna(subset=["error_qoq"])
    if len(subset) == 0:
        return pd.DataFrame()

    errors = subset["error_qoq"]
    errors_tty = subset["error_tty"].dropna()

    direction_correct = ((subset["nowcast_qoq"] > 0) == (subset["actual_qoq"] > 0)).mean()
    ci_70_covers = (
        (subset["actual_qoq"] >= subset["ci_70_lower"])
        & (subset["actual_qoq"] <= subset["ci_70_upper"])
    ).mean()
    ci_90_covers = (
        (subset["actual_qoq"] >= subset["ci_90_lower"])
        & (subset["actual_qoq"] <= subset["ci_90_upper"])
    ).mean()

    naive_errors = subset["actual_qoq"] - subset["actual_qoq"].rolling(4).mean().shift(1)
    naive_rmse = (
        np.sqrt((naive_errors ** 2).mean()) if len(naive_errors.dropna()) > 0 else np.nan
    )
    corr = subset[["nowcast_qoq", "actual_qoq"]].corr().iloc[0, 1]
    nowcast_std = subset["nowcast_qoq"].std()

    row = {
        "info_set": "T-0",
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
    }
    return pd.DataFrame([row]).set_index("info_set")


def _print_backtest_summary(bt: BacktestResults) -> None:
    """Print backtest summary to terminal."""
    print("\n" + "=" * 80)
    print(f"  BVAR BACKTEST SUMMARY: {bt.config.start} to {bt.config.end or 'latest'}")
    print("=" * 80)

    s = bt.summary
    if len(s) == 0:
        print("\n  No successful nowcasts.")
        return

    print(f"\n  {'Info Set':<8} {'RMSE':>7} {'MAE':>7} {'Bias':>8} {'Dir%':>6} "
          f"{'Corr':>7} {'NCstd':>7} {'70%CI':>6} {'90%CI':>6} {'Naive':>7}")
    print("  " + "-" * 76)

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
              f"{row['naive_rmse_qoq']:>6.3f}%")

    print("\n  RMSE = Root Mean Squared Error (Q/Q growth)")
    print("  MAE = Mean Absolute Error")
    print("  Dir% = Direction accuracy (positive/negative growth)")
    print("  Corr = Correlation between nowcast and actual (shape-tracking)")
    print("  NCstd = Std deviation of nowcasts (model variance — flat = ~naive)")
    print("  70%CI, 90%CI = Confidence interval coverage rates")
    print("  Naive = RMSE of trailing 4-quarter average benchmark")
    print("\n  Note: Pseudo real-time backtest using latest-revised data.")
    print("=" * 80)


def _save_results(bt: BacktestResults) -> None:
    """Save backtest results to disk."""
    output_dir = Path(BACKTEST_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    bt.results.to_parquet(output_dir / "backtest_results.parquet", index=False)
    bt.summary.to_csv(output_dir / "backtest_summary.csv")

    with (output_dir / "backtest_summary.txt").open("w") as f:
        f.write(f"BVAR Backtest: {bt.config.start} to {bt.config.end or 'latest'}\n\n")
        f.write(bt.summary.to_string())
        f.write("\n\nNote: Pseudo real-time backtest using latest-revised data.\n")

    print(f"\n  Results saved to {output_dir}/")


def _plot_backtest(bt: BacktestResults) -> None:
    """Generate backtest evaluation charts."""
    mg.set_chart_dir(BACKTEST_CHART_DIR)
    mg.clear_chart_dir()
    _plot_actual_vs_nowcast(bt)
    _plot_errors(bt)


def _plot_actual_vs_nowcast(bt: BacktestResults) -> None:
    """Actual vs nowcast at T-0 with 90% confidence band."""
    t0 = bt.results.set_index("quarter").sort_index()

    ci_band = pd.DataFrame({
        "lower": t0["ci_90_lower"],
        "upper": t0["ci_90_upper"],
    })
    ax = mg.fill_between_plot(ci_band, color="green", alpha=0.12, label="90% CI")

    df = pd.DataFrame({
        "Actual": t0["actual_qoq"],
        "BVAR Nowcast (T-0)": t0["nowcast_qoq"],
    })
    mg.line_plot(df, ax=ax, color=["navy", "green"], width=[2, 1.5], style=["-", "--"])

    mg.finalise_plot(
        ax,
        title="Actual vs BVAR Nowcast GDP Growth (Q/Q, T-0)",
        ylabel="Per cent",
        rfooter="Source: ABS 5206.0",
        lfooter="Pseudo real-time backtest, latest-revised data. ",
        y0=True,
        legend={"loc": "lower left", "fontsize": "small"},
        show=False,
    )


def _plot_errors(bt: BacktestResults) -> None:
    """Error time series."""
    err = bt.results.set_index("quarter").sort_index()["error_qoq"]

    mg.line_plot_finalise(
        err.rename("BVAR Error"),
        title="BVAR Nowcast Errors (Q/Q Growth, T-0)",
        ylabel="Error (percentage points)",
        rfooter=f"{bt.config.start} to {bt.config.end or 'latest'}",
        lfooter="Positive = overestimate. Pseudo real-time backtest. ",
        y0=True,
        width=2,
        legend={"loc": "lower left", "fontsize": "small"},
        show=False,
    )


# --- CLI ---


def main() -> None:
    """CLI entry point for backtesting."""
    parser = argparse.ArgumentParser(description="GDP Nowcast BVAR Backtest")
    parser.add_argument("--start", default="2022Q1", help="First quarter (default: 2022Q1)")
    parser.add_argument("--end", default=None, help="Last quarter (default: latest)")
    args = parser.parse_args()

    config = BacktestConfig(start=args.start, end=args.end)
    run_backtest(config)


if __name__ == "__main__":
    main()
