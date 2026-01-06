"""Inflation expectations Stage 2: Diagnostics and plotting.

This module handles:
- Loading saved results from Stage 1
- Running diagnostics
- Generating all analysis plots

Run with: uv run python -m src.models.expectations.stage2
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import mgplot as mg
import pandas as pd

from src.data.inflation import get_trimmed_mean_annual
from src.models.common.diagnostics import check_model_diagnostics
from src.models.common.timeseries import plot_posterior_timeseries
from src.models.expectations.common import CHART_DIR, MODEL_NAMES, MODEL_TYPES, OUTPUT_DIR


# --- Results Container ---


@dataclass
class ExpectationsResults:
    """Container for loaded results."""

    trace: az.InferenceData
    measures: pd.DataFrame
    inflation: pd.Series
    index: pd.PeriodIndex
    model_type: str

    def expectations_posterior(self) -> pd.DataFrame:
        """Get full posterior samples."""
        samples = self.trace.posterior["pi_exp"].values
        n_chains, n_draws, n_time = samples.shape
        flat = samples.reshape(n_chains * n_draws, n_time).T
        return pd.DataFrame(flat, index=self.index)

    def expectations_median(self) -> pd.Series:
        """Get posterior median."""
        return self.expectations_posterior().median(axis=1)

    def expectations_hdi(self, prob: float = 0.9) -> pd.DataFrame:
        """Get HDI bounds."""
        post = self.expectations_posterior()
        alpha = (1 - prob) / 2
        return pd.DataFrame({
            "lower": post.quantile(alpha, axis=1),
            "median": post.median(axis=1),
            "upper": post.quantile(1 - alpha, axis=1),
        }, index=self.index)


# --- Loading ---


def load_results(model_type: str, output_dir: Path | None = None) -> ExpectationsResults:
    """Load saved results from stage1."""
    output_dir = output_dir or OUTPUT_DIR

    # Load trace
    trace = az.from_netcdf(output_dir / f"expectations_{model_type}_trace.nc")

    # Load metadata
    with open(output_dir / f"expectations_{model_type}_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return ExpectationsResults(
        trace=trace,
        measures=metadata["measures"],
        inflation=metadata["inflation"],
        index=metadata["index"],
        model_type=model_type,
    )


def load_all_results(output_dir: Path | None = None) -> dict[str, ExpectationsResults]:
    """Load all three model results."""
    return {
        model_type: load_results(model_type, output_dir)
        for model_type in MODEL_TYPES
    }


# --- Diagnostics ---


def run_diagnostics(results: ExpectationsResults, verbose: bool = True) -> None:
    """Run and print diagnostics for a model."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Diagnostics: {MODEL_NAMES[results.model_type]}")
        print("=" * 60)

        check_model_diagnostics(results.trace)

        # Parameter estimates
        trace_vars = list(results.trace.posterior.data_vars)
        summary_vars = []
        for var in ["alpha", "lambda_bias", "sigma_obs", "sigma_inflation", "sigma_headline",
                    "sigma_early", "sigma_late"]:
            if var in trace_vars:
                summary_vars.append(var)

        if summary_vars:
            print(f"\nParameter Estimates:")
            print(az.summary(results.trace, var_names=summary_vars))
        else:
            print(f"\nNo sampled parameters (all fixed).")

        # Latest values
        hdi = results.expectations_hdi()
        print(f"\nLatest Expectations (90% HDI):")
        print(hdi.tail(4))


# --- Plotting ---


def generate_plots(
    all_results: dict[str, ExpectationsResults],
    chart_dir: Path | None = None,
) -> None:
    """Generate all analysis plots."""
    chart_dir = chart_dir or CHART_DIR
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    results_target = all_results["target"]
    results_short = all_results["short"]
    results_market = all_results["market"]

    # Prepare overlay data
    trimmed = get_trimmed_mean_annual().data
    trimmed = trimmed[trimmed.index >= results_target.index[0]]
    trimmed.name = "Trimmed Mean Inflation"

    market_1y = results_target.measures["market_1y"].dropna()
    market_1y.name = "Market Economists (1yr)"
    breakeven_series = results_target.measures["breakeven"].dropna()
    breakeven_series.name = "Breakeven (10yr)"

    plot_kwargs = {
        "axhspan": {"ymin": 2, "ymax": 3, "color": "red", "alpha": 0.1, "zorder": -1},
        "axhline": {"y": 2.5, "color": "black", "linestyle": "dashed", "linewidth": 0.75},
        "legend": {"loc": "best", "fontsize": "x-small"},
    }

    # Get posteriors
    posterior_target = results_target.expectations_posterior()
    posterior_short = results_short.expectations_posterior()
    posterior_market = results_market.expectations_posterior()

    # Chart 1: Target-anchored
    ax = plot_posterior_timeseries(data=posterior_target, legend_stem="Target Anchored", finalise=False)
    mg.line_plot(trimmed, ax=ax, color="darkorange", width=2, annotate=False, zorder=5)
    mg.finalise_plot(
        ax,
        title="Target Anchored Inflation Expectations",
        lfooter="Australia. Model: market_1y + breakeven + anchor.",
        rfooter=f"Sample: {results_target.index[0]} to {results_target.index[-1]}",
        **plot_kwargs,
    )

    # Chart 2: Short-run
    ax = plot_posterior_timeseries(data=posterior_short, legend_stem="Short Run", finalise=False)
    mg.line_plot(market_1y, ax=ax, color="darkorange", width=1.5, annotate=False, zorder=5)
    mg.line_plot(trimmed, ax=ax, color="brown", width=1.5, annotate=False, zorder=4)
    mg.finalise_plot(
        ax,
        title="Short Run Inflation Expectations (1 Year)",
        lfooter="Australia. Model: market_1y only, no anchor.",
        rfooter=f"Sample: {results_short.index[0]} to {results_short.index[-1]}",
        **plot_kwargs,
    )

    # Chart 3: Long-run (market)
    ax = plot_posterior_timeseries(data=posterior_market, legend_stem="Long Run", finalise=False)
    mg.line_plot(breakeven_series, ax=ax, color="darkorange", width=1.5, annotate=False, zorder=5)
    mg.line_plot(trimmed, ax=ax, color="brown", width=1.5, annotate=False, zorder=4)
    mg.finalise_plot(
        ax,
        title="Long Run Inflation Expectations (10-Year Bond Informed)",
        lfooter="Australia. Model: breakeven only, no anchor.",
        rfooter=f"Sample: {results_market.index[0]} to {results_market.index[-1]}",
        **plot_kwargs,
    )

    # Chart 4: All three with distributions
    ax = plot_posterior_timeseries(data=posterior_target, legend_stem="Target Anchored",
                                   color="steelblue", finalise=False)
    ax = plot_posterior_timeseries(data=posterior_short, legend_stem="Short Run (1yr)",
                                   color="darkorange", ax=ax, finalise=False)
    ax = plot_posterior_timeseries(data=posterior_market, legend_stem="Long Run (10yr)",
                                   color="darkgreen", ax=ax, finalise=False)
    mg.finalise_plot(
        ax,
        title="Inflation Expectations: Three Measures",
        lfooter="Australia. Blue=target anchored, orange=short run (1yr), green=long run (10yr bond).",
        rfooter=f"Sample: {results_target.index[0]} to {results_target.index[-1]}",
        **plot_kwargs,
    )

    # Chart 5: Medians comparison
    median_target = posterior_target.median(axis=1)
    median_target.name = "Target Anchored"
    median_short = posterior_short.median(axis=1)
    median_short.name = "Short Run (1yr)"
    median_market = posterior_market.median(axis=1)
    median_market.name = "Long Run (10yr)"

    ax = mg.line_plot(median_target, color="steelblue", width=2, annotate=False)
    mg.line_plot(median_short, ax=ax, color="darkorange", width=2, annotate=False)
    mg.line_plot(median_market, ax=ax, color="darkgreen", width=2, annotate=False)
    mg.finalise_plot(
        ax,
        title="Inflation Expectations: Median Comparison",
        lfooter="Australia. Blue=target anchored, orange=short run (1yr), green=long run (10yr bond).",
        rfooter=f"Sample: {results_target.index[0]} to {results_target.index[-1]}",
        **plot_kwargs,
    )

    print(f"\nCharts saved to: {chart_dir}")


# --- CLI ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expectations Stage 2: Diagnostics & Plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    print("=" * 60)
    print("EXPECTATIONS STAGE 2: DIAGNOSTICS & PLOTS")
    print("=" * 60)

    # Load all results
    print("\nLoading saved results...")
    all_results = load_all_results()

    # Run diagnostics
    for model_type, results in all_results.items():
        run_diagnostics(results)

    # Generate plots
    if not args.no_plots:
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        generate_plots(all_results)

    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)
