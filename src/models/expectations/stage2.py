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
from src.data.rba_loader import get_inflation_expectations
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
    """Load all available model results, skipping missing ones."""
    results = {}
    for model_type in MODEL_TYPES:
        try:
            results[model_type] = load_results(model_type, output_dir)
        except FileNotFoundError:
            print(f"  Skipping {model_type}: no saved results found")
    return results


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

# Common plot styling
PLOT_KWARGS = {
    "axhspan": {"ymin": 2, "ymax": 3, "color": "red", "alpha": 0.1, "zorder": -1},
    "axhline": {"y": 2.5, "color": "black", "linestyle": "dashed", "linewidth": 0.75},
    "legend": {"loc": "best", "fontsize": "x-small"},
}


def _plot_model(
    results: ExpectationsResults,
    title: str,
    lfooter: str,
    legend_stem: str,
    overlays: list[tuple[pd.Series, str]] | None = None,
    axvspan: dict | None = None,  # Empty dict if no span needed
) -> None:
    """Plot a single model's posterior with optional overlays."""
    posterior = results.expectations_posterior()
    ax = plot_posterior_timeseries(data=posterior, legend_stem=legend_stem, finalise=False)

    if overlays:
        for series, color in overlays:
            mg.line_plot(series, ax=ax, color=color, width=1.5, annotate=False, zorder=5)

    mg.finalise_plot(
        ax,
        title=title,
        lfooter=lfooter,
        rfooter=f"Sample: {results.index[0]} to {results.index[-1]}",
        axvspan=axvspan,
        **PLOT_KWARGS,
    )


def generate_plots(
    all_results: dict[str, ExpectationsResults],
    chart_dir: Path | None = None,
) -> None:
    """Generate all analysis plots."""
    chart_dir = chart_dir or CHART_DIR
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    # Use first available result for shared data
    first_result = next(iter(all_results.values()))

    # Prepare overlay data
    trimmed = get_trimmed_mean_annual().data
    trimmed = trimmed[trimmed.index >= first_result.index[0]]
    trimmed.name = "Trimmed Mean Inflation"

    market_1y = first_result.measures["market_1y"].dropna()
    market_1y.name = "Market Economists (1yr)"
    breakeven_series = first_result.measures["breakeven"].dropna()
    breakeven_series.name = "Breakeven (10yr)"

    pie_rbaq = get_inflation_expectations().data
    pie_rbaq = pie_rbaq[pie_rbaq.index >= first_result.index[0]]
    pie_rbaq.name = "RBA PIE_RBAQ"

    # Single-model plot specs: (model_type, title, lfooter, legend_stem, overlays, axvspan_fn)
    plot_specs = [
        ("target", "Target Anchored Inflation Expectations",
         "Australia. Model: market_1y + breakeven + business + market_yoy + anchor.",
         "Target Anchored", [(trimmed, "darkorange")], None),
        ("target", "Target Anchored Inflation Expectations vs RBA PIE_RBAQ",
         "Australia. Model: market_1y + breakeven + business + market_yoy + anchor.",
         "Target Anchored", [(pie_rbaq, "darkorange")], None),
        ("unanchored", "Unanchored Inflation Expectations",
         "Australia. Model: market_1y + breakeven + business + market_yoy (no target anchor).",
         "Unanchored", [(trimmed, "darkorange")], None),
        ("unanchored", "Unanchored Inflation Expectations vs RBA PIE_RBAQ",
         "Australia. Model: market_1y + breakeven + business + market_yoy (no target anchor).",
         "Unanchored", [(pie_rbaq, "darkorange")], None),
        ("short", "Short Run Inflation Expectations (1 Year)",
         "Australia. Bayesian signal extraction from market economist 1-year expectations.",
         "Short Run", [(market_1y, "darkorange"), (trimmed, "brown")],
         lambda r: {"xmin": r.index[0].ordinal, "xmax": market_1y.index[0].ordinal,
                    "color": "goldenrod", "alpha": 0.2, "label": "Proxy period (headline CPI)"}),
        ("market", "Long Run Inflation Expectations (10-Year)",
         "Australia. Bayesian signal extraction from breakeven inflation and nominal bonds.",
         "Long Run", [(breakeven_series, "darkorange"), (trimmed, "brown")],
         lambda r: {"xmin": r.index[0].ordinal, "xmax": breakeven_series.index[0].ordinal,
                    "color": "goldenrod", "alpha": 0.2, "label": "Proxy period (nominal bonds)"}),
    ]

    for model_type, title, lfooter, legend_stem, overlays, axvspan_fn in plot_specs:
        if model_type in all_results:
            results = all_results[model_type]
            axvspan = axvspan_fn(results) if axvspan_fn else None
            _plot_model(results, title, lfooter, legend_stem, overlays, axvspan)

    # Comparison plots (require all three: target, short, market)
    if all(k in all_results for k in ("target", "short", "market")):
        posterior_target = all_results["target"].expectations_posterior()
        posterior_short = all_results["short"].expectations_posterior()
        posterior_market = all_results["market"].expectations_posterior()

        # Three distributions
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
            rfooter=f"Sample: {all_results['target'].index[0]} to {all_results['target'].index[-1]}",
            **PLOT_KWARGS,
        )

        # Medians comparison
        medians = [
            (posterior_target.median(axis=1), "Target Anchored", "steelblue"),
            (posterior_short.median(axis=1), "Short Run (1yr)", "darkorange"),
            (posterior_market.median(axis=1), "Long Run (10yr)", "darkgreen"),
        ]
        ax = None
        for series, name, color in medians:
            series.name = name
            ax = mg.line_plot(series, ax=ax, color=color, width=2, annotate=False)
        mg.finalise_plot(
            ax,
            title="Inflation Expectations: Median Comparison",
            lfooter="Australia. Blue=target anchored, orange=short run (1yr), green=long run (10yr bond).",
            rfooter=f"Sample: {all_results['target'].index[0]} to {all_results['target'].index[-1]}",
            **PLOT_KWARGS,
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
