"""NAIRU + Output Gap Stage 2: Analysis and plotting.

This module handles:
- Loading saved results from Stage 1
- Running diagnostics and hypothesis tests
- Generating all analysis plots
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import mgplot as mg
import numpy as np
import pandas as pd
import pymc as pm

from src.analysis import (
    check_for_zero_coeffs,
    check_model_diagnostics,
    decompose_hcoe_inflation,
    decompose_inflation,
    decompose_wage_inflation,
    get_scalar_var,
    get_vector_var,
    plot_equilibrium_rates,
    plot_gdp_vs_potential,
    plot_hcoe_decomposition,
    plot_inflation_decomposition,
    plot_nairu,
    plot_obs_grid,
    plot_output_gap,
    plot_phillips_curve_slope,
    plot_phillips_curves,
    plot_posteriors_bar,
    plot_posteriors_kde,
    plot_potential_growth,
    plot_taylor_rule,
    plot_unemployment_gap,
    plot_wage_decomposition,
    posterior_predictive_checks,
    residual_autocorrelation_analysis,
)
from src.analysis.plot_productivity import (
    plot_labour_productivity,
    plot_mfp,
    plot_productivity_comparison,
)
from src.data import get_cash_rate_monthly
from src.models.nairu_output_gap_stage1 import build_model

# --- Constants ---

MODEL_NAME = "Joint NAIRU + Output Gap Model"
RFOOTER_OUTPUT = "Joint NAIRU + Output Gap Model"

# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "model_outputs"
DEFAULT_CHART_DIR = Path(__file__).parent.parent.parent / "charts" / "nairu_output_gap"


# --- Results Container ---


@dataclass
class NAIRUResults:
    """Results from NAIRU + Output Gap estimation."""

    trace: az.InferenceData
    obs: dict[str, np.ndarray]
    obs_index: pd.PeriodIndex
    model: pm.Model

    def nairu_posterior(self) -> pd.DataFrame:
        """Extract NAIRU posterior as DataFrame."""
        samples = get_vector_var("nairu", self.trace)
        samples.index = self.obs_index
        return samples

    def potential_posterior(self) -> pd.DataFrame:
        """Extract potential output posterior as DataFrame."""
        samples = get_vector_var("potential_output", self.trace)
        samples.index = self.obs_index
        return samples

    def nairu_median(self) -> pd.Series:
        """NAIRU point estimate (posterior median)."""
        return self.nairu_posterior().median(axis=1)

    def potential_median(self) -> pd.Series:
        """Potential output point estimate (posterior median)."""
        return self.potential_posterior().median(axis=1)

    def unemployment_gap(self) -> pd.Series:
        """Unemployment gap = U - NAIRU."""
        U = pd.Series(self.obs["U"], index=self.obs_index)
        return U - self.nairu_median()

    def output_gap(self) -> pd.Series:
        """Output gap = log(GDP) - log(potential)."""
        log_gdp = pd.Series(self.obs["log_gdp"], index=self.obs_index)
        return log_gdp - self.potential_median()


# --- Load Functions ---


def load_results(
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
) -> tuple[az.InferenceData, dict[str, np.ndarray], pd.PeriodIndex]:
    """Load model results from disk.

    Args:
        output_dir: Directory containing saved results
        prefix: Filename prefix used when saving

    Returns:
        Tuple of (trace, obs, obs_index)

    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)

    # Load trace
    trace_path = output_dir / f"{prefix}_trace.nc"
    trace = az.from_netcdf(str(trace_path))
    print(f"Loaded trace from: {trace_path}")

    # Load observations
    obs_path = output_dir / f"{prefix}_obs.pkl"
    with open(obs_path, "rb") as f:
        data = pickle.load(f)
    obs = data["obs"]
    obs_index = data["obs_index"]
    print(f"Loaded observations from: {obs_path}")
    print(f"  Period: {obs_index.min()} to {obs_index.max()} ({len(obs_index)} periods)")

    return trace, obs, obs_index


# --- Analysis Functions ---


def test_theoretical_expectations(trace: az.InferenceData) -> pd.DataFrame:
    """Test whether parameters match theoretical expectations.

    For parameters expected to equal a value (α≈0.3), we test:
        - Probability that parameter differs from expected value
        - Whether the expected value falls within the 90% HDI

    For parameters expected to have a sign (β<0, γ<0), we test:
        - Probability that parameter has the expected sign
    """
    results = []

    # Define tests: (parameter, expected_value or 'negative'/'positive'/(low,high), description)
    tests = [
        ("alpha_capital", (0.20, 0.35), "Capital share ∈ (0.20, 0.35)"),
        ("rho_potential", "between_0_1", "Potential growth persistence ∈ (0,1)"),
        ("beta_okun", "negative", "Okun coefficient < 0"),
        # Regime-specific Phillips curve slopes (all should be negative)
        ("gamma_pi_pre_gfc", "negative", "Price Phillips (pre-GFC) < 0"),
        ("gamma_pi_gfc", "negative", "Price Phillips (post-GFC) < 0"),
        ("gamma_pi_covid", "negative", "Price Phillips (post-COVID) < 0"),
        ("gamma_wg_pre_gfc", "negative", "Wage Phillips (pre-GFC) < 0"),
        ("gamma_wg_gfc", "negative", "Wage Phillips (post-GFC) < 0"),
        ("gamma_wg_covid", "negative", "Wage Phillips (post-COVID) < 0"),
        ("rho_wg", "between_0_1", "Wage persistence ∈ (0,1)"),
        ("phi_wg", "positive", "Demand deflator → wages > 0"),
        ("theta_wg", "positive", "Trend expectations → wages > 0"),
        # Hourly COE Phillips curve (regime-specific slopes, all negative)
        ("gamma_hcoe_pre_gfc", "negative", "Hourly COE Phillips (pre-GFC) < 0"),
        ("gamma_hcoe_gfc", "negative", "Hourly COE Phillips (post-GFC) < 0"),
        ("gamma_hcoe_covid", "negative", "Hourly COE Phillips (post-COVID) < 0"),
        ("rho_hcoe", "between_0_1", "Hourly COE persistence ∈ (0,1)"),
        ("phi_hcoe", "positive", "Hourly COE demand deflator > 0"),
        ("theta_hcoe", "positive", "Hourly COE trend expectations > 0"),
        ("beta_is", "positive", "IS interest rate effect > 0"),
        ("rho_is", "between_0_1", "IS persistence ∈ (0,1)"),
        # Participation rate equation
        ("beta_pr", "negative", "Discouraged worker effect < 0"),
        # Exchange rate equation
        ("beta_er_r", "positive", "ER interest rate effect > 0 (UIP)"),
        ("rho_er", "between_0_1", "ER persistence ∈ (0,1)"),
        # Import price pass-through
        ("beta_pt", "negative", "Pass-through < 0"),
        ("beta_oil", "positive", "Oil effect on import prices > 0"),
        ("rho_pt", "between_0_1", "Import price persistence ∈ (0,1)"),
    ]

    for param, expected, description in tests:
        try:
            samples = get_scalar_var(param, trace).values
        except KeyError:
            # Parameter not in model (e.g., equation not included)
            continue

        median = np.median(samples)
        hdi_90 = az.hdi(samples, hdi_prob=0.90)

        if isinstance(expected, tuple):
            # Test for value within range (low, high)
            low, high = expected
            prob_in_range = np.mean((samples >= low) & (samples <= high))

            results.append({
                "Parameter": param,
                "Hypothesis": description,
                "Median": f"{median:.3f}",
                "90% HDI": f"[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]",
                "Expected in HDI": "-",
                f"P({low} ≤ θ ≤ {high})": f"{prob_in_range:.1%}",
                "Result": "PASS" if prob_in_range > 0.90 else ("WEAK" if prob_in_range > 0.50 else "FAIL")
            })
        elif isinstance(expected, (int, float)):
            # Test for equality to expected value
            in_hdi = hdi_90[0] <= expected <= hdi_90[1]
            prob_above = np.mean(samples > expected)

            results.append({
                "Parameter": param,
                "Hypothesis": description,
                "Median": f"{median:.3f}",
                "90% HDI": f"[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]",
                "Expected in HDI": "✓" if in_hdi else "✗",
                "P(θ > expected)": f"{prob_above:.1%}",
                "Result": "PASS" if in_hdi else "FAIL"
            })
        elif expected == "between_0_1":
            # Test for value between 0 and 1 (stable persistence)
            prob_valid = np.mean((samples > 0) & (samples < 1))

            results.append({
                "Parameter": param,
                "Hypothesis": description,
                "Median": f"{median:.3f}",
                "90% HDI": f"[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]",
                "Expected in HDI": "-",
                "P(0 < θ < 1)": f"{prob_valid:.1%}",
                "Result": "PASS" if prob_valid > 0.99 else ("WEAK" if prob_valid > 0.90 else "FAIL")
            })
        else:
            # Test for sign
            if expected == "negative":
                prob_correct = np.mean(samples < 0)
            else:  # positive
                prob_correct = np.mean(samples > 0)

            results.append({
                "Parameter": param,
                "Hypothesis": description,
                "Median": f"{median:.3f}",
                "90% HDI": f"[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]",
                "Expected in HDI": "-",
                "P(correct sign)": f"{prob_correct:.1%}",
                "Result": "PASS" if prob_correct > 0.99 else ("WEAK" if prob_correct > 0.90 else "FAIL")
            })

    df = pd.DataFrame(results)
    return df


def plot_all(
    results: NAIRUResults,
    inflation_annual: pd.Series | None = None,
    cash_rate_monthly: pd.Series | None = None,
    show: bool = False,
) -> None:
    """Generate all standard plots."""
    plot_nairu(results, show=show)
    plot_unemployment_gap(results, show=show)
    plot_output_gap(results, show=show)
    plot_gdp_vs_potential(results, show=show)
    plot_potential_growth(results, show=show)
    if cash_rate_monthly is not None and inflation_annual is not None:
        plot_taylor_rule(results, inflation_annual, cash_rate_monthly, show=show)
        plot_equilibrium_rates(results, cash_rate_monthly, show=show)


# --- Main Entry Point ---


def run_stage2(
    output_dir: Path | str | None = None,
    chart_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
    verbose: bool = False,
    show_plots: bool = False,
) -> NAIRUResults:
    """Run Stage 2: Load results and perform all analysis/plotting.

    Args:
        output_dir: Directory containing saved results from Stage 1
        chart_dir: Directory to save charts
        prefix: Filename prefix used when saving
        verbose: Print detailed output
        show_plots: Display plots interactively

    Returns:
        NAIRUResults container

    """
    # Set output directory for charts
    if chart_dir is None:
        chart_dir = DEFAULT_CHART_DIR
    chart_dir = Path(chart_dir)
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    print("Running NAIRU + Output Gap analysis (Stage 2)...\n")

    # Load results
    trace, obs, obs_index = load_results(output_dir=output_dir, prefix=prefix)

    # Rebuild model (needed for posterior predictive checks)
    model = build_model(obs)

    # Create results container
    results = NAIRUResults(
        trace=trace,
        obs=obs,
        obs_index=obs_index,
        model=model,
    )

    # Plot observation grid
    plot_obs_grid(obs, obs_index)

    # Diagnostics
    check_model_diagnostics(results.trace)

    # Check for zero coefficients
    zero_check = check_for_zero_coeffs(
        results.trace,
        critical_params=[
            "gamma_pi_pre_gfc", "gamma_pi_gfc", "gamma_pi_covid",
            "gamma_wg_pre_gfc", "gamma_wg_gfc", "gamma_wg_covid",
            "gamma_hcoe_pre_gfc", "gamma_hcoe_gfc", "gamma_hcoe_covid",
            "beta_okun",
        ]
    )
    if verbose:
        print(zero_check.T)

    # Plot scalar posteriors (bar and KDE)
    plot_posteriors_bar(
        results.trace,
        model_name=MODEL_NAME,
        show=show_plots,
    )
    plot_posteriors_kde(
        results.trace,
        model_name=MODEL_NAME,
        show=show_plots,
    )

    # Posterior predictive checks and residual analysis
    obs_vars = {
        "okun_law": obs["ΔU"],
        "observed_price_inflation": obs["π"],
        "observed_wage_growth": obs["Δulc"],
        "observed_hourly_coe": obs["Δhcoe"],
        "observed_twi_change": obs["Δtwi"],
        "observed_import_price": obs["Δ4ρm"],
    }
    var_labels = {
        "okun_law": "Change in Unemployment (pp)",
        "observed_price_inflation": "Quarterly Inflation (%)",
        "observed_wage_growth": "Unit Labour Cost Growth (%)",
        "observed_hourly_coe": "Hourly COE Growth (%)",
        "observed_twi_change": "TWI Change (%)",
        "observed_import_price": "Import Price Growth (%)",
    }

    ppc_data = posterior_predictive_checks(
        trace=results.trace,
        model=model,
        obs_vars=obs_vars,
        obs_index=obs_index,
        var_labels=var_labels,
        model_name=MODEL_NAME,
        rfooter=RFOOTER_OUTPUT,
        show=show_plots,
    )

    residual_autocorrelation_analysis(
        ppc=ppc_data,
        obs_vars=obs_vars,
        obs_index=obs_index,
        var_labels=var_labels,
        model_name=MODEL_NAME,
        rfooter=RFOOTER_OUTPUT,
        show=show_plots,
    )

    # Theoretical expectations tests
    hypothesis_results = test_theoretical_expectations(results.trace)
    print(hypothesis_results.to_string(index=False))

    # Print summary
    if verbose:
        print("\nRecent NAIRU estimates:")
        nairu = results.nairu_median()
        U = pd.Series(results.obs["U"], index=results.obs_index)
        summary = pd.DataFrame({
            "NAIRU": nairu,
            "U": U,
            "U_gap": U - nairu,
        })
        print(summary.tail(8).round(2))

        print("\nRecent output gap:")
        print(results.output_gap().tail(8).round(2))

    # Get cash rate and inflation data for Taylor rule plots
    cash_rate_monthly = get_cash_rate_monthly().data
    π4 = pd.Series(results.obs["π4"], index=results.obs_index)

    # Generate all plots
    plot_all(
        results,
        inflation_annual=π4,
        cash_rate_monthly=cash_rate_monthly,
        show=show_plots,
    )

    # Phillips curve plots
    plot_phillips_curves(
        results.trace,
        results.obs,
        results.obs_index,
        model_name=MODEL_NAME,
        show=show_plots,
    )
    plot_phillips_curve_slope(
        results.trace,
        results.obs_index,
        curve_type="price",
        model_name=MODEL_NAME,
        show=show_plots,
    )
    plot_phillips_curve_slope(
        results.trace,
        results.obs_index,
        curve_type="wage",
        model_name=MODEL_NAME,
        show=show_plots,
    )
    plot_phillips_curve_slope(
        results.trace,
        results.obs_index,
        curve_type="hcoe",
        model_name=MODEL_NAME,
        show=show_plots,
    )

    # Price inflation decomposition (demand vs supply)
    decomp = decompose_inflation(results.trace, results.obs, results.obs_index)
    plot_inflation_decomposition(decomp, rfooter=RFOOTER_OUTPUT)

    # Wage-ULC inflation decomposition (demand vs price pass-through)
    wage_decomp = decompose_wage_inflation(results.trace, results.obs, results.obs_index)
    plot_wage_decomposition(wage_decomp, rfooter=RFOOTER_OUTPUT)

    # Wage-HCOE inflation decomposition (demand component)
    hcoe_decomp = decompose_hcoe_inflation(results.trace, results.obs, results.obs_index)
    plot_hcoe_decomposition(hcoe_decomp, rfooter=RFOOTER_OUTPUT)

    # Derived productivity plots (LP and MFP from wage data)
    ulc_growth = pd.Series(results.obs["Δulc"], index=results.obs_index)
    hcoe_growth = pd.Series(results.obs["Δhcoe"], index=results.obs_index)
    capital_growth = pd.Series(results.obs["capital_growth"], index=results.obs_index)
    hours_growth = pd.Series(results.obs["hours_growth"], index=results.obs_index)

    plot_labour_productivity(
        ulc_growth=ulc_growth,
        hcoe_growth=hcoe_growth,
        filter_type="henderson",
        model_name=MODEL_NAME,
        show=show_plots,
    )
    plot_labour_productivity(
        ulc_growth=ulc_growth,
        hcoe_growth=hcoe_growth,
        filter_type="hp",
        model_name=MODEL_NAME,
        show=show_plots,
    )
    plot_mfp(
        ulc_growth=ulc_growth,
        hcoe_growth=hcoe_growth,
        capital_growth=capital_growth,
        hours_growth=hours_growth,
        filter_type="henderson",
        model_name=MODEL_NAME,
        show=show_plots,
    )
    plot_mfp(
        ulc_growth=ulc_growth,
        hcoe_growth=hcoe_growth,
        capital_growth=capital_growth,
        hours_growth=hours_growth,
        filter_type="hp",
        model_name=MODEL_NAME,
        show=show_plots,
    )
    plot_productivity_comparison(
        ulc_growth=ulc_growth,
        hcoe_growth=hcoe_growth,
        capital_growth=capital_growth,
        hours_growth=hours_growth,
        filter_type="henderson",
        model_name=MODEL_NAME,
        show=show_plots,
    )
    plot_productivity_comparison(
        ulc_growth=ulc_growth,
        hcoe_growth=hcoe_growth,
        capital_growth=capital_growth,
        hours_growth=hours_growth,
        filter_type="hp",
        model_name=MODEL_NAME,
        show=show_plots,
    )

    print(f"\nCharts saved to: {chart_dir}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NAIRU + Output Gap analysis (Stage 2)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory with saved results")
    parser.add_argument("--chart-dir", type=str, default=None, help="Directory to save charts")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    run_stage2(
        output_dir=args.output_dir,
        chart_dir=args.chart_dir,
        verbose=args.verbose,
        show_plots=args.show,
    )
