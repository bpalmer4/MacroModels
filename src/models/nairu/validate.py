"""Model validation: diagnostics, hypothesis tests, and diagnostic charts.

Diagnostic charts are saved to charts/{variant}/validation/.

This module handles:
- MCMC diagnostics (R-hat, ESS, divergences)
- Zero coefficient checks
- Posterior distribution plots (bar and KDE)
- Posterior predictive checks (requires model rebuild)
- Residual autocorrelation analysis
- Theoretical expectations tests
"""

from pathlib import Path

import arviz as az
import mgplot as mg
import numpy as np
import pandas as pd

from src.models.common.diagnostics import check_for_zero_coeffs, check_model_diagnostics
from src.models.common.extraction import get_scalar_var
from src.models.nairu.analysis.plot_obs_grid import plot_obs_grid
from src.models.nairu.analysis.plot_posteriors_bar import plot_posteriors_bar
from src.models.nairu.analysis.plot_posteriors_kde import plot_posteriors_kde
from src.models.nairu.analysis.posterior_predictive import posterior_predictive_checks
from src.models.nairu.analysis.residual_autocorrelation import residual_autocorrelation_analysis
from src.models.nairu.config import ModelConfig
from src.models.nairu.results import DEFAULT_CHART_BASE, NAIRUResults, load_results

# --- Validation thresholds ---

PASS_THRESHOLD = 0.90       # probability threshold for PASS
WEAK_THRESHOLD = 0.50       # probability threshold for WEAK (below = FAIL)
STRONG_PASS = 0.99          # stricter threshold for sign/range tests
NEAR_ZERO_THRESHOLD = 0.01  # absolute value below which a sample is "near zero"
BUNCHED_THRESHOLD = 0.20    # fraction near zero that triggers BUNCHED warning

# --- Theoretical Expectations ---


# Master list of all parameter tests.
# Each test is (parameter_name, expected, description).
# Tests for parameters not in the trace are automatically skipped.
THEORETICAL_TESTS = [
    # Okun's Law - simple form
    ("beta_okun", "negative", "Okun coefficient < 0"),
    # Okun's Law - gap-to-gap form (truncated)
    ("tau1_okun", "nonzero", "Okun output gap effect != 0"),
    ("tau2_okun", "between_0_1", "Okun persistence in (0,1)"),
    # Price Phillips curve - single slope (NOT truncated)
    ("gamma_pi", "negative", "Price Phillips slope < 0"),
    # Price Phillips curve - regime-switching (NOT truncated)
    ("gamma_pi_pre_gfc", "negative", "Price Phillips (pre-GFC) < 0"),
    ("gamma_pi_gfc", "negative", "Price Phillips (post-GFC) < 0"),
    ("gamma_pi_covid", "negative", "Price Phillips (post-COVID) < 0"),
    # Wage Phillips curve - single slope (truncated upper=0)
    ("gamma_wg", "nonzero", "Wage Phillips slope != 0"),
    # Wage Phillips curve - regime-switching (truncated upper=0)
    ("gamma_wg_pre_gfc", "nonzero", "Wage Phillips (pre-GFC) != 0"),
    ("gamma_wg_gfc", "nonzero", "Wage Phillips (post-GFC) != 0"),
    ("gamma_wg_covid", "nonzero", "Wage Phillips (post-COVID) != 0"),
    ("phi_wg", "positive", "Demand deflator -> wages > 0"),
    # Hourly COE Phillips curve - single slope (truncated)
    ("gamma_hcoe", "nonzero", "Hourly COE Phillips slope != 0"),
    # Hourly COE Phillips curve - regime-switching (truncated)
    ("gamma_hcoe_pre_gfc", "nonzero", "Hourly COE Phillips (pre-GFC) != 0"),
    ("gamma_hcoe_gfc", "nonzero", "Hourly COE Phillips (post-GFC) != 0"),
    ("gamma_hcoe_covid", "nonzero", "Hourly COE Phillips (post-COVID) != 0"),
    ("phi_hcoe", "nonzero", "Hourly COE demand deflator != 0"),
    ("psi_hcoe", "nonzero", "Productivity -> hourly wages != 0"),
    # IS curve (truncated lower=0)
    ("beta_is", "nonzero", "IS interest rate effect != 0"),
    ("delta_dsr", "nonzero", "Debt servicing effect != 0"),
    ("eta_hw", "nonzero", "Housing wealth effect != 0"),
    ("gamma_fi", "nonzero", "Fiscal impulse effect != 0"),
    ("rho_is", "between_0_1", "IS persistence in (0,1)"),
    # Participation rate equation (truncated upper=0)
    ("beta_pr", "nonzero", "Discouraged worker effect != 0"),
    # Exchange rate equation (truncated lower=0)
    ("beta_er_r", "nonzero", "ER interest rate effect != 0"),
    ("rho_er", "between_0_1", "ER persistence in (0,1)"),
    # Import price pass-through (truncated)
    ("beta_pt", "nonzero", "Pass-through != 0"),
    ("beta_oil", "nonzero", "Oil effect on import prices != 0"),
    ("rho_pt", "between_0_1", "Import price persistence in (0,1)"),
    # Import price -> inflation (not truncated, test for positive)
    ("rho_pi", "positive", "Import prices -> inflation > 0"),
    # GSCPI control
    ("xi_gscpi", "positive", "GSCPI -> inflation > 0"),
    # Employment equation (truncated)
    ("beta_emp_ygap", "nonzero", "Output gap -> employment != 0"),
    ("beta_emp_wage", "nonzero", "Real wage gap -> employment != 0"),
    # Net exports equation (truncated upper=0)
    ("beta_nx_ygap", "nonzero", "Output gap -> net exports != 0"),
    ("beta_nx_twi", "nonzero", "TWI appreciation -> net exports != 0"),
]


def test_theoretical_expectations(trace: az.InferenceData) -> pd.DataFrame:
    """Test whether parameters match theoretical expectations.

    Tests are automatically skipped for parameters not present in the trace.

    Returns:
        DataFrame with test results (one row per tested parameter)

    """
    results = []

    for param, expected, description in THEORETICAL_TESTS:
        try:
            samples = get_scalar_var(param, trace).to_numpy()
        except KeyError:
            continue

        median = np.median(samples)
        hdi_90 = az.hdi(samples, hdi_prob=0.90)

        if isinstance(expected, tuple):
            low, high = expected
            prob_in_range = np.mean((samples >= low) & (samples <= high))
            results.append({
                "Parameter": param,
                "Hypothesis": description,
                "Median": f"{median:.3f}",
                "90% HDI": f"[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]",
                "P(in range)": f"{prob_in_range:.1%}",
                "Result": ("PASS" if prob_in_range > PASS_THRESHOLD
                           else ("*WEAK*" if prob_in_range > WEAK_THRESHOLD else "*FAIL*")),
            })
        elif isinstance(expected, (int, float)):
            in_hdi = hdi_90[0] <= expected <= hdi_90[1]
            prob_above = np.mean(samples > expected)
            results.append({
                "Parameter": param,
                "Hypothesis": description,
                "Median": f"{median:.3f}",
                "90% HDI": f"[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]",
                "In HDI": "Y" if in_hdi else "N",
                "P(above)": f"{prob_above:.1%}",
                "Result": "PASS" if in_hdi else "*FAIL*",
            })
        elif expected == "between_0_1":
            prob_valid = np.mean((samples > 0) & (samples < 1))
            results.append({
                "Parameter": param,
                "Hypothesis": description,
                "Median": f"{median:.3f}",
                "90% HDI": f"[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]",
                "P(0<t<1)": f"{prob_valid:.1%}",
                "Result": ("PASS" if prob_valid > STRONG_PASS
                           else ("*WEAK*" if prob_valid > PASS_THRESHOLD else "*FAIL*")),
            })
        elif expected == "nonzero":
            hdi_excludes_zero = (hdi_90[0] > 0) or (hdi_90[1] < 0)
            p_near_zero = np.mean(np.abs(samples) < NEAR_ZERO_THRESHOLD)
            is_bunched = p_near_zero > BUNCHED_THRESHOLD

            if is_bunched:
                result = "*BUNCHED*"
            elif not hdi_excludes_zero:
                result = "*WEAK*"
            else:
                result = "PASS"

            results.append({
                "Parameter": param,
                "Hypothesis": description,
                "Median": f"{median:.3f}",
                "90% HDI": f"[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]",
                "P(|t|<.01)": f"{p_near_zero:.0%}",
                "Result": result,
            })
        else:
            # Sign test
            prob_correct = np.mean(samples < 0) if expected == "negative" else np.mean(samples > 0)

            results.append({
                "Parameter": param,
                "Hypothesis": description,
                "Median": f"{median:.3f}",
                "90% HDI": f"[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]",
                "P(sign)": f"{prob_correct:.1%}",
                "Result": ("PASS" if prob_correct > STRONG_PASS
                           else ("*WEAK*" if prob_correct > PASS_THRESHOLD else "*FAIL*")),
            })

    return pd.DataFrame(results)


# --- Config-driven obs_vars mapping ---


def _build_obs_vars(
    obs: dict[str, np.ndarray],
    config: ModelConfig,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Build obs_vars and var_labels from config (no trace sniffing).

    Returns:
        (obs_vars, var_labels) dicts keyed by PyMC observed variable name

    """
    obs_vars: dict[str, np.ndarray] = {}
    var_labels: dict[str, str] = {}

    # Price Phillips curve (always included)
    obs_vars["observed_price_inflation"] = obs["π"]
    var_labels["observed_price_inflation"] = "Quarterly Inflation (%)"

    # Okun's Law
    if config.include_okun:
        if config.okun_gap_form:
            obs_vars["okun"] = obs["U"]
            var_labels["okun"] = "Unemployment Rate (%)"
        else:
            obs_vars["okun_law"] = obs["ΔU"]
            var_labels["okun_law"] = "Change in Unemployment (pp)"

    # Wage growth (ULC)
    if config.include_wage_growth:
        obs_vars["observed_wage_growth"] = obs["Δulc"]
        var_labels["observed_wage_growth"] = "Unit Labour Cost Growth (%)"

    # Hourly COE
    if config.include_hourly_coe:
        obs_vars["observed_hourly_coe"] = obs["Δhcoe"]
        var_labels["observed_hourly_coe"] = "Hourly COE Growth (%)"

    # Employment
    if config.include_employment:
        obs_vars["observed_employment"] = obs["emp_growth"]
        var_labels["observed_employment"] = "Employment Growth (%)"

    # Exchange rate
    if config.include_exchange_rate:
        obs_vars["observed_twi_change"] = obs["Δtwi"]
        var_labels["observed_twi_change"] = "TWI Change (%)"

    # Import price
    if config.include_import_price:
        obs_vars["observed_import_price"] = obs["Δ4ρm"]
        var_labels["observed_import_price"] = "Import Price Growth (%)"

    # Net exports
    if config.include_net_exports:
        obs_vars["observed_net_exports"] = obs["Δnx_ratio"]
        var_labels["observed_net_exports"] = "Change in NX/GDP (pp)"

    return obs_vars, var_labels


# --- Critical parameters for zero-coefficient check ---


def _build_critical_params(config: ModelConfig) -> list[str]:  # noqa: PLR0912 — flat feature-flag list
    """Build critical parameter list from config (no trace sniffing)."""
    params: list[str] = []

    # Okun
    if config.include_okun:
        if config.okun_gap_form:
            params += ["tau1_okun", "tau2_okun"]
        else:
            params += ["beta_okun"]

    # Price Phillips
    if config.regime_switching:
        params += ["gamma_pi_pre_gfc", "gamma_pi_gfc", "gamma_pi_covid"]
    else:
        params += ["gamma_pi"]

    # Wage Phillips
    if config.include_wage_growth:
        if config.regime_switching:
            params += ["gamma_wg_pre_gfc", "gamma_wg_gfc", "gamma_wg_covid"]
        else:
            params += ["gamma_wg"]

    # Hourly COE
    if config.include_hourly_coe:
        if config.regime_switching:
            params += ["gamma_hcoe_pre_gfc", "gamma_hcoe_gfc", "gamma_hcoe_covid"]
        else:
            params += ["gamma_hcoe"]

    # Employment
    if config.include_employment:
        params += ["beta_emp_ygap", "beta_emp_wage"]

    # Net exports
    if config.include_net_exports:
        params += ["beta_nx_ygap", "beta_nx_twi"]

    return params


# --- Main validation entry point ---


def run_validate(
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
    chart_dir: Path | str | None = None,
    verbose: bool = False,
    show_plots: bool = False,
) -> NAIRUResults:
    """Run model validation: diagnostics, hypothesis tests, diagnostic charts.

    Charts are saved to charts/{variant}/validation/.

    Args:
        output_dir: Directory containing saved results
        prefix: Filename prefix
        chart_dir: Override chart directory (default: charts/{variant})
        verbose: Print detailed output
        show_plots: Display plots interactively

    Returns:
        NAIRUResults container (with model rebuilt for PPC)

    """
    # Load results with model rebuild (needed for PPC)
    results = load_results(output_dir=output_dir, prefix=prefix, rebuild_model=True)
    config = results.config

    # Set chart directory to validation subdirectory
    if chart_dir is None:
        chart_dir = DEFAULT_CHART_BASE / config.chart_dir_name
    chart_dir = Path(chart_dir)
    validation_dir = chart_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    mg.set_chart_dir(str(validation_dir))
    mg.clear_chart_dir()

    rfooter = config.rfooter

    print(f"Running validation [{config.label}]...\n")

    # --- 1. Input data overview ---
    plot_obs_grid(results.obs, results.obs_index)

    # --- 2. MCMC diagnostics ---
    check_model_diagnostics(results.trace)

    # --- 3. Zero coefficient check ---
    critical_params = _build_critical_params(config)
    zero_check = check_for_zero_coeffs(results.trace, critical_params=critical_params)
    if verbose:
        print(zero_check.T)

    # --- 4. Posterior distribution plots ---
    plot_posteriors_bar(results.trace, rfooter=rfooter, show=show_plots)
    plot_posteriors_kde(results.trace, rfooter=rfooter, show=show_plots)

    # --- 5. Posterior predictive checks ---
    obs_vars, var_labels = _build_obs_vars(results.obs, config)

    ppc_data = posterior_predictive_checks(
        results.trace,
        results.model,
        obs_vars,
        results.obs_index,
        var_labels=var_labels,
        rfooter=rfooter,
        show=show_plots,
    )

    # --- 6. Residual autocorrelation ---
    skip_autocorr = []
    if config.include_import_price:
        skip_autocorr.append("observed_import_price")

    residual_autocorrelation_analysis(
        ppc_data,
        obs_vars,
        results.obs_index,
        var_labels=var_labels,
        rfooter=rfooter,
        show=show_plots,
        skip_autocorr_warning=skip_autocorr,
    )

    # --- 7. Theoretical expectations tests ---
    hypothesis_results = test_theoretical_expectations(results.trace)
    print(hypothesis_results.to_string(index=False))

    # --- 8. Summary ---
    if verbose:
        print("\nRecent NAIRU estimates:")
        nairu = results.nairu_median()
        U = pd.Series(results.obs["U"], index=results.obs_index)
        summary = pd.DataFrame({"NAIRU": nairu, "U": U, "U_gap": U - nairu})
        print(summary.tail(8).round(2))

        print("\nRecent output gap:")
        print(results.output_gap().tail(8).round(2))

    print(f"\nValidation charts saved to: {validation_dir}")

    return results
