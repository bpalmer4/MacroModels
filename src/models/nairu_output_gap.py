"""NAIRU + Output Gap joint estimation model.

Bayesian state-space model that jointly estimates:
- NAIRU (Non-Accelerating Inflation Rate of Unemployment)
- Potential output (via Cobb-Douglas production function)
- Output gap and unemployment gap

Equations:
1. NAIRU: Gaussian random walk (state equation)
2. Potential output: Cobb-Douglas with time-varying drift (state equation)
3. Okun's Law: Links output gap to unemployment changes
4. Phillips Curve: Links inflation to unemployment gap
5. Wage Phillips Curve: Links wage growth to unemployment gap
6. IS Curve: Links output gap to real interest rate gap
7. Participation Rate: Links participation to unemployment gap (discouraged worker)
8. Exchange Rate: UIP-style TWI equation linking to interest rate differential
9. Import Price Pass-Through: Links import prices to TWI changes

Data sources:
- ABS: GDP, unemployment, CPI, hours worked, capital stock, MFP, import prices,
       participation rate
- RBA: Cash rate, inflation expectations, Trade-Weighted Index (TWI)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import arviz as az
import mgplot as mg
import numpy as np
import pandas as pd
import pymc as pm

from src.analysis import (
    check_for_zero_coeffs,
    check_model_diagnostics,
    decompose_inflation,
    get_scalar_var,
    get_vector_var,
    plot_equilibrium_rates,
    plot_gdp_vs_potential,
    plot_inflation_decomposition,
    plot_nairu,
    plot_obs_grid,
    plot_output_gap,
    plot_posterior_timeseries,
    plot_posteriors_bar,
    plot_posteriors_kde,
    plot_potential_growth,
    plot_taylor_rule,
    plot_unemployment_gap,
    posterior_predictive_checks,
    residual_autocorrelation_analysis,
)
from src.data import (
    get_capital_growth_qrtly,
    get_cash_rate_monthly,
    # Interest rates
    get_cash_rate_qrtly,
    get_coal_change_annual,
    get_gdp_growth,
    # Supply shocks
    get_gscpi_qrtly,
    get_import_price_growth_annual,
    get_inflation_annual,
    # Inflation
    get_inflation_qrtly,
    get_labour_force_growth_qrtly,
    get_participation_rate_change_qrtly,
    # GDP
    get_log_gdp,
    # MFP
    get_mfp_annual,
    # Exchange rates
    get_twi_change_annual,
    get_twi_change_qrtly,
    # Terms of trade
    get_tot_change_qrtly,
    # Oil
    get_oil_change_annual,
    # Costs
    get_ulc_growth_qrtly,
    # Labour force
    get_unemployment_rate_qrtly,
    hma,
)
from src.data.gov_spending import get_fiscal_impulse_qrtly
from src.data.abs_loader import load_series
from src.data.gdp import get_gdp
from src.data.rba_loader import PI_TARGET, get_inflation_anchor
from src.data.series_specs import HOURS_WORKED_INDEX
from src.equations import (
    exchange_rate_equation,
    import_price_equation,
    is_equation,
    nairu_equation,
    okun_law_equation,
    participation_equation,
    potential_output_equation,
    price_inflation_equation,
    wage_growth_equation,
)
from src.models.base import SamplerConfig, sample_model

# --- Constants ---

ALPHA = 0.3  # Capital share for Cobb-Douglas
HMA_TERM = 13  # Henderson MA smoothing term

# Plotting constants
MODEL_NAME = "Joint NAIRU + Output Gap Model"
RFOOTER_OUTPUT = "Joint NAIRU + Output Gap Model"


# --- Data Preparation ---


def _get_productivity_trend(trend_weight: float = 0.75) -> pd.Series:
    """Get labour productivity trend for backfilling MFP.

    Calculates GDP per hour worked and blends with linear trend.
    Used to extend MFP series before ABS 5204 data is available.

    Args:
        trend_weight: Weight on linear trend (default 0.75)

    Returns:
        Blended quarterly productivity growth

    """
    gdp = get_gdp(gdp_type="CVM", seasonal="SA").data
    hours = load_series(HOURS_WORKED_INDEX).data

    productivity_index = gdp / hours
    log_productivity = np.log(productivity_index) * 100
    productivity_growth = log_productivity.diff(1).dropna()

    # Fit linear trend
    x = np.arange(len(productivity_growth))
    slope, intercept = np.polyfit(x, productivity_growth.values, 1)
    linear_trend = pd.Series(intercept + slope * x, index=productivity_growth.index)

    # Blend: trend_weight × linear + (1 - trend_weight) × raw
    return trend_weight * linear_trend + (1 - trend_weight) * productivity_growth


def _prepare_mfp_growth() -> pd.Series:
    """Prepare MFP growth: smooth annual data, convert to quarterly, backfill."""
    mfp = get_mfp_annual().data
    mfp_smoothed = hma(mfp.dropna(), 25)

    # Convert annual to quarterly contribution and interpolate
    mfp_quarterly_rate = ((1 + mfp_smoothed / 100) ** 0.25 - 1) * 100
    mfp_quarterly_rate = (
        mfp_quarterly_rate.to_timestamp(how="end")
        .resample("QE-DEC")
        .last()
        .to_period("Q-DEC")
    ).interpolate()

    # Backfill with productivity trend where MFP is unavailable
    productivity_trend = _get_productivity_trend()
    mfp_quarterly_rate = mfp_quarterly_rate.reindex(productivity_trend.index)
    return mfp_quarterly_rate.where(mfp_quarterly_rate.notna(), other=productivity_trend)


def _prepare_labour_force_growth() -> pd.Series:
    """Prepare labour force growth with COVID smoothing."""
    lf_growth_raw = get_labour_force_growth_qrtly().data
    lf_growth_smoothed = hma(lf_growth_raw.dropna(), HMA_TERM)
    lf_growth_smoothed = lf_growth_smoothed.reindex(lf_growth_raw.index)

    # Replace COVID period with smoothed values
    covid_period = pd.period_range("2020Q1", "2023Q2", freq="Q")
    return lf_growth_raw.where(
        ~lf_growth_raw.index.isin(covid_period),
        other=lf_growth_smoothed,
    )


def _prepare_capital_growth() -> pd.Series:
    """Prepare capital growth with Henderson smoothing."""
    capital_growth_raw = get_capital_growth_qrtly().data
    capital_growth = hma(capital_growth_raw.dropna(), HMA_TERM)
    return capital_growth.reindex(capital_growth_raw.index)


def _prepare_coal_change() -> pd.Series:
    """Prepare coal price change (annual log change, lagged 1 quarter)."""
    coal = get_coal_change_annual().data
    return coal.shift(1)


def _prepare_gscpi() -> pd.Series:
    """Prepare GSCPI as COVID supply shock proxy (masked, lagged 2 quarters).

    The GSCPI is only used during the COVID period (2020Q1-2023Q2) to capture
    supply chain disruptions. Outside this period, it is set to zero.
    """
    gscpi = get_gscpi_qrtly().data

    # Mask to COVID period only (zero otherwise)
    covid_period = pd.period_range("2020Q1", "2023Q2", freq="Q")
    gscpi_masked = gscpi.where(gscpi.index.isin(covid_period), other=0.0)

    # Reindex to full range (1960Q1 to current quarter) and fill missing with 0
    current_q = pd.Period.now("Q")
    full_range = pd.period_range("1960Q1", current_q, freq="Q")
    gscpi_full = gscpi_masked.reindex(full_range, fill_value=0.0)

    # Lag by 2 quarters (supply chain effects take time)
    return gscpi_full.shift(2).fillna(0.0)


def compute_r_star(
    capital_growth: pd.Series,
    lf_growth: pd.Series,
    mfp_growth: pd.Series,
    alpha: float = ALPHA,
) -> pd.Series:
    """Compute deterministic r* as smoothed potential growth.

    r* ≈ α×g_K + (1-α)×g_L + g_MFP (annualized and smoothed)
    """
    quarterly_growth = (
        alpha * capital_growth
        + (1 - alpha) * lf_growth
        + mfp_growth
    )
    annual_growth = quarterly_growth.rolling(4).sum()
    annual_growth = annual_growth.bfill()
    return hma(annual_growth.dropna(), HMA_TERM)


def build_observations(
    start: str | None = None,
    end: str | None = None,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex]:
    """Build observation dictionary for model.

    Loads all data from library, applies model-specific transformations,
    aligns to common sample, and returns as numpy arrays.

    Args:
        start: Start period (e.g., "1980Q1")
        end: End period
        verbose: Print sample info

    Returns:
        Tuple of (observations dict, period index)

    """
    # --- Load data from library ---

    # Unemployment
    U = get_unemployment_rate_qrtly().data
    ΔU = U.diff(1)
    ΔU_1 = ΔU.shift(1)
    U_1 = U.shift(1)

    # Participation rate
    Δpr = get_participation_rate_change_qrtly().data

    # GDP
    log_gdp = get_log_gdp().data
    gdp_growth = get_gdp_growth().data

    # Production inputs (with model-specific smoothing)
    capital_growth = _prepare_capital_growth()
    lf_growth = _prepare_labour_force_growth()
    mfp_growth = _prepare_mfp_growth()

    # Inflation
    π = get_inflation_qrtly().data
    π4 = get_inflation_annual().data
    π_anchor = get_inflation_anchor().data

    # Interest rates
    cash_rate = get_cash_rate_qrtly().data

    # Unit labour costs
    Δulc = get_ulc_growth_qrtly().data

    # Import prices
    Δ4ρm = get_import_price_growth_annual().data
    Δ4ρm_1 = Δ4ρm.shift(1)

    # Coal price change (supply shock)
    Δ4coal_1 = _prepare_coal_change()

    # GSCPI (COVID supply chain pressure, masked and lagged)
    ξ_2 = _prepare_gscpi()

    # TWI (Trade-Weighted Index)
    Δtwi = get_twi_change_qrtly().data
    Δtwi_1 = Δtwi.shift(1)
    Δ4twi = get_twi_change_annual().data
    Δ4twi_1 = Δ4twi.shift(1)

    # Terms of trade
    Δtot = get_tot_change_qrtly().data
    Δtot_1 = Δtot.shift(1)

    # Oil prices (AUD)
    Δ4oil = get_oil_change_annual().data
    Δ4oil_1 = Δ4oil.shift(1)

    # Fiscal impulse (G growth - GDP growth)
    fiscal_impulse = get_fiscal_impulse_qrtly().data
    fiscal_impulse_1 = fiscal_impulse.shift(1)

    # Compute r*
    r_star = compute_r_star(capital_growth, lf_growth, mfp_growth)

    # Real rate gap (for exchange rate equation)
    r_gap = cash_rate - π_anchor - r_star
    r_gap_1 = r_gap.shift(1)

    # Build DataFrame
    observed = pd.DataFrame({
        # Inflation
        "π": π,
        "π4": π4,
        "π_anchor": π_anchor,
        # Unemployment
        "U": U,
        "U_1": U_1,
        "ΔU": ΔU,
        "ΔU_1": ΔU_1,
        "ΔU_1_over_U": ΔU_1 / U,
        # Participation
        "Δpr": Δpr,
        # GDP
        "log_gdp": log_gdp,
        "gdp_growth": gdp_growth,
        # Production inputs
        "capital_growth": capital_growth,
        "lf_growth": lf_growth,
        "mfp_growth": mfp_growth,
        # Rates
        "cash_rate": cash_rate,
        "det_r_star": r_star,
        # Unit labor costs
        "Δulc": Δulc,
        # Import prices and supply shocks
        "Δ4ρm": Δ4ρm,
        "Δ4ρm_1": Δ4ρm_1,
        "Δ4coal_1": Δ4coal_1,
        "ξ_2": ξ_2,  # GSCPI (COVID supply chain pressure)
        # Exchange rates
        "Δtwi": Δtwi,
        "Δtwi_1": Δtwi_1,
        "Δ4twi": Δ4twi,
        "Δ4twi_1": Δ4twi_1,
        "r_gap_1": r_gap_1,
        # Terms of trade
        "Δtot": Δtot,
        "Δtot_1": Δtot_1,
        # Oil prices
        "Δ4oil": Δ4oil,
        "Δ4oil_1": Δ4oil_1,
        # Fiscal impulse
        "fiscal_impulse": fiscal_impulse,
        "fiscal_impulse_1": fiscal_impulse_1,
    })

    # Apply sample period
    if start:
        observed = observed[observed.index >= pd.Period(start)]
    if end:
        observed = observed[observed.index <= pd.Period(end)]

    # Fill GSCPI with 0 for periods outside its data range (pre-1998, post-2024)
    # This must be done before dropna() to avoid truncating the sample
    observed["ξ_2"] = observed["ξ_2"].fillna(0.0)

    # Drop missing
    observed = observed.dropna()

    print(f"Observations: {observed.index.min()} to {observed.index.max()} ({len(observed)} periods)")

    # Warn if any NaNs remain (shouldn't happen after dropna)
    if observed.isna().any().any():
        print("WARNING: NaN values remain in observations after dropna()")

    # Check index is unique
    if not observed.index.is_unique:
        raise ValueError("Duplicate periods in observations index")

    # Check index has no gaps (missing periods)
    expected_periods = pd.period_range(observed.index.min(), observed.index.max(), freq="Q")
    if len(observed) != len(expected_periods):
        missing = expected_periods.difference(observed.index)
        raise ValueError(f"{len(missing)} missing period(s) in observations: {list(missing)}")


    # Convert to dict of numpy arrays
    obs_dict = {col: observed[col].to_numpy() for col in observed.columns}
    obs_index = cast("pd.PeriodIndex", observed.index)

    return obs_dict, obs_index


# --- Model Assembly ---


def build_model(
    obs: dict[str, np.ndarray],
    nairu_const: dict[str, Any] | None = None,
    potential_const: dict[str, Any] | None = None,
    exchange_rate_const: dict[str, Any] | None = None,
    import_price_const: dict[str, Any] | None = None,
    participation_const: dict[str, Any] | None = None,
    include_exchange_rate: bool = True,
    include_import_price: bool = True,
    include_participation: bool = True,
) -> pm.Model:
    """Build the joint NAIRU + Output Gap model.

    Args:
        obs: Observation dictionary from build_observations()
        nairu_const: Fixed values for NAIRU equation
        potential_const: Fixed values for potential output equation
        exchange_rate_const: Fixed values for exchange rate equation
        import_price_const: Fixed values for import price equation
        participation_const: Fixed values for participation rate equation
        include_exchange_rate: Whether to include TWI equation (default True)
        include_import_price: Whether to include import price pass-through (default True)
        include_participation: Whether to include participation rate equation (default True)

    Returns:
        PyMC Model ready for sampling

    """
    if nairu_const is None:
        nairu_const = {"nairu_innovation": 0.25}
    if potential_const is None:
        potential_const = {"potential_innovation": 0.3}

    model = pm.Model()

    # State equations
    nairu = nairu_equation(obs, model, constant=nairu_const)
    potential = potential_output_equation(obs, model, constant=potential_const)

    # Observation equations
    okun_law_equation(obs, model, nairu, potential)
    price_inflation_equation(obs, model, nairu)
    wage_growth_equation(obs, model, nairu)
    is_equation(obs, model, potential)

    # Labour supply equation (optional)
    if include_participation:
        participation_equation(obs, model, nairu, constant=participation_const)

    # Open economy equations (optional)
    if include_exchange_rate:
        exchange_rate_equation(obs, model, constant=exchange_rate_const)
    if include_import_price:
        import_price_equation(obs, model, constant=import_price_const)

    return model


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
        from src.analysis import get_vector_var
        samples = get_vector_var("nairu", self.trace)
        samples.index = self.obs_index
        return samples

    def potential_posterior(self) -> pd.DataFrame:
        """Extract potential output posterior as DataFrame."""
        from src.analysis import get_vector_var
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


# --- Main Entry Point ---


def run_model(
    start: str | None = "1980Q1",
    end: str | None = None,
    config: SamplerConfig | None = None,
    verbose: bool = False,
) -> NAIRUResults:
    """Run the full NAIRU + Output Gap estimation.

    Args:
        start: Start period
        end: End period
        config: Sampler configuration
        verbose: Print progress messages

    Returns:
        NAIRUResults with trace and computed series

    """
    if config is None:
        config = SamplerConfig()

    # Build observations
    obs, obs_index = build_observations(start=start, end=end)

    # Build model
    model = build_model(obs)

    # Sample
    print("Sampling...")
    trace = sample_model(model, config)

    return NAIRUResults(
        trace=trace,
        obs=obs,
        obs_index=obs_index,
        model=model,
    )


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
        ("beta_okun", "negative", "Okun coefficient < 0"),
        ("gamma_pi", "negative", "Phillips curve slope < 0"),
        ("gamma_wg", "negative", "Wage Phillips curve slope < 0"),
        ("beta_is", "positive", "IS interest rate effect > 0"),
        ("rho_is", "between_0_1", "IS persistence ∈ (0,1)"),
        # Note: gamma_fi has truncated prior (lower=0), so sign test is uninformative
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
            # Parameter not in model (e.g., IS equation not included)
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


# --- CLI Entry Point ---


def main(verbose: bool = False) -> None:
    """Run the full NAIRU + Output Gap estimation pipeline."""
    # Set output directory for charts
    chart_dir = Path(__file__).parent.parent.parent / "charts" / "nairu_output_gap"
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    print("Running NAIRU + Output Gap model...\n")

    # Sampling configuration (matches notebook)
    config = SamplerConfig(
        draws=10_000,
        tune=3_500,
        chains=5,
        cores=5,
        target_accept=0.90,
    )

    # Build observations
    obs, obs_index = build_observations()

    # Plot observation grid
    plot_obs_grid(obs, obs_index)

    # Build and sample model
    model = build_model(obs)

    print("\nSampling...")
    trace = sample_model(model, config)
    print("\n\n\n\n\n")

    # Create results container
    results = NAIRUResults(
        trace=trace,
        obs=obs,
        obs_index=obs_index,
        model=model,
    )

    # Diagnostics
    check_model_diagnostics(results.trace)

    # Check for zero coefficients
    zero_check = check_for_zero_coeffs(
        results.trace,
        critical_params=["gamma_pi", "gamma_wg", "beta_okun"]
    )
    if verbose:
        print(zero_check.T)

    # Plot scalar posteriors (bar and KDE)
    plot_posteriors_bar(
        results.trace,
        model_name=MODEL_NAME,
        show=False,
    )
    plot_posteriors_kde(
        results.trace,
        model_name=MODEL_NAME,
        show=False,
    )

    # Posterior predictive checks and residual analysis
    obs_vars = {
        "okun_law": obs["ΔU"],
        "observed_price_inflation": obs["π"],
        "observed_wage_growth": obs["Δulc"],
        "observed_twi_change": obs["Δtwi"],
        "observed_import_price": obs["Δ4ρm"],
    }
    var_labels = {
        "okun_law": "Change in Unemployment (pp)",
        "observed_price_inflation": "Quarterly Inflation (%)",
        "observed_wage_growth": "Unit Labour Cost Growth (%)",
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
        show=False,
    )

    residual_autocorrelation_analysis(
        ppc=ppc_data,
        obs_vars=obs_vars,
        obs_index=obs_index,
        var_labels=var_labels,
        model_name=MODEL_NAME,
        rfooter=RFOOTER_OUTPUT,
        show=False,
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
        show=False,
    )

    # Inflation decomposition (demand vs supply)
    decomp = decompose_inflation(results.trace, results.obs, results.obs_index)
    plot_inflation_decomposition(decomp, rfooter=RFOOTER_OUTPUT)

    print(f"\nCharts saved to: {chart_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NAIRU + Output Gap model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()
    main(verbose=args.verbose)
