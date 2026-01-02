"""NAIRU + Output Gap Stage 3: Model-Consistent Forecasting.

This module generates 2-quarter ahead forecasts using estimated model coefficients:
- IS curve: projects output gap using historical rate gaps
- Okun's Law: derives unemployment changes from output gap
- NAIRU: random walk (stays at current level)
- Potential: grows at Cobb-Douglas drift

Interpretation: "No new shocks, economy trends toward equilibrium"
"""

from dataclasses import dataclass, field
from pathlib import Path

import mgplot as mg
import numpy as np
import pandas as pd

from src.models.nairu.analysis import get_scalar_var, get_vector_var
from src.models.nairu.stage2 import load_results, NAIRUResults, build_model
from src.utilities.rate_conversion import quarterly, annualize

# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"
DEFAULT_CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "nairu_output_gap"

# Forecast horizon (4 quarters with move-and-hold assumption for rates)
FORECAST_HORIZON = 4

# Policy scenarios (bp change from current rate)
DEFAULT_POLICY_SCENARIOS = {
    "+200bp": 2.00,
    "+100bp": 1.00,
    "+50bp": 0.50,
    "+25bp": 0.25,
    "hold": 0.00,
    "-25bp": -0.25,
    "-50bp": -0.50,
    "-100bp": -1.00,
    "-200bp": -2.00,
}

# DSR passthrough: ~1.0pp per 100bp rate change
# Derivation:
#   - Housing debt: ~$2.2T
#   - Gross disposable income: ~$1.6T/yr
#   - Variable rate share: ~70%
#   - Cash rate to mortgage passthrough: ~100% for increases
#   - Extra interest per 100bp = $2.2T × 70% × 1% = $15.4B
#   - DSR change = $15.4B / $1.6T ≈ 1.0pp
RATE_TO_DSR_PASSTHROUGH = 1.0

# FX channel passthrough (from RBA Bulletin April 2025)
# Full chain: Rate → TWI → Import prices → Inflation
# RBA estimates: 100bp rate increase → 0.25-0.5pp lower inflation over 2 years
# We use 0.35pp (midpoint) distributed across forecast horizon
#
# IMPORTANT: Fed counterfactual assumptions
# This calibration assumes the RBA moves unilaterally (Fed holds steady).
# The FX channel depends on RELATIVE interest rates:
#   - OVERSTATES effect if Fed moves in tandem with RBA (TWI unchanged)
#   - UNDERSTATES effect if Fed moves opposite to RBA (TWI moves more)
# Example: If Fed cuts 100bp while RBA hikes 50bp, relative divergence is 150bp,
# and FX channel effect would be ~50% larger than calibrated.
#
# Note: Model's estimated UIP coefficients are too small (UIP puzzle):
#   - β_er_r (~0.08% TWI per 100bp) vs RBA's 5-10%
# We use RBA's total effect calibration for policy-relevant forecasts.
RATE_TO_INFLATION_FX = 0.35 / 4  # pp inflation per 100bp per quarter (annualized 0.35pp over ~4Q)

# Housing wealth channel passthrough (from RBA research)
# Rate → House prices → Household wealth → Consumption
# RBA estimates: 100bp rate increase → 2-4% house price decline (annualized)
# That's roughly -0.5% to -1% per quarter
# We use -1% per quarter per 100bp as the passthrough
RATE_TO_HOUSING_WEALTH = -1.0  # % change in housing wealth growth per 100bp

# Demand transmission scaling factor
# The model estimates β_okun ≈ -0.16, which combined with estimated Phillips curve
# gives a demand channel of ~0.18pp inflation per 100bp rate change.
# RBA estimates total transmission of ~0.64pp, with FX accounting for ~0.35pp,
# implying demand channel should be ~0.29pp.
# Scaling factor: 0.29 / 0.18 ≈ 1.6
#
# Why the gap? Model's demand channel may understate transmission because:
# 1. Expectations channel not explicitly modeled (credible policy amplifies effects)
# 2. Credit/lending channel (banks tighten standards when rates rise)
# 3. Sample period includes low-rate era where transmission was weaker
DEMAND_TRANSMISSION_MULTIPLIER = 1.6  # scales unemployment response to match RBA demand channel


@dataclass
class ForecastResults:
    """Container for model-consistent forecast results."""

    forecast_index: pd.PeriodIndex
    obs_index: pd.PeriodIndex

    # Posterior samples for forecast periods (rows=periods, cols=samples)
    nairu_forecast: pd.DataFrame
    potential_forecast: pd.DataFrame
    output_gap_forecast: pd.DataFrame
    unemployment_forecast: pd.DataFrame
    inflation_forecast: pd.DataFrame  # quarterly inflation

    # Model coefficients used (posterior medians for display)
    coefficients: dict

    # Growth/drift rates
    potential_growth: float

    # Historical final values (for context)
    nairu_final: float
    potential_final: float
    output_gap_final: float
    unemployment_final: float

    # Policy scenario info
    scenario_name: str = "baseline"
    cash_rate: float = None  # cash rate used for T+2

    def _median(self, df: pd.DataFrame) -> pd.Series:
        return df.median(axis=1)

    def _quantiles(self, df: pd.DataFrame, prob: float = 0.90) -> pd.DataFrame:
        lower = (1 - prob) / 2
        upper = 1 - lower
        return pd.DataFrame(
            {
                "lower": df.quantile(lower, axis=1),
                "median": df.median(axis=1),
                "upper": df.quantile(upper, axis=1),
            },
            index=df.index,
        )

    def nairu_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.nairu_forecast, prob)

    def potential_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.potential_forecast, prob)

    def output_gap_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.output_gap_forecast, prob)

    def unemployment_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.unemployment_forecast, prob)

    def unemployment_gap_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """U - NAIRU gap with uncertainty."""
        gap = self.unemployment_forecast - self.nairu_forecast
        return self._quantiles(gap, prob)

    def inflation_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """Quarterly inflation forecast with uncertainty."""
        return self._quantiles(self.inflation_forecast, prob)

    def inflation_annual_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """Annualised inflation forecast (compound conversion)."""
        annual = annualize(self.inflation_forecast)
        return self._quantiles(annual, prob)

    def summary(self) -> pd.DataFrame:
        """Point forecast summary (posterior medians)."""
        return pd.DataFrame(
            {
                "Output Gap": self._median(self.output_gap_forecast),
                "U": self._median(self.unemployment_forecast),
                "NAIRU": self._median(self.nairu_forecast),
                "U Gap": self._median(self.unemployment_forecast - self.nairu_forecast),
                "π (qtr)": self._median(self.inflation_forecast),
                "π (ann)": annualize(self._median(self.inflation_forecast)),
            },
            index=self.forecast_index,
        )

    def print_summary(self) -> None:
        """Print formatted forecast summary."""
        print("\n" + "=" * 70)
        print("MODEL-CONSISTENT FORECAST (No Shocks, Trend to Equilibrium)")
        print("=" * 70)

        print(f"\nModel coefficients (posterior median):")
        print(f"  rho_is (output gap persistence): {self.coefficients['rho_is']:.3f}")
        print(f"  beta_is (interest rate sensitivity): {self.coefficients['beta_is']:.3f}")
        print(f"  delta_dsr (debt servicing sensitivity): {self.coefficients['delta_dsr']:.3f}")
        print(f"  eta_hw (housing wealth sensitivity): {self.coefficients['eta_hw']:.3f}")
        print(f"  beta_okun (Okun coefficient): {self.coefficients['beta_okun']:.3f}")
        print(f"  gamma_pi_covid (Phillips slope): {self.coefficients['gamma_pi_covid']:.3f}")
        print(f"  Potential growth: {self.potential_growth:.3f}%/qtr ({self.potential_growth * 4:.2f}% ann.)")
        print(f"  Rate→DSR pass-through: {RATE_TO_DSR_PASSTHROUGH:.2f}pp per 100bp")
        print(f"  Rate→Housing wealth: {RATE_TO_HOUSING_WEALTH:.1f}% growth per 100bp")
        print(f"  Rate→Inflation (FX): {RATE_TO_INFLATION_FX * 4:.2f}pp per 100bp (RBA calibration)")
        print(f"  Demand transmission multiplier: {DEMAND_TRANSMISSION_MULTIPLIER:.1f}x (to match RBA demand channel)")

        print(f"\nStarting point ({self.obs_index[-1]}):")
        print(f"  Output gap: {self.output_gap_final:+.4f}")
        print(f"  Unemployment: {self.unemployment_final:.2f}%")
        print(f"  NAIRU: {self.nairu_final:.2f}%")
        print(f"  U gap: {self.unemployment_final - self.nairu_final:+.2f} pp")

        print("\n" + "-" * 70)
        print("Point Forecasts (posterior median)")
        print("-" * 70)
        print(self.summary().round(3).to_string())

        print("\n" + "-" * 70)
        print("Output Gap Forecast (90% HDI)")
        print("-" * 70)
        print(self.output_gap_hdi().round(4).to_string())

        print("\n" + "-" * 70)
        print("Unemployment Forecast (90% HDI)")
        print("-" * 70)
        print(self.unemployment_hdi().round(2).to_string())

        print("\n" + "-" * 70)
        print("Unemployment Gap Forecast (90% HDI)")
        print("-" * 70)
        print(self.unemployment_gap_hdi().round(2).to_string())

        print("\n" + "-" * 70)
        print("Inflation Forecast - Annualised (90% HDI)")
        print("-" * 70)
        print(self.inflation_annual_hdi().round(2).to_string())

        print("\n" + "=" * 70)


def forecast(
    results: NAIRUResults,
    cash_rate_override: float | None = None,
    scenario_name: str = "baseline",
) -> ForecastResults:
    """Generate model-consistent 2-quarter forecast.

    Uses estimated coefficients to project forward:
    1. IS curve: y_gap_t = rho * y_gap_{t-1} - beta * rate_gap_{t-2}
    2. Okun's Law: delta_U_t = beta_okun * y_gap_t
    3. NAIRU: stays at final posterior (random walk)
    4. Potential: grows at Cobb-Douglas drift

    Args:
        results: NAIRUResults from stage2
        cash_rate_override: Override cash rate for period T (affects T+2 forecast)
        scenario_name: Name for this scenario

    Returns:
        ForecastResults with model-consistent forecasts

    """
    obs = results.obs
    obs_index = results.obs_index
    trace = results.trace
    h = FORECAST_HORIZON

    # --- Extract posterior samples ---
    nairu_posterior = results.nairu_posterior()
    potential_posterior = results.potential_posterior()
    n_samples = nairu_posterior.shape[1]

    # Coefficient posteriors (full distributions)
    rho_is = get_scalar_var("rho_is", trace).values
    beta_is = get_scalar_var("beta_is", trace).values
    delta_dsr = get_scalar_var("delta_dsr", trace).values
    eta_hw = get_scalar_var("eta_hw", trace).values
    beta_okun = get_scalar_var("beta_okun", trace).values
    gamma_pi_covid = get_scalar_var("gamma_pi_covid", trace).values
    # FX channel coefficients
    beta_pt = get_scalar_var("beta_pt", trace).values  # TWI → import prices
    rho_pi = get_scalar_var("rho_pi", trace).values    # import prices → inflation

    # Align sample sizes (trace may have different number of samples)
    n_coef_samples = len(rho_is)
    if n_coef_samples != n_samples:
        # Resample coefficients to match state variable samples
        idx = np.random.choice(n_coef_samples, size=n_samples, replace=True)
        rho_is = rho_is[idx]
        beta_is = beta_is[idx]
        delta_dsr = delta_dsr[idx]
        eta_hw = eta_hw[idx]
        beta_okun = beta_okun[idx]
        gamma_pi_covid = gamma_pi_covid[idx]
        beta_pt = beta_pt[idx]
        rho_pi = rho_pi[idx]

    # --- Historical values needed for forecasting ---
    # Final period states
    nairu_T = nairu_posterior.iloc[-1].values
    potential_T = potential_posterior.iloc[-1].values
    log_gdp_T = obs["log_gdp"][-1]
    U_T = obs["U"][-1]

    # Output gap at T (for each posterior sample)
    output_gap_T = log_gdp_T - potential_T

    # Rate gaps for IS curve (lagged 2 periods)
    # rate_gap = (cash_rate - pi_anchor) - r_star
    real_rate = obs["cash_rate"] - obs["π_anchor"]
    rate_gap = real_rate - obs["det_r_star"]

    # Rate gaps for IS curve (lagged 2 periods)
    # T+1: uses rate_gap at T-1 (historical)
    # T+2: uses rate_gap at T (policy decision)
    # T+3: uses rate_gap at T+1 (hold assumption)
    # T+4: uses rate_gap at T+2 (hold assumption)
    rate_gap_Tm1 = rate_gap[-2]  # T-1 (historical)

    # Rate gap at T onwards - can be overridden for policy scenarios
    # Under "move and hold", rate stays at new level
    cash_rate_T = obs["cash_rate"][-1]
    if cash_rate_override is not None:
        cash_rate_T = cash_rate_override

    # Compute rate gap assuming rate holds at cash_rate_T
    pi_anchor_T = obs["π_anchor"][-1]
    r_star_T = obs["det_r_star"][-1]
    real_rate_hold = cash_rate_T - pi_anchor_T
    rate_gap_hold = real_rate_hold - r_star_T  # used for T, T+1, T+2, ...

    # Rate change for DSR transmission
    # If policy changes rate, this feeds through to DSR at lag 1
    rate_change = cash_rate_T - obs["cash_rate"][-1]  # 0 if no override

    # Inflation anchor (2.5% annual, convert to quarterly)
    pi_anchor_annual = 2.5
    pi_anchor_qtr = quarterly(pi_anchor_annual)

    # --- Compute potential growth (Cobb-Douglas drift) ---
    n_recent = min(4, len(obs["capital_growth"]))
    alpha = float(obs["alpha_capital"][-n_recent:].mean())
    g_K = float(obs["capital_growth"][-n_recent:].mean())
    g_L = float(obs["lf_growth"][-n_recent:].mean())
    g_MFP = float(obs["mfp_growth"][-n_recent:].mean())
    potential_growth = alpha * g_K + (1 - alpha) * g_L + g_MFP

    # --- Create forecast period index ---
    last_period = obs_index[-1]
    forecast_index = pd.period_range(
        start=last_period + 1,
        periods=h,
        freq=last_period.freq,
    )

    # --- Project forward (for each posterior sample) ---

    # Arrays to hold forecasts (h periods x n_samples)
    nairu_fcst = np.zeros((h, n_samples))
    potential_fcst = np.zeros((h, n_samples))
    output_gap_fcst = np.zeros((h, n_samples))
    unemployment_fcst = np.zeros((h, n_samples))
    inflation_fcst = np.zeros((h, n_samples))

    # Initial conditions
    output_gap_prev = output_gap_T
    U_prev = U_T

    for t in range(h):
        # NAIRU: random walk, stays at current level
        nairu_fcst[t] = nairu_T

        # Potential: grows at Cobb-Douglas drift
        potential_fcst[t] = potential_T + (t + 1) * potential_growth

        # IS curve: y_gap_t = rho * y_gap_{t-1} - beta * rate_gap_{t-2} - delta * Δdsr_{t-1} + eta * Δhw_{t-1}
        # Rate gap timing:
        #   t=0 (T+1): uses rate_gap at T-1 (historical)
        #   t=1 (T+2): uses rate_gap at T (policy decision)
        #   t=2+ (T+3, T+4): uses rate_gap_hold (move and hold)
        # DSR timing:
        #   t=0 (T+1): uses Δdsr at T (historical, from obs)
        #   t=1+ (T+2, ...): rate change feeds through to DSR at lag 1
        # Housing wealth timing:
        #   t=0 (T+1): uses Δhw at T (historical, from obs)
        #   t=1+ (T+2, ...): rate change feeds through to housing wealth at lag 1
        if t == 0:
            rate_gap_lag2 = rate_gap_Tm1
            # Use last observed DSR change (already lagged in obs)
            delta_dsr_lag1 = obs["Δdsr_1"][-1]
            # Use last observed housing wealth change (already lagged in obs)
            delta_hw_lag1 = obs["Δhw_1"][-1]
            # Use last observed import price change (already lagged in obs)
            delta_import_price = obs["Δ4ρm_1"][-1]
        else:
            rate_gap_lag2 = rate_gap_hold
            # Rate change → DSR change at lag 1
            # For t>=1, the rate change from period T feeds through
            delta_dsr_lag1 = rate_change * RATE_TO_DSR_PASSTHROUGH
            # Rate change → Housing wealth change at lag 1
            # rate_change is in pp (e.g., 0.5 for 50bp), RATE_TO_HOUSING_WEALTH is per 100bp
            delta_hw_lag1 = obs["Δhw_1"][-1] + rate_change * RATE_TO_HOUSING_WEALTH
            # FX channel: use RBA-calibrated passthrough (model estimates too small)
            # rate_change is in pp (e.g., 0.5 for 50bp)
            fx_inflation_effect = -rate_change * RATE_TO_INFLATION_FX  # negative: rate up → inflation down

        output_gap_fcst[t] = (
            rho_is * output_gap_prev
            - beta_is * rate_gap_lag2
            - delta_dsr * delta_dsr_lag1
            + eta_hw * delta_hw_lag1
        )

        # Okun's Law: delta_U = beta_okun * output_gap
        # Apply calibration factor to match RBA total transmission estimate
        delta_U = beta_okun * output_gap_fcst[t] * DEMAND_TRANSMISSION_MULTIPLIER
        unemployment_fcst[t] = U_prev + delta_U

        # Phillips curve: π = π_anchor + γ × u_gap + FX effect
        # u_gap = (U - NAIRU) / U
        # FX channel: at t=0 use model's ρ_pi × historical import prices
        #             at t>=1 use RBA-calibrated rate→inflation passthrough
        u_gap = (unemployment_fcst[t] - nairu_fcst[t]) / unemployment_fcst[t]
        if t == 0:
            inflation_fcst[t] = pi_anchor_qtr + gamma_pi_covid * u_gap + rho_pi * delta_import_price
        else:
            inflation_fcst[t] = pi_anchor_qtr + gamma_pi_covid * u_gap + fx_inflation_effect

        # Update for next iteration
        output_gap_prev = output_gap_fcst[t]
        U_prev = unemployment_fcst[t]

    # --- Package results ---
    cols = nairu_posterior.columns

    return ForecastResults(
        forecast_index=forecast_index,
        obs_index=obs_index,
        nairu_forecast=pd.DataFrame(nairu_fcst, index=forecast_index, columns=cols),
        potential_forecast=pd.DataFrame(potential_fcst, index=forecast_index, columns=cols),
        output_gap_forecast=pd.DataFrame(output_gap_fcst, index=forecast_index, columns=cols),
        unemployment_forecast=pd.DataFrame(unemployment_fcst, index=forecast_index, columns=cols),
        inflation_forecast=pd.DataFrame(inflation_fcst, index=forecast_index, columns=cols),
        coefficients={
            "rho_is": float(np.median(rho_is)),
            "beta_is": float(np.median(beta_is)),
            "delta_dsr": float(np.median(delta_dsr)),
            "eta_hw": float(np.median(eta_hw)),
            "beta_okun": float(np.median(beta_okun)),
            "gamma_pi_covid": float(np.median(gamma_pi_covid)),
            # Note: beta_pt and rho_pi not used in forecasts (RBA calibration used instead)
        },
        potential_growth=potential_growth,
        nairu_final=float(np.median(nairu_T)),
        potential_final=float(np.median(potential_T)),
        output_gap_final=float(np.median(output_gap_T)),
        unemployment_final=U_T,
        scenario_name=scenario_name,
        cash_rate=cash_rate_T,
    )


def run_scenarios(
    results: NAIRUResults,
    scenarios: dict[str, float] | None = None,
) -> dict[str, ForecastResults]:
    """Run multiple policy scenarios.

    Args:
        results: NAIRUResults from stage2
        scenarios: Dict of {name: bp_change} from current rate.
                  Default: +50bp, +25bp, hold, -25bp, -50bp

    Returns:
        Dict of {scenario_name: ForecastResults}

    """
    if scenarios is None:
        scenarios = DEFAULT_POLICY_SCENARIOS

    # Get current cash rate from observations
    current_rate = results.obs["cash_rate"][-1]

    scenario_results = {}
    for name, bp_change in scenarios.items():
        new_rate = current_rate + bp_change
        scenario_results[name] = forecast(
            results,
            cash_rate_override=new_rate,
            scenario_name=name,
        )

    return scenario_results


def print_scenario_comparison(
    scenario_results: dict[str, ForecastResults],
    current_rate: float,
) -> None:
    """Print comparison table across scenarios."""
    print("\n" + "=" * 80)
    print("POLICY SCENARIO COMPARISON")
    print(f"Current cash rate in model: {current_rate:.2f}%")
    print("=" * 80)

    # Get forecast periods from first result
    first_result = next(iter(scenario_results.values()))
    periods = first_result.forecast_index

    # Build comparison tables for each period
    for period in periods:
        print(f"\n{period}:")
        print("-" * 80)

        rows = []
        for name, result in scenario_results.items():
            idx = result.forecast_index.get_loc(period)
            rows.append({
                "Scenario": name,
                "Cash Rate": f"{result.cash_rate:.2f}%",
                "Output Gap": f"{result.output_gap_forecast.iloc[idx].median():.3f}",
                "U": f"{result.unemployment_forecast.iloc[idx].median():.2f}%",
                "U Gap": f"{(result.unemployment_forecast.iloc[idx] - result.nairu_forecast.iloc[idx]).median():.2f}",
                "π (ann)": f"{annualize(result.inflation_forecast.iloc[idx].median()):.2f}%",
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

    print("\n" + "=" * 80)


def plot_scenario_inflation(
    scenario_results: dict[str, ForecastResults],
    chart_dir: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot inflation forecasts across policy scenarios.

    Args:
        scenario_results: Dict of {scenario_name: ForecastResults} from run_scenarios
        chart_dir: Directory to save charts. If None, uses DEFAULT_CHART_DIR.
        show: Display plots interactively.

    """
    if chart_dir is None:
        chart_dir = DEFAULT_CHART_DIR
    chart_dir = Path(chart_dir)
    mg.set_chart_dir(str(chart_dir))

    # Get scenario names and colors
    scenario_order = ["+200bp", "+100bp", "+50bp", "+25bp", "hold", "-25bp", "-50bp", "-100bp", "-200bp"]
    colors = {
        "+200bp": "darkred",
        "+100bp": "red",
        "+50bp": "orangered",
        "+25bp": "orange",
        "hold": "black",
        "-25bp": "deepskyblue",
        "-50bp": "steelblue",
        "-100bp": "blue",
        "-200bp": "darkblue",
    }

    # Build DataFrame of annualized inflation medians
    inflation_data = {}
    for name in scenario_order:
        if name in scenario_results:
            result = scenario_results[name]
            inflation_median = result.inflation_forecast.median(axis=1)
            inflation_annual = annualize(inflation_median)
            inflation_data[name] = inflation_annual

    df = pd.DataFrame(inflation_data)

    # Plot each scenario
    ax = None
    for name in scenario_order:
        if name in df.columns:
            series = df[name]
            series.name = name
            if ax is None:
                ax = mg.line_plot(series, color=colors[name], width=2 if name == "hold" else 1.5)
            else:
                mg.line_plot(series, ax=ax, color=colors[name], width=2 if name == "hold" else 1.5)

    # Add reference line at 2.5% target
    if ax is not None:
        ax.axhline(y=2.5, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="Target (2.5%)")

        # Get current cash rate for title
        hold_result = scenario_results.get("hold")
        current_rate = hold_result.cash_rate if hold_result else None
        rate_str = f" (from {current_rate:.2f}%)" if current_rate is not None else ""

        mg.finalise_plot(
            ax,
            title=f"Inflation Forecast by Policy Scenario{rate_str}",
            ylabel="Per cent per annum",
            legend={"loc": "best", "fontsize": "x-small"},
            lheader="Trimmed mean, annualised",
            lfooter="Australia. Ceteris paribus: model-consistent forecasts, no new shocks.",
            rfooter="4-quarter horizon. Demand, FX, DSR, housing wealth channels.",
            show=show,
        )


def run_stage3(
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
    run_scenarios_flag: bool = True,
    plot_scenarios: bool = True,
    chart_dir: Path | str | None = None,
    verbose: bool = False,
) -> ForecastResults | dict[str, ForecastResults]:
    """Run Stage 3: Load results and generate model-consistent forecast.

    Args:
        output_dir: Directory containing saved results from Stage 1
        prefix: Filename prefix used when saving
        run_scenarios_flag: If True, run all policy scenarios
        plot_scenarios: If True, plot inflation scenario comparison
        chart_dir: Directory to save charts. If None, uses DEFAULT_CHART_DIR.
        verbose: Print detailed output

    Returns:
        ForecastResults (single) or dict of ForecastResults (scenarios)

    """
    print("Loading model results...")

    # Load results
    trace, obs, obs_index, constants = load_results(output_dir=output_dir, prefix=prefix)

    # Rebuild model and create results container
    model = build_model(obs)
    results = NAIRUResults(
        trace=trace,
        obs=obs,
        obs_index=obs_index,
        model=model,
    )

    current_rate = obs["cash_rate"][-1]

    if run_scenarios_flag:
        # Run all policy scenarios
        scenario_results = run_scenarios(results)

        # Print baseline summary
        baseline = scenario_results["hold"]
        baseline.print_summary()

        # Print scenario comparison
        print_scenario_comparison(scenario_results, current_rate)

        # Plot scenario inflation comparison
        if plot_scenarios:
            plot_scenario_inflation(scenario_results, chart_dir=chart_dir, show=False)

        return scenario_results
    else:
        # Single baseline forecast
        forecast_results = forecast(results)
        forecast_results.print_summary()
        return forecast_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run NAIRU + Output Gap model-consistent forecasting (Stage 3)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory with saved results")
    parser.add_argument(
        "--no-scenarios",
        action="store_true",
        help="Run baseline only (no policy scenarios)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plotting",
    )
    args = parser.parse_args()

    run_stage3(
        output_dir=args.output_dir,
        run_scenarios_flag=not args.no_scenarios,
        plot_scenarios=not args.no_plots,
        verbose=args.verbose,
    )
