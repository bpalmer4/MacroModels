"""NAIRU + Output Gap Stage 3: Model-Consistent Forecasting.

This module generates 4-quarter ahead forecasts using estimated model coefficients:
- IS curve: projects output gap using historical rate gaps
- Okun's Law: derives unemployment changes from output gap
- NAIRU: random walk (stays at current level)
- Potential: grows at Cobb-Douglas drift

Interpretation: "No new shocks, economy trends toward equilibrium"
"""

from dataclasses import dataclass
from pathlib import Path

import mgplot as mg
import numpy as np
import pandas as pd

from src.data.inflation import get_trimmed_mean_qrtly
from src.models.nairu.analysis import get_scalar_var
from src.models.nairu.stage2 import NAIRUResults, build_model, load_results
from src.utilities.rate_conversion import annualize, quarterly

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

# RBA demand channel benchmark (from RBA Bulletin, ex-FX)
# Total transmission: ~0.64pp inflation per 100bp over 2 years
# FX channel: ~0.35pp (calibrated separately via RATE_TO_INFLATION_FX)
# Demand channel: ~0.29pp (residual)
#
# The model's estimated demand chain (IS → Okun → Phillips) typically
# understates this because it cannot capture:
# 1. Expectations channel (credible policy amplifies effects)
# 2. Credit/lending channel (banks tighten standards when rates rise)
# 3. Sample period includes low-rate era where transmission was weaker
#
# We compute the multiplier dynamically from the posterior each run,
# scaling beta_is so the model's demand channel matches the RBA benchmark.
RBA_DEMAND_CHANNEL_PP = 0.29  # pp inflation per 100bp, ex-FX


def _compute_demand_multiplier(
    beta_is: np.ndarray,
    rho_is: np.ndarray,
    tau1_okun: np.ndarray,
    tau2_okun: np.ndarray,
    gamma_pi: np.ndarray,
    u_level: float,
    h: int = FORECAST_HORIZON,
) -> float:
    """Compute demand transmission multiplier from posterior medians.

    Simulates a 100bp shock through the estimated demand chain:
        rate gap → output gap (IS) → unemployment gap (Okun) → inflation (Phillips)
    and compares the cumulative inflation effect to the RBA benchmark.

    Returns:
        Multiplier to apply to beta_is (floored at 1.0 — never weaken transmission).

    """
    # Use posterior medians for a stable point estimate
    b_is = float(np.median(beta_is))
    r_is = float(np.median(rho_is))
    t1 = float(np.median(tau1_okun))
    t2 = float(np.median(tau2_okun))
    g_pi = float(np.median(gamma_pi))

    # Simulate 100bp shock (rate_gap = 1.0) over forecast horizon
    og = 0.0  # output gap
    ug = 0.0  # unemployment gap
    cumulative_inflation = 0.0

    for t in range(h):
        # IS curve: rate gap hits with 2Q lag, so only from t >= 2
        rate_effect = 1.0 if t >= 2 else 0.0
        og = r_is * og - b_is * rate_effect
        # Okun's law
        ug = t2 * ug + t1 * og
        # Phillips curve (quarterly inflation effect)
        u_gap_ratio = ug / u_level
        cumulative_inflation += g_pi * u_gap_ratio

    model_demand_pp = abs(cumulative_inflation)

    if model_demand_pp < 1e-6:
        # Model estimates near-zero transmission; use a conservative cap
        return 2.0

    multiplier = RBA_DEMAND_CHANNEL_PP / model_demand_pp
    # Floor at 1.0: never weaken the model's own estimates
    return max(1.0, multiplier)


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
    output_gap_final_posterior: pd.Series  # Full posterior for sample-by-sample GDP growth
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

    def gdp_growth_forecast(self) -> pd.DataFrame:
        """Quarterly GDP growth forecast (%).

        GDP growth = potential_growth + Δ(output_gap)

        Returns:
            DataFrame with quarterly GDP growth (rows=periods, cols=samples)

        """
        # Change in output gap from previous period
        # First period: change from final historical output gap (sample-by-sample)
        # Subsequent periods: change from previous forecast period
        output_gap_change = self.output_gap_forecast.diff()
        output_gap_change.iloc[0] = (
            self.output_gap_forecast.iloc[0] - self.output_gap_final_posterior
        )

        # GDP growth = potential growth (constant) + output gap change
        gdp_growth = self.potential_growth + output_gap_change

        return gdp_growth

    def gdp_growth_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """Quarterly GDP growth forecast with uncertainty."""
        return self._quantiles(self.gdp_growth_forecast(), prob)

    def gdp_growth_annual_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """Annualised GDP growth forecast (compound conversion)."""
        annual = annualize(self.gdp_growth_forecast())
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

        print("\nModel coefficients (posterior median):")
        if "tau1_okun" in self.coefficients:
            print(f"  tau1_okun (Okun output gap effect): {self.coefficients['tau1_okun']:.3f}")
            print(f"  tau2_okun (Okun persistence): {self.coefficients['tau2_okun']:.3f}")
        else:
            print(f"  beta_okun (Okun coefficient): {self.coefficients['beta_okun']:.3f}")
        if "rho_is" in self.coefficients:
            print(f"  rho_is (output gap persistence): {self.coefficients['rho_is']:.3f}")
            print(f"  beta_is (interest rate sensitivity): {self.coefficients['beta_is']:.3f}")
            if "gamma_fi" in self.coefficients:
                print(f"  gamma_fi (fiscal impulse effect): {self.coefficients['gamma_fi']:.3f}")
            if "delta_dsr" in self.coefficients:
                print(f"  delta_dsr (debt servicing sensitivity): {self.coefficients['delta_dsr']:.3f}")
            if "eta_hw" in self.coefficients:
                print(f"  eta_hw (housing wealth sensitivity): {self.coefficients['eta_hw']:.3f}")
        gamma_key = "gamma_pi_covid" if "gamma_pi_covid" in self.coefficients else "gamma_pi"
        print(f"  {gamma_key} (Phillips slope): {self.coefficients[gamma_key]:.3f}")
        print(f"  Potential growth: {self.potential_growth:.3f}%/qtr ({self.potential_growth * 4:.2f}% ann.)")

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
    seed: int = 42,
) -> ForecastResults:
    """Generate model-consistent 4-quarter forecast.

    Uses estimated coefficients to project forward:
    1. IS curve: y_gap_t = rho * y_gap_{t-1} - beta * rate_gap_{t-2}
    2. Okun's Law: delta_U_t = beta_okun * y_gap_t
    3. NAIRU: stays at final posterior (random walk)
    4. Potential: grows at Cobb-Douglas drift

    Args:
        results: NAIRUResults from stage2
        cash_rate_override: Override cash rate for period T (affects T+2 forecast)
        scenario_name: Name for this scenario
        seed: Random seed for reproducibility

    Returns:
        ForecastResults with model-consistent forecasts

    """
    rng = np.random.default_rng(seed)
    obs = results.obs
    obs_index = results.obs_index
    trace = results.trace
    h = FORECAST_HORIZON

    # --- Extract posterior samples ---
    nairu_posterior = results.nairu_posterior()
    potential_posterior = results.potential_posterior()
    n_samples = nairu_posterior.shape[1]

    # Coefficient posteriors (full distributions)
    has_okun_gap = "tau1_okun" in trace.posterior
    if has_okun_gap:
        tau1_okun = get_scalar_var("tau1_okun", trace).to_numpy()
        tau2_okun = get_scalar_var("tau2_okun", trace).to_numpy()
    else:
        beta_okun = get_scalar_var("beta_okun", trace).to_numpy()
    # IS curve (optional)
    has_is_curve = "rho_is" in trace.posterior
    if has_is_curve:
        rho_is = get_scalar_var("rho_is", trace).to_numpy()
        beta_is = get_scalar_var("beta_is", trace).to_numpy()
        gamma_fi = get_scalar_var("gamma_fi", trace).to_numpy() if "gamma_fi" in trace.posterior else 0.0
        delta_dsr = get_scalar_var("delta_dsr", trace).to_numpy() if "delta_dsr" in trace.posterior else 0.0
        eta_hw = get_scalar_var("eta_hw", trace).to_numpy() if "eta_hw" in trace.posterior else 0.0
    # Phillips slope: regime-switching uses covid regime, single-slope uses gamma_pi
    gamma_var = "gamma_pi_covid" if "gamma_pi_covid" in trace.posterior else "gamma_pi"
    gamma_pi_covid = get_scalar_var(gamma_var, trace).to_numpy()
    # FX channel coefficients (import price equation may be excluded)
    has_import_price = "beta_pt" in trace.posterior
    beta_pt = get_scalar_var("beta_pt", trace).to_numpy() if has_import_price else None
    has_rho_pi = "rho_pi" in trace.posterior
    rho_pi = get_scalar_var("rho_pi", trace).to_numpy() if has_rho_pi else 0.0

    # Align sample sizes (trace may have different number of samples)
    n_coef_samples = len(tau1_okun if has_okun_gap else beta_okun)
    if n_coef_samples != n_samples:
        # Resample coefficients to match state variable samples
        idx = rng.choice(n_coef_samples, size=n_samples, replace=True)
        if has_okun_gap:
            tau1_okun = tau1_okun[idx]
            tau2_okun = tau2_okun[idx]
        else:
            beta_okun = beta_okun[idx]
        if has_is_curve:
            rho_is = rho_is[idx]
            beta_is = beta_is[idx]
            if not np.isscalar(gamma_fi):
                gamma_fi = gamma_fi[idx]
            if not np.isscalar(delta_dsr):
                delta_dsr = delta_dsr[idx]
            if not np.isscalar(eta_hw):
                eta_hw = eta_hw[idx]
        gamma_pi_covid = gamma_pi_covid[idx]
        if beta_pt is not None:
            beta_pt = beta_pt[idx]
        if not np.isscalar(rho_pi):
            rho_pi = rho_pi[idx]

    # --- Historical values needed for forecasting ---
    # Final period states
    nairu_T = nairu_posterior.iloc[-1].to_numpy()
    potential_T = potential_posterior.iloc[-1].to_numpy()
    log_gdp_T = obs["log_gdp"][-1]
    U_T = obs["U"][-1]

    # --- Demand transmission multiplier (dynamic calibration) ---
    # Scale beta_is so the model's demand chain matches the RBA benchmark.
    # Computed from posterior medians; applied to all samples.
    if has_is_curve and has_okun_gap:
        demand_multiplier = _compute_demand_multiplier(
            beta_is, rho_is, tau1_okun, tau2_okun, gamma_pi_covid, U_T,
        )
        if demand_multiplier > 1.0:
            beta_is = beta_is * demand_multiplier

    # Output gap at T (for each posterior sample)
    output_gap_T = log_gdp_T - potential_T

    # Unemployment gap at T
    u_gap_T = U_T - nairu_T

    # IS curve historical values (only if IS curve is in the model)
    if has_is_curve:
        real_rate = obs["cash_rate"] - obs["π_exp"]
        rate_gap = real_rate - obs["det_r_star"]
        rate_gap_Tm1 = rate_gap[-2]

    # Rate gap at T onwards - can be overridden for policy scenarios
    cash_rate_T = obs["cash_rate"][-1]
    if cash_rate_override is not None:
        cash_rate_T = cash_rate_override

    if has_is_curve:
        pi_exp_T = obs["π_exp"][-1]
        r_star_T = obs["det_r_star"][-1]
        real_rate_hold = cash_rate_T - pi_exp_T
        rate_gap_hold = real_rate_hold - r_star_T

    # Rate change for FX transmission
    rate_change = cash_rate_T - obs["cash_rate"][-1]  # 0 if no override

    # Inflation expectations (hold at final observed value from signal extraction model)
    pi_exp_qtr = quarterly(obs["π_exp"][-1])

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
    u_gap_prev = u_gap_T
    unemployment_prev = np.full(n_samples, U_T)

    for t in range(h):
        # NAIRU: random walk, stays at current level
        nairu_fcst[t] = nairu_T

        # Potential: grows at Cobb-Douglas drift
        potential_fcst[t] = potential_T + (t + 1) * potential_growth

        # Output gap projection
        if has_is_curve:
            if t == 0:
                rate_gap_lag2 = rate_gap_Tm1
                fiscal_lag1 = obs["fiscal_impulse_1"][-1]
                delta_dsr_lag1 = obs["Δdsr_1"][-1]
                delta_hw_lag1 = obs["Δhw_1"][-1]
                delta_import_price = obs["Δ4ρm_1"][-1]
            else:
                rate_gap_lag2 = rate_gap_hold
                fiscal_lag1 = 0.0
                delta_dsr_lag1 = rate_change * RATE_TO_DSR_PASSTHROUGH
                delta_hw_lag1 = obs["Δhw_1"][-1] + rate_change * RATE_TO_HOUSING_WEALTH

            output_gap_fcst[t] = (
                rho_is * output_gap_prev
                - beta_is * rate_gap_lag2
                + gamma_fi * fiscal_lag1
                - delta_dsr * delta_dsr_lag1
                + eta_hw * delta_hw_lag1
            )
        else:
            # Without IS curve: output gap decays as AR(1)
            if has_okun_gap:
                output_gap_fcst[t] = tau2_okun * output_gap_prev
            else:
                output_gap_fcst[t] = 0.85 * output_gap_prev
            if t == 0:
                delta_import_price = obs["Δ4ρm_1"][-1]

        # Okun's Law: map output gap to unemployment
        if has_okun_gap:
            u_gap_fcst = tau2_okun * u_gap_prev + tau1_okun * output_gap_fcst[t]
            unemployment_fcst[t] = nairu_fcst[t] + u_gap_fcst
        else:
            delta_u = beta_okun * output_gap_fcst[t]
            unemployment_fcst[t] = unemployment_prev + delta_u
            u_gap_fcst = unemployment_fcst[t] - nairu_fcst[t]
            unemployment_prev = unemployment_fcst[t]

        # Phillips curve: π = π_exp + γ × u_gap/U + FX effect
        u_gap_ratio = u_gap_fcst / unemployment_fcst[t]
        fx_inflation_effect = -rate_change * RATE_TO_INFLATION_FX if t > 0 else 0.0
        if t == 0:
            inflation_fcst[t] = pi_exp_qtr + gamma_pi_covid * u_gap_ratio + rho_pi * delta_import_price
        else:
            inflation_fcst[t] = pi_exp_qtr + gamma_pi_covid * u_gap_ratio + fx_inflation_effect

        # Update for next iteration
        output_gap_prev = output_gap_fcst[t]
        u_gap_prev = u_gap_fcst

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
            **({"tau1_okun": float(np.median(tau1_okun)),
                "tau2_okun": float(np.median(tau2_okun))} if has_okun_gap else
               {"beta_okun": float(np.median(beta_okun))}),
            **({"rho_is": float(np.median(rho_is)),
                "beta_is": float(np.median(beta_is)),
                **({"demand_multiplier": demand_multiplier} if has_okun_gap else {}),
                **({} if np.isscalar(gamma_fi) else {"gamma_fi": float(np.median(gamma_fi))}),
                **({} if np.isscalar(delta_dsr) else {"delta_dsr": float(np.median(delta_dsr))}),
                **({} if np.isscalar(eta_hw) else {"eta_hw": float(np.median(eta_hw))}),
                } if has_is_curve else {}),
            gamma_var: float(np.median(gamma_pi_covid)),
        },
        potential_growth=potential_growth,
        nairu_final=float(np.median(nairu_T)),
        potential_final=float(np.median(potential_T)),
        output_gap_final=float(np.median(output_gap_T)),
        output_gap_final_posterior=pd.Series(output_gap_T, index=cols),
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
                "U Gap": f"{(result.unemployment_forecast.iloc[idx]
                            - result.nairu_forecast.iloc[idx]).median():.2f}",
                "π (ann)": f"{annualize(result.inflation_forecast.iloc[idx].median()):.2f}%",
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

    print("\n" + "=" * 80)


def plot_scenario_inflation(
    scenario_results: dict[str, ForecastResults],
    chart_obs: pd.DataFrame | None = None,
    anchor_label: str = "",
    n_history: int = 4,
    chart_dir: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot inflation forecasts across policy scenarios.

    Args:
        scenario_results: Dict of {scenario_name: ForecastResults} from run_scenarios
        chart_obs: Extended observations DataFrame for charting (may extend beyond model sample)
        anchor_label: Label describing expectations anchor mode (for chart footer)
        n_history: Number of historical periods to show before forecast
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

    # Plot historical actuals — use chart_obs if available (extends beyond model sample)
    model_end = scenario_results["hold"].obs_index[-1]
    if chart_obs is not None and "π" in chart_obs.columns:
        cpi = annualize(chart_obs["π"].dropna())
    else:
        cpi = annualize(get_trimmed_mean_qrtly().data)
    hist_actual = cpi.loc[cpi.index >= model_end - n_history + 1]
    hist_actual.name = "Actual"
    ax = mg.line_plot(hist_actual, color="black", width=2)

    # Plot each scenario
    for name in scenario_order:
        if name in df.columns:
            series = df[name]
            series.name = name
            if ax is None:
                ax = mg.line_plot(series, color=colors[name], width=2 if name == "hold" else 1.5)
            else:
                mg.line_plot(series, ax=ax, color=colors[name], width=2 if name == "hold" else 1.5)

    # Add target band and reference line
    if ax is not None:
        ax.axhspan(2.0, 3.0, color="red", alpha=0.1, label="Target band (2-3%)")
        ax.axhline(y=2.5, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="Target (2.5%)")

        # Add error bars at first point to show starting point uncertainty
        # All scenarios have the same first point (before policy propagates)
        hold_result = scenario_results.get("hold")
        if hold_result is not None:
            first_period = hold_result.forecast_index[0]
            # Get 90% HDI for first period inflation
            first_inflation = hold_result.inflation_forecast.iloc[0]
            first_annual = annualize(first_inflation)
            median_val = first_annual.median()
            lower_val = first_annual.quantile(0.05)
            upper_val = first_annual.quantile(0.95)

            # Draw error bar at first point
            ax.errorbar(
                first_period.ordinal,
                median_val,
                yerr=[[median_val - lower_val], [upper_val - median_val]],
                fmt="none",
                ecolor="black",
                elinewidth=2,
                capsize=6,
                capthick=2,
                zorder=10,
                label=f"90% HDI for {first_period} (ex. shocks)",
            )

        # Get current cash rate for title
        current_rate = hold_result.cash_rate if hold_result else None
        rate_str = f" (from {current_rate:.2f}%)" if current_rate is not None else ""

        mg.finalise_plot(
            ax,
            title=f"Inflation Scenarios by Policy Rate{rate_str}",
            ylabel="Per cent per annum",
            legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
            lheader="Trimmed mean, annualised",
            rheader="Scenarios assume RBA moves then holds.",
            lfooter=f"Australia. {anchor_label}." if anchor_label else "Australia.",
            rfooter="Ceteris paribus, no new shocks. RBA-calibrated transmission.",
            show=show,
        )


def plot_scenario_gdp_growth(
    scenario_results: dict[str, ForecastResults],
    obs: dict | None = None,
    obs_index: pd.PeriodIndex | None = None,
    chart_obs: pd.DataFrame | None = None,
    anchor_label: str = "",
    n_history: int = 4,
    chart_dir: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot GDP growth forecasts across policy scenarios.

    Args:
        scenario_results: Dict of {scenario_name: ForecastResults} from run_scenarios
        obs: Observations dict containing historical data
        obs_index: PeriodIndex for observations
        chart_obs: Extended observations DataFrame for charting (may extend beyond model sample)
        anchor_label: Label describing expectations anchor mode (for chart footer)
        n_history: Number of historical periods to show before forecast
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

    # Build DataFrame of annualized GDP growth medians
    gdp_data = {}
    for name in scenario_order:
        if name in scenario_results:
            result = scenario_results[name]
            gdp_growth_qtr = result.gdp_growth_forecast().median(axis=1)
            gdp_growth_annual = annualize(gdp_growth_qtr)
            gdp_data[name] = gdp_growth_annual

    df = pd.DataFrame(gdp_data)

    # Plot historical actuals — use chart_obs if available (extends beyond model sample)
    ax = None
    if chart_obs is not None and "log_gdp" in chart_obs.columns:
        log_gdp = chart_obs["log_gdp"].dropna()
        gdp_growth_qtr = log_gdp.diff()
        gdp_growth_annual = annualize(gdp_growth_qtr)
        model_end = scenario_results["hold"].obs_index[-1]
        hist_recent = gdp_growth_annual.loc[gdp_growth_annual.index >= model_end - n_history + 1]
        hist_recent.name = "Actual"
        ax = mg.line_plot(hist_recent, color="black", width=2)
    elif obs is not None and obs_index is not None:
        log_gdp = pd.Series(obs["log_gdp"], index=obs_index)
        gdp_growth_qtr = log_gdp.diff()
        gdp_growth_annual = annualize(gdp_growth_qtr)
        hist_recent = gdp_growth_annual.iloc[-n_history:]
        hist_recent.name = "Actual"
        ax = mg.line_plot(hist_recent, color="black", width=2)

    # Plot each scenario
    for name in scenario_order:
        if name in df.columns:
            series = df[name]
            series.name = name
            if ax is None:
                ax = mg.line_plot(series, color=colors[name], width=2 if name == "hold" else 1.5)
            else:
                mg.line_plot(series, ax=ax, color=colors[name], width=2 if name == "hold" else 1.5)

    if ax is not None:
        # Get current cash rate and potential growth for context
        hold_result = scenario_results.get("hold")
        current_rate = hold_result.cash_rate if hold_result else None
        potential_growth = hold_result.potential_growth if hold_result else None
        rate_str = f" (from {current_rate:.2f}%)" if current_rate is not None else ""

        # Add reference line at potential growth
        if potential_growth is not None:
            potential_annual = annualize(potential_growth)
            ax.axhline(
                y=potential_annual,
                color="grey",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"Potential growth ({potential_annual:.1f}%)",
            )

        # Add error bars at first point to show starting point uncertainty
        # All scenarios have the same first point (before policy propagates)
        if hold_result is not None:
            first_period = hold_result.forecast_index[0]
            # Get 90% HDI for first period GDP growth
            first_gdp_growth = hold_result.gdp_growth_forecast().iloc[0]
            first_annual = annualize(first_gdp_growth)
            median_val = first_annual.median()
            lower_val = first_annual.quantile(0.05)
            upper_val = first_annual.quantile(0.95)

            # Draw error bar at first point
            ax.errorbar(
                first_period.ordinal,
                median_val,
                yerr=[[median_val - lower_val], [upper_val - median_val]],
                fmt="none",
                ecolor="black",
                elinewidth=2,
                capsize=6,
                capthick=2,
                zorder=10,
                label=f"90% HDI for {first_period} (ex. shocks)",
            )

        mg.finalise_plot(
            ax,
            title=f"GDP Growth Scenarios by Policy Rate{rate_str}",
            ylabel="Per cent per annum",
            legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
            lheader="Annualised quarterly growth",
            rheader="Scenarios assume RBA moves then holds.",
            lfooter=f"Australia. {anchor_label}." if anchor_label else "Australia.",
            rfooter="Ceteris paribus, no new shocks. RBA-calibrated transmission.",
            show=show,
        )


def plot_scenario_unemployment(
    scenario_results: dict[str, ForecastResults],
    obs: dict | None = None,
    obs_index: pd.PeriodIndex | None = None,
    chart_obs: pd.DataFrame | None = None,
    anchor_label: str = "",
    n_history: int = 4,
    chart_dir: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot unemployment scenarios across policy scenarios.

    Args:
        scenario_results: Dict of {scenario_name: ForecastResults} from run_scenarios
        obs: Observations dict containing historical data
        obs_index: PeriodIndex for observations
        chart_obs: Extended observations DataFrame for charting (may extend beyond model sample)
        anchor_label: Label describing expectations anchor mode (for chart footer)
        n_history: Number of historical periods to show before scenario
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

    # Build DataFrame of unemployment medians
    unemployment_data = {}
    for name in scenario_order:
        if name in scenario_results:
            result = scenario_results[name]
            unemployment_median = result.unemployment_forecast.median(axis=1)
            unemployment_data[name] = unemployment_median

    df = pd.DataFrame(unemployment_data)

    # Plot historical actuals — use chart_obs if available (extends beyond model sample)
    ax = None
    if chart_obs is not None and "U" in chart_obs.columns:
        unemployment_hist = chart_obs["U"].dropna()
        model_end = scenario_results["hold"].obs_index[-1]
        hist_recent = unemployment_hist.loc[unemployment_hist.index >= model_end - n_history + 1]
        hist_recent.name = "Actual"
        ax = mg.line_plot(hist_recent, color="black", width=2)
    elif obs is not None and obs_index is not None:
        unemployment_hist = pd.Series(obs["U"], index=obs_index)
        hist_recent = unemployment_hist.iloc[-n_history:]
        hist_recent.name = "Actual"
        ax = mg.line_plot(hist_recent, color="black", width=2)

    # Plot each scenario
    for name in scenario_order:
        if name in df.columns:
            series = df[name]
            series.name = name
            if ax is None:
                ax = mg.line_plot(series, color=colors[name], width=2 if name == "hold" else 1.5)
            else:
                mg.line_plot(series, ax=ax, color=colors[name], width=2 if name == "hold" else 1.5)

    if ax is not None:
        # Get current cash rate and NAIRU for context
        hold_result = scenario_results.get("hold")
        current_rate = hold_result.cash_rate if hold_result else None
        nairu = hold_result.nairu_final if hold_result else None
        rate_str = f" (from {current_rate:.2f}%)" if current_rate is not None else ""

        # Add reference line at NAIRU
        if nairu is not None:
            ax.axhline(
                y=nairu,
                color="grey",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"NAIRU ({nairu:.1f}%)",
            )

        # Add error bars at first point to show starting point uncertainty
        if hold_result is not None:
            first_period = hold_result.forecast_index[0]
            # Get 90% HDI for first period unemployment
            first_unemployment = hold_result.unemployment_forecast.iloc[0]
            median_val = first_unemployment.median()
            lower_val = first_unemployment.quantile(0.05)
            upper_val = first_unemployment.quantile(0.95)

            # Draw error bar at first point
            ax.errorbar(
                first_period.ordinal,
                median_val,
                yerr=[[median_val - lower_val], [upper_val - median_val]],
                fmt="none",
                ecolor="black",
                elinewidth=2,
                capsize=6,
                capthick=2,
                zorder=10,
                label=f"90% HDI for {first_period} (ex. shocks)",
            )

        mg.finalise_plot(
            ax,
            title=f"Unemployment Scenarios by Policy Rate{rate_str}",
            ylabel="Per cent",
            legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
            lheader="Unemployment rate. Responds more slowly and over longer horizons than inflation.",
            rheader="Scenarios assume RBA moves then holds.",
            lfooter=f"Australia. {anchor_label}." if anchor_label else "Australia.",
            rfooter="Ceteris paribus, no new shocks. RBA-calibrated transmission.",
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
    trace, obs, obs_index, constants, anchor_label, chart_obs, model_kwargs = load_results(output_dir=output_dir, prefix=prefix)

    # Rebuild model and create results container
    model = build_model(obs, **model_kwargs)
    results = NAIRUResults(
        trace=trace,
        obs=obs,
        obs_index=obs_index,
        model=model,
        anchor_label=anchor_label,
        chart_obs=chart_obs,
    )

    current_rate = obs["cash_rate"][-1]

    # Check if model has full transmission channels for credible scenarios.
    # Each channel can be toggled independently; all are needed for the
    # rate → output gap → unemployment → inflation pass-through to be realistic.
    _required_channels = {
        "gamma_pi_covid": "regime-switching Phillips curve",
        "beta_is": "IS curve (rate → output gap)",
        "beta_er_r": "exchange rate (UIP)",
        "beta_pt": "import price pass-through",
        "beta_pr": "participation (discouraged worker)",
    }
    _missing = {v for k, v in _required_channels.items() if k not in trace.posterior}
    if run_scenarios_flag and _missing:
        print(
            "Skipping policy scenarios: model missing channels needed for "
            f"credible transmission: {', '.join(sorted(_missing))}."
        )
        run_scenarios_flag = False

    if run_scenarios_flag:
        # Run all policy scenarios
        scenario_results = run_scenarios(results)

        # Print baseline summary
        baseline = scenario_results["hold"]
        baseline.print_summary()

        # Print scenario comparison
        print_scenario_comparison(scenario_results, current_rate)

        # Plot scenario comparisons
        if plot_scenarios:
            plot_scenario_inflation(scenario_results, chart_obs=chart_obs, anchor_label=anchor_label, chart_dir=chart_dir, show=False)
            plot_scenario_gdp_growth(
                scenario_results, obs=obs, obs_index=obs_index, chart_obs=chart_obs, anchor_label=anchor_label, chart_dir=chart_dir, show=False
            )
            plot_scenario_unemployment(
                scenario_results, obs=obs, obs_index=obs_index, chart_obs=chart_obs, anchor_label=anchor_label, chart_dir=chart_dir, show=False
            )

        return scenario_results
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
