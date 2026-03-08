"""Model-consistent forecasting with policy scenarios.

Generates 4-quarter ahead forecasts by sampling forward from estimated
model coefficients and innovation distributions. Each posterior draw
gets its own forward path with sampled shocks, giving proper uncertainty.

The median path is the central forecast; HDI bands show the range of
outcomes under normal operating conditions (typical noise, no new
structural breaks).

Calibrations applied (RBA-based):
- FX channel: 0.35pp inflation per 100bp (RBA Bulletin April 2025)
- Demand multiplier: dynamically calibrated from posterior
- DSR pass-through: 1.0pp per 100bp
- Housing wealth: -1.0% per 100bp
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.models.common.extraction import get_scalar_var
from src.models.nairu.results import NAIRUResults, load_results
from src.utilities.rate_conversion import annualize, quarterly

# Forecast horizon (4 quarters, move-and-hold assumption for rates)
FORECAST_HORIZON = 4

# Number of posterior samples (5000 is sufficient for stable HDI)
N_SAMPLES = 5000

# Student-t degrees of freedom for NAIRU innovations (matches nairu.py)
NAIRU_NU = 4

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

# Scenario ordering and colors for plotting
SCENARIO_ORDER = ["+200bp", "+100bp", "+50bp", "+25bp", "hold",
                  "-25bp", "-50bp", "-100bp", "-200bp"]
SCENARIO_COLORS = {
    "+200bp": "darkred", "+100bp": "red", "+50bp": "orangered",
    "+25bp": "orange", "hold": "black", "-25bp": "deepskyblue",
    "-50bp": "steelblue", "-100bp": "blue", "-200bp": "darkblue",
}

# --- Calibration constants (RBA-based) ---

# FX channel: 100bp rate increase -> 0.25-0.5pp lower inflation over 2 years
# We use 0.35pp (midpoint) distributed across forecast horizon
# Assumes RBA moves unilaterally (Fed holds steady)
FX_PASSTHROUGH_PER_100BP = 0.35 / 4

# DSR pass-through: ~1.0pp per 100bp
# $2.2T housing debt x 70% variable x 1% / $1.6T disposable income
DSR_PASSTHROUGH_PER_100BP = 1.0

# Housing wealth: 100bp -> ~1% decline in housing wealth growth per quarter
HOUSING_WEALTH_PASSTHROUGH_PER_100BP = -1.0

# RBA demand channel benchmark: ~0.29pp inflation per 100bp (ex-FX)
RBA_DEMAND_CHANNEL_PP = 0.29

# Required trace parameters for credible scenario analysis
REQUIRED_CHANNELS = {
    "beta_is": "IS curve (rate -> output gap)",
    "beta_er_r": "exchange rate (UIP)",
    "beta_pt": "import price pass-through",
    "beta_pr": "participation (discouraged worker)",
}


def _compute_demand_multiplier(
    beta_is: np.ndarray,
    rho_is: np.ndarray,
    tau1_okun: np.ndarray,
    tau2_okun: np.ndarray,
    gamma_pi: np.ndarray,
    u_level: float,
    h: int = 8,
) -> float:
    """Compute demand transmission multiplier from posterior medians.

    Simulates a 100bp shock through the estimated demand chain:
        rate gap -> output gap (IS) -> unemployment gap (Okun) -> inflation (Phillips)
    over 8 quarters and compares to RBA benchmark (0.29pp per 100bp).

    Returns:
        Multiplier to apply to beta_is.

    """
    b_is = float(np.median(beta_is))
    r_is = float(np.median(rho_is))
    t1 = float(np.median(tau1_okun))
    t2 = float(np.median(tau2_okun))
    g_pi = float(np.median(gamma_pi))

    og = 0.0
    ug = 0.0
    cumulative_inflation = 0.0

    for t in range(h):
        rate_effect = 1.0 if t >= 2 else 0.0
        og = r_is * og - b_is * rate_effect
        ug = t2 * ug + t1 * og
        cumulative_inflation += g_pi * (ug / u_level)

    model_demand_pp = abs(cumulative_inflation)
    if model_demand_pp < 1e-6:
        return 2.0

    return RBA_DEMAND_CHANNEL_PP / model_demand_pp


# --- Results container ---


@dataclass
class ForecastResults:
    """Container for forecast results (one scenario)."""

    scenario_name: str
    cash_rate: float
    forecast_index: pd.PeriodIndex
    obs_index: pd.PeriodIndex

    # Posterior samples (rows=periods, cols=samples)
    nairu_forecast: pd.DataFrame
    potential_forecast: pd.DataFrame
    output_gap_forecast: pd.DataFrame
    unemployment_forecast: pd.DataFrame
    inflation_forecast: pd.DataFrame  # quarterly

    # Context
    nairu_final: float
    potential_final: float
    log_gdp_final: float
    unemployment_final: float
    potential_growth: float
    coefficients: dict

    def _quantiles(self, df: pd.DataFrame, prob: float = 0.90) -> pd.DataFrame:
        lower = (1 - prob) / 2
        upper = 1 - lower
        return pd.DataFrame(
            {"lower": df.quantile(lower, axis=1),
             "median": df.median(axis=1),
             "upper": df.quantile(upper, axis=1)},
            index=df.index,
        )

    def output_gap_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.output_gap_forecast, prob)

    def unemployment_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.unemployment_forecast, prob)

    def unemployment_gap_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.unemployment_forecast - self.nairu_forecast, prob)

    def inflation_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.inflation_forecast, prob)

    def inflation_annual_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(annualize(self.inflation_forecast), prob)

    def output_samples(self) -> pd.DataFrame:
        """GDP (log) = potential + output_gap."""
        return self.potential_forecast + self.output_gap_forecast

    def output_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.output_samples(), prob)

    def gdp_growth_forecast(self) -> pd.DataFrame:
        """Quarterly GDP growth = potential_growth + delta(output_gap)."""
        og_change = self.output_gap_forecast.diff()
        og_change.iloc[0] = (
            self.output_gap_forecast.iloc[0]
            - (self.log_gdp_final - self.potential_final)
        )
        return self.potential_growth + og_change

    def gdp_growth_annual_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(annualize(self.gdp_growth_forecast()), prob)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Output Gap": self.output_gap_forecast.median(axis=1),
                "U": self.unemployment_forecast.median(axis=1),
                "NAIRU": self.nairu_forecast.median(axis=1),
                "U Gap": (self.unemployment_forecast - self.nairu_forecast).median(axis=1),
                "pi (ann)": annualize(self.inflation_forecast).median(axis=1),
            },
            index=self.forecast_index,
        )

    def print_summary(self) -> None:
        print(f"\n{'=' * 70}")
        print(f"FORECAST: {self.scenario_name} (cash rate {self.cash_rate:.2f}%)")
        print(f"{'=' * 70}")
        print(f"\nStarting point ({self.obs_index[-1]}):")
        print(f"  Output gap: {self.log_gdp_final - self.potential_final:+.4f}")
        print(f"  Unemployment: {self.unemployment_final:.2f}%")
        print(f"  NAIRU: {self.nairu_final:.2f}%")
        print(f"\n{self.summary().round(3).to_string()}")
        print(f"{'=' * 70}")


# --- Forecasting ---


def forecast(
    results: NAIRUResults,
    cash_rate_override: float | None = None,
    scenario_name: str = "baseline",
    n_samples: int = N_SAMPLES,
    seed: int = 42,
) -> ForecastResults:
    """Generate forward-sampled forecast with uncertainty.

    For each posterior draw, samples future innovations and propagates
    the state equations forward. Gives proper uncertainty bands.

    """
    rng = np.random.default_rng(seed)
    obs = results.obs
    obs_index = results.obs_index
    trace = results.trace
    h = FORECAST_HORIZON

    # --- Extract posterior samples ---
    nairu_posterior = results.nairu_posterior()
    potential_posterior = results.potential_posterior()

    # Subsample if needed
    total_samples = nairu_posterior.shape[1]
    if n_samples < total_samples:
        idx = rng.choice(total_samples, size=n_samples, replace=False)
        nairu_posterior = nairu_posterior.iloc[:, idx]
        potential_posterior = potential_posterior.iloc[:, idx]
    else:
        n_samples = total_samples

    # --- Coefficient extraction helper ---
    def get_coef(name: str) -> np.ndarray:
        arr = get_scalar_var(name, trace).to_numpy()
        if len(arr) != n_samples:
            arr = arr[rng.choice(len(arr), size=n_samples, replace=True)]
        return arr

    # Okun coefficients
    has_okun_gap = "tau1_okun" in trace.posterior
    if has_okun_gap:
        tau1_okun = get_coef("tau1_okun")
        tau2_okun = get_coef("tau2_okun")
    else:
        beta_okun = get_coef("beta_okun")

    # IS curve coefficients
    has_is_curve = "rho_is" in trace.posterior
    if has_is_curve:
        rho_is = get_coef("rho_is")
        beta_is = get_coef("beta_is")
        gamma_fi = get_coef("gamma_fi") if "gamma_fi" in trace.posterior else 0.0
        delta_dsr = get_coef("delta_dsr") if "delta_dsr" in trace.posterior else 0.0
        eta_hw = get_coef("eta_hw") if "eta_hw" in trace.posterior else 0.0

    # Phillips slope (use covid regime if available)
    gamma_var = "gamma_pi_covid" if "gamma_pi_covid" in trace.posterior else "gamma_pi"
    gamma_pi = get_coef(gamma_var)

    # Import price -> inflation
    rho_pi = get_coef("rho_pi") if "rho_pi" in trace.posterior else 0.0

    # Innovation scales
    sigma_is = get_coef("epsilon_is") if has_is_curve else np.zeros(n_samples)
    sigma_okun = get_coef("epsilon_okun")
    sigma_pi = get_coef("epsilon_pi")
    sigma_nairu = results.constants.get("nairu_innovation", 0.25)
    has_student_t_nairu = sigma_nairu < 0.2
    sigma_potential = results.constants.get("potential_innovation", 0.30)

    # --- Demand multiplier ---
    if has_is_curve and has_okun_gap:
        demand_multiplier = _compute_demand_multiplier(
            beta_is, rho_is, tau1_okun, tau2_okun, gamma_pi, obs["U"][-1],
        )
        if demand_multiplier > 1.0:
            beta_is = beta_is * demand_multiplier

    # --- Historical values ---
    nairu_T = nairu_posterior.iloc[-1].to_numpy()
    potential_T = potential_posterior.iloc[-1].to_numpy()
    log_gdp_T = obs["log_gdp"][-1]
    U_T = obs["U"][-1]
    output_gap_T = log_gdp_T - potential_T
    u_gap_T = U_T - nairu_T

    # Rate gap
    cash_rate_T = obs["cash_rate"][-1]
    if cash_rate_override is not None:
        cash_rate_T = cash_rate_override
    rate_change = cash_rate_T - obs["cash_rate"][-1]

    if has_is_curve:
        real_rate = obs["cash_rate"] - obs["π_exp"]
        rate_gap = real_rate - obs["det_r_star"]
        rate_gap_Tm1 = rate_gap[-2]
        real_rate_hold = cash_rate_T - obs["π_exp"][-1]
        rate_gap_hold = real_rate_hold - obs["det_r_star"][-1]

    pi_exp_qtr = quarterly(obs["π_exp"][-1])

    # Potential growth (Cobb-Douglas drift)
    alpha = float(obs["alpha_capital"][-1])
    g_K = float(obs["capital_growth"][-1])
    g_L = float(obs["lf_growth"][-1])
    g_MFP = float(obs["mfp_growth"][-1])
    potential_growth = alpha * g_K + (1 - alpha) * g_L + g_MFP

    # Detect skewnormal potential
    has_skewnormal = sigma_potential < 0.08
    if has_skewnormal:
        skew_alpha = 1.0
        delta = skew_alpha / np.sqrt(1 + skew_alpha**2)
        skew_loc = -sigma_potential * delta * np.sqrt(2 / np.pi)

    # --- Forecast index ---
    last_period = obs_index[-1]
    forecast_index = pd.period_range(start=last_period + 1, periods=h, freq=last_period.freq)

    # --- Forward sample ---
    nairu_fcst = np.zeros((h, n_samples))
    potential_fcst = np.zeros((h, n_samples))
    output_gap_fcst = np.zeros((h, n_samples))
    unemployment_fcst = np.zeros((h, n_samples))
    inflation_fcst = np.zeros((h, n_samples))

    nairu_prev = nairu_T.copy()
    potential_prev = potential_T.copy()
    output_gap_prev = output_gap_T.copy()
    u_gap_prev = u_gap_T.copy()
    unemployment_prev = np.full(n_samples, U_T)

    for t in range(h):
        # Sample innovations
        if has_student_t_nairu:
            eps_nairu = rng.standard_t(df=NAIRU_NU, size=n_samples) * sigma_nairu
        else:
            eps_nairu = rng.normal(0, sigma_nairu, size=n_samples)
        if has_skewnormal:
            eps_potential = stats.skewnorm.rvs(
                a=skew_alpha, loc=skew_loc, scale=sigma_potential,
                size=n_samples, random_state=rng,
            )
        else:
            eps_potential = rng.normal(0, sigma_potential, size=n_samples)
        eps_is = rng.normal(0, sigma_is) if has_is_curve else 0.0
        eps_okun = rng.normal(0, sigma_okun)
        eps_pi = rng.normal(0, sigma_pi)

        # NAIRU: random walk
        nairu_fcst[t] = nairu_prev + eps_nairu
        nairu_prev = nairu_fcst[t]

        # Potential: Cobb-Douglas drift + innovation
        potential_fcst[t] = potential_prev + potential_growth + eps_potential
        potential_prev = potential_fcst[t]

        # Output gap
        if has_is_curve:
            if t == 0:
                rg_lag2 = rate_gap_Tm1
                fiscal_lag1 = obs["fiscal_impulse_1"][-1]
                dsr_lag1 = obs["Δdsr_1"][-1]
                hw_lag1 = obs["Δhw_1"][-1]
                import_effect = rho_pi * obs["Δ4ρm_1"][-1]
            else:
                rg_lag2 = rate_gap_hold
                fiscal_lag1 = 0.0
                dsr_lag1 = rate_change * DSR_PASSTHROUGH_PER_100BP
                hw_lag1 = obs["Δhw_1"][-1] + rate_change * HOUSING_WEALTH_PASSTHROUGH_PER_100BP
                import_effect = -rate_change * FX_PASSTHROUGH_PER_100BP

            output_gap_fcst[t] = (
                rho_is * output_gap_prev
                - beta_is * rg_lag2
                + gamma_fi * fiscal_lag1
                - delta_dsr * dsr_lag1
                + eta_hw * hw_lag1
                + eps_is
            )
        else:
            if has_okun_gap:
                output_gap_fcst[t] = tau2_okun * output_gap_prev
            else:
                output_gap_fcst[t] = 0.85 * output_gap_prev
            if t == 0:
                import_effect = rho_pi * obs["Δ4ρm_1"][-1]
            else:
                import_effect = -rate_change * FX_PASSTHROUGH_PER_100BP
        output_gap_prev = output_gap_fcst[t]

        # Okun's Law
        if has_okun_gap:
            u_gap_fcst = tau2_okun * u_gap_prev + tau1_okun * output_gap_fcst[t] + eps_okun
            unemployment_fcst[t] = nairu_fcst[t] + u_gap_fcst
        else:
            delta_u = beta_okun * output_gap_fcst[t] + eps_okun
            unemployment_fcst[t] = unemployment_prev + delta_u
            u_gap_fcst = unemployment_fcst[t] - nairu_fcst[t]
            unemployment_prev = unemployment_fcst[t]
        u_gap_prev = u_gap_fcst

        # Phillips curve
        u_gap_ratio = u_gap_fcst / unemployment_fcst[t]
        inflation_fcst[t] = pi_exp_qtr + gamma_pi * u_gap_ratio + import_effect + eps_pi

    # --- Package ---
    cols = [f"s{i}" for i in range(n_samples)]

    return ForecastResults(
        scenario_name=scenario_name,
        cash_rate=cash_rate_T,
        forecast_index=forecast_index,
        obs_index=obs_index,
        nairu_forecast=pd.DataFrame(nairu_fcst, index=forecast_index, columns=cols),
        potential_forecast=pd.DataFrame(potential_fcst, index=forecast_index, columns=cols),
        output_gap_forecast=pd.DataFrame(output_gap_fcst, index=forecast_index, columns=cols),
        unemployment_forecast=pd.DataFrame(unemployment_fcst, index=forecast_index, columns=cols),
        inflation_forecast=pd.DataFrame(inflation_fcst, index=forecast_index, columns=cols),
        nairu_final=float(np.median(nairu_T)),
        potential_final=float(np.median(potential_T)),
        log_gdp_final=log_gdp_T,
        unemployment_final=U_T,
        potential_growth=potential_growth,
        coefficients={
            **({"tau1_okun": float(np.median(tau1_okun)),
                "tau2_okun": float(np.median(tau2_okun))} if has_okun_gap else
               {"beta_okun": float(np.median(beta_okun))}),
            **({"rho_is": float(np.median(rho_is)),
                "beta_is": float(np.median(beta_is))} if has_is_curve else {}),
            gamma_var: float(np.median(gamma_pi)),
        },
    )


# --- Scenario runners ---


def run_scenarios(
    results: NAIRUResults,
    scenarios: dict[str, float] | None = None,
    n_samples: int = N_SAMPLES,
) -> dict[str, ForecastResults]:
    """Run multiple policy scenarios.

    Args:
        results: NAIRUResults from estimation
        scenarios: {name: bp_change} from current rate
        n_samples: Number of posterior samples per scenario

    Returns:
        {scenario_name: ForecastResults}

    """
    if scenarios is None:
        scenarios = DEFAULT_POLICY_SCENARIOS

    current_rate = results.obs["cash_rate"][-1]

    return {
        name: forecast(
            results,
            cash_rate_override=current_rate + bp_change,
            scenario_name=name,
            n_samples=n_samples,
        )
        for name, bp_change in scenarios.items()
    }


def check_transmission_channels(trace) -> set[str]:
    """Check which required channels are missing from the trace."""
    return {v for k, v in REQUIRED_CHANNELS.items() if k not in trace.posterior}


def print_scenario_comparison(
    scenario_results: dict[str, ForecastResults],
    current_rate: float,
) -> None:
    """Print comparison table across scenarios."""
    print(f"\n{'=' * 80}")
    print(f"SCENARIO COMPARISON (cash rate: {current_rate:.2f}%)")
    print(f"{'=' * 80}")

    first_result = next(iter(scenario_results.values()))

    for period in first_result.forecast_index:
        print(f"\n{period}:")
        print("-" * 80)

        rows = []
        for name in SCENARIO_ORDER:
            if name not in scenario_results:
                continue
            r = scenario_results[name]
            idx = r.forecast_index.get_loc(period)
            og = r.output_gap_forecast.iloc[idx]
            u = r.unemployment_forecast.iloc[idx]
            nairu = r.nairu_forecast.iloc[idx]
            pi_ann = annualize(r.inflation_forecast.iloc[idx])

            rows.append({
                "Scenario": name,
                "Rate": f"{r.cash_rate:.2f}%",
                "OG": f"{og.median():.3f} [{og.quantile(0.05):.2f},{og.quantile(0.95):.2f}]",
                "U": f"{u.median():.2f}%",
                "U Gap": f"{(u - nairu).median():+.2f}",
                "pi (ann)": f"{pi_ann.median():.2f}% [{pi_ann.quantile(0.05):.1f},{pi_ann.quantile(0.95):.1f}]",
            })

        print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n{'=' * 80}")


# --- Entry point ---


def run_forecast(
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
    n_samples: int = N_SAMPLES,
    chart_dir: Path | str | None = None,
    verbose: bool = False,
    show_plots: bool = False,
) -> dict[str, ForecastResults] | None:
    """Run forecast scenarios and generate charts.

    Args:
        output_dir: Directory containing saved results
        prefix: Filename prefix
        n_samples: Number of posterior samples
        chart_dir: Override chart directory
        verbose: Print detailed output
        show_plots: Display plots interactively

    Returns:
        Dict of scenario results, or None if channels are missing

    """
    results = load_results(output_dir=output_dir, prefix=prefix, rebuild_model=False)
    config = results.config

    # Check transmission channels
    missing = check_transmission_channels(results.trace)
    if missing:
        print(f"Skipping scenarios: missing {', '.join(sorted(missing))}.")
        return None

    current_rate = results.obs["cash_rate"][-1]
    print(f"Running scenarios (n={n_samples}, rate={current_rate:.2f}%)...")

    scenario_results = run_scenarios(results, n_samples=n_samples)

    # Print hold summary
    scenario_results["hold"].print_summary()
    print_scenario_comparison(scenario_results, current_rate)

    # Plot
    from src.models.nairu.forecast_plots import plot_all_scenarios

    if chart_dir is None:
        from src.models.nairu.results import DEFAULT_CHART_BASE
        chart_dir = DEFAULT_CHART_BASE / config.chart_dir_name

    plot_all_scenarios(
        scenario_results,
        results=results,
        chart_dir=chart_dir,
        rfooter=config.rfooter,
        show=show_plots,
    )

    return scenario_results
