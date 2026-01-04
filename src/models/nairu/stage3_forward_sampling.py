"""NAIRU + Output Gap Stage 3: Bayesian Forward Sampling.

This module generates probabilistic scenario analysis by sampling forward
from the estimated model, rather than deterministic projection.

Key differences from stage3.py (deterministic):
- Samples future innovations from estimated distributions (not "no shocks")
- Propagates full uncertainty (parameters + states + shocks)
- Paths may or may not converge to equilibrium

Calibrations applied (RBA-based, defensible):
- FX channel: 0.35pp inflation per 100bp (RBA Bulletin April 2025)
- Demand multiplier: 1.6x on Okun coefficient (aligns with RBA demand channel)
- DSR pass-through: 1.0pp per 100bp
- Housing wealth: -1.0% per 100bp

Interpretation: "What does the model think happens next, with realistic transmission?"
"""

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import mgplot as mg
import numpy as np
import pandas as pd
from scipy import stats

from src.models.nairu.analysis import get_scalar_var
from src.models.nairu.stage2 import load_results, NAIRUResults, build_model
from src.utilities.rate_conversion import quarterly, annualize

# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"
DEFAULT_CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "nairu_output_gap"

# Forecast horizon
FORECAST_HORIZON = 4

# Number of posterior samples to use (5000 is sufficient for stable HDI estimates)
N_SAMPLES = 5000

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

# =============================================================================
# CALIBRATION PARAMETERS (RBA-based, defensible)
# =============================================================================

# FX channel pass-through (RBA Bulletin April 2025)
# 100bp rate increase → 5-10% TWI appreciation → 0.25-0.5pp lower inflation over 2 years
# We use 0.35pp (midpoint) distributed across forecast horizon
# Note: Assumes RBA moves unilaterally (Fed holds steady)
FX_PASSTHROUGH_PER_100BP = 0.35 / 4  # pp inflation per quarter per 100bp

# Demand transmission multiplier (RBA research alignment)
# Model's Okun coefficient underestimates total demand channel because:
# - Expectations channel not explicitly modeled
# - Credit/lending channel not modeled
# - Business cash flow effects not modeled
# RBA estimates demand channel ~0.29pp vs model's ~0.18pp → multiplier ~1.6
DEMAND_MULTIPLIER = 1.6

# DSR pass-through: ~1.0pp DSR increase per 100bp rate rise
# Derivation: $2.2T housing debt × 70% variable × 1% / $1.6T disposable income
DSR_PASSTHROUGH_PER_100BP = 1.0

# Housing wealth pass-through: ~1% decline in housing wealth growth per 100bp
# RBA estimates 100bp → 2-4% house price decline (annualized), ~1%/quarter
HOUSING_WEALTH_PASSTHROUGH_PER_100BP = -1.0


@dataclass
class BayesianScenarioResults:
    """Container for Bayesian forward sampling results."""

    scenario_name: str
    cash_rate: float
    forecast_index: pd.PeriodIndex
    obs_index: pd.PeriodIndex

    # Posterior predictive samples (rows=periods, cols=samples)
    nairu_samples: pd.DataFrame
    potential_samples: pd.DataFrame
    output_gap_samples: pd.DataFrame
    unemployment_samples: pd.DataFrame
    inflation_samples: pd.DataFrame  # quarterly

    # Final historical values
    nairu_final: float
    unemployment_final: float
    potential_final: float
    log_gdp_final: float

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
        return self._quantiles(self.nairu_samples, prob)

    def output_gap_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.output_gap_samples, prob)

    def unemployment_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.unemployment_samples, prob)

    def inflation_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.inflation_samples, prob)

    def inflation_annual_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        annual = annualize(self.inflation_samples)
        return self._quantiles(annual, prob)

    def output_samples(self) -> pd.DataFrame:
        """GDP (log) = potential + output_gap."""
        return self.potential_samples + self.output_gap_samples

    def output_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.output_samples(), prob)

    def potential_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        return self._quantiles(self.potential_samples, prob)

    def summary(self) -> pd.DataFrame:
        """Point estimates (posterior medians)."""
        return pd.DataFrame(
            {
                "Output Gap": self.output_gap_samples.median(axis=1),
                "U": self.unemployment_samples.median(axis=1),
                "NAIRU": self.nairu_samples.median(axis=1),
                "U Gap": (self.unemployment_samples - self.nairu_samples).median(axis=1),
                "π (ann)": annualize(self.inflation_samples).median(axis=1),
            },
            index=self.forecast_index,
        )


def sample_skewnormal(loc: float, scale: float, alpha: float, size: int) -> np.ndarray:
    """Sample from skew-normal distribution.

    Uses scipy's skewnorm which parameterizes differently from PyMC.
    PyMC: SkewNormal(mu, sigma, alpha)
    scipy: skewnorm(a) with loc and scale

    For our zero-mean adjusted SkewNormal with alpha=1:
    - We shift location to achieve E[X] = 0
    """
    # scipy skewnorm: a is the shape parameter (same as alpha)
    return stats.skewnorm.rvs(a=alpha, loc=loc, scale=scale, size=size)


def bayesian_forward_sample(
    results: NAIRUResults,
    cash_rate_override: float | None = None,
    scenario_name: str = "baseline",
    n_samples: int = N_SAMPLES,
    seed: int = 42,
) -> BayesianScenarioResults:
    """Generate Bayesian forward samples for scenario analysis.

    For each posterior sample of parameters and final states, samples
    future innovations and propagates the state equations forward.

    Args:
        results: NAIRUResults from stage2
        cash_rate_override: Override cash rate for scenario
        scenario_name: Name for this scenario
        n_samples: Number of samples to use (default: 5000)
        seed: Random seed for reproducibility

    Returns:
        BayesianScenarioResults with posterior predictive distributions

    """
    np.random.seed(seed)

    obs = results.obs
    obs_index = results.obs_index
    trace = results.trace
    h = FORECAST_HORIZON

    # --- Extract posterior samples ---
    nairu_posterior = results.nairu_posterior()
    potential_posterior = results.potential_posterior()

    # Subsample posterior if n_samples < total available
    total_samples = nairu_posterior.shape[1]
    if n_samples < total_samples:
        idx = np.random.choice(total_samples, size=n_samples, replace=False)
        nairu_posterior = nairu_posterior.iloc[:, idx]
        potential_posterior = potential_posterior.iloc[:, idx]
    else:
        n_samples = total_samples

    # --- Extract coefficient posteriors ---
    def get_coef(name: str) -> np.ndarray:
        arr = get_scalar_var(name, trace).values
        if len(arr) != n_samples:
            # Resample to match
            idx = np.random.choice(len(arr), size=n_samples, replace=True)
            arr = arr[idx]
        return arr

    rho_is = get_coef("rho_is")
    beta_is = get_coef("beta_is")
    delta_dsr = get_coef("delta_dsr")
    eta_hw = get_coef("eta_hw")
    beta_okun = get_coef("beta_okun")
    alpha_okun = get_coef("alpha_okun")
    gamma_okun = get_coef("gamma_okun")
    gamma_pi_covid = get_coef("gamma_pi_covid")

    # Innovation scales (observation equation residuals)
    sigma_is = get_coef("epsilon_is")
    sigma_okun = get_coef("epsilon_okun")
    sigma_pi = get_coef("epsilon_pi")

    # State innovation scales (fixed in estimation, but we use them for sampling)
    # From model constants
    sigma_nairu = 0.25  # Fixed in estimation
    sigma_potential = 0.30  # Fixed in estimation

    # --- Historical values needed ---
    nairu_T = nairu_posterior.iloc[-1].values
    potential_T = potential_posterior.iloc[-1].values
    log_gdp_T = obs["log_gdp"][-1]
    U_T = obs["U"][-1]
    output_gap_T = log_gdp_T - potential_T

    # Rate gaps
    real_rate = obs["cash_rate"] - obs["π_anchor"]
    rate_gap = real_rate - obs["det_r_star"]
    rate_gap_Tm1 = rate_gap[-2]

    # Current/scenario cash rate
    cash_rate_T = obs["cash_rate"][-1]
    if cash_rate_override is not None:
        cash_rate_T = cash_rate_override

    # Rate gap under scenario
    pi_anchor_T = obs["π_anchor"][-1]
    r_star_T = obs["det_r_star"][-1]
    real_rate_hold = cash_rate_T - pi_anchor_T
    rate_gap_hold = real_rate_hold - r_star_T

    # DSR and housing wealth changes (historical)
    delta_dsr_hist = obs["Δdsr_1"][-1]
    delta_hw_hist = obs["Δhw_1"][-1]

    # Historical import price change (for t=0 inflation)
    delta_import_price = obs["Δ4ρm_1"][-1]
    rho_pi = get_coef("rho_pi")  # import price pass-through coefficient

    # Rate change for transmission
    rate_change = cash_rate_T - obs["cash_rate"][-1]

    # Inflation anchor (quarterly)
    pi_anchor_qtr = quarterly(2.5)

    # Potential growth (Cobb-Douglas components - last Q values held constant)
    # "As-is" assumption: current trajectory continues without structural changes
    alpha = float(obs["alpha_capital"][-1])
    g_K = float(obs["capital_growth"][-1])  # Last Q capital growth
    g_L = float(obs["lf_growth"][-1])        # Last Q labor force growth
    g_MFP = float(obs["mfp_growth"][-1])     # Last Q MFP growth
    potential_growth = alpha * g_K + (1 - alpha) * g_L + g_MFP

    # SkewNormal parameters for potential output innovations
    # From production.py: alpha=1, sigma=0.3, zero-mean adjusted
    skew_alpha = 1.0
    skew_sigma = sigma_potential
    # Zero-mean adjustment: shift location so E[X] = 0
    # E[SkewNormal(mu, sigma, alpha)] = mu + sigma * delta * sqrt(2/pi)
    # where delta = alpha / sqrt(1 + alpha^2)
    delta = skew_alpha / np.sqrt(1 + skew_alpha**2)
    mean_shift = skew_sigma * delta * np.sqrt(2 / np.pi)
    skew_loc = -mean_shift  # Shift so mean is zero

    # --- Create forecast index ---
    last_period = obs_index[-1]
    forecast_index = pd.period_range(
        start=last_period + 1,
        periods=h,
        freq=last_period.freq,
    )

    # --- Allocate arrays ---
    nairu_fcst = np.zeros((h, n_samples))
    potential_fcst = np.zeros((h, n_samples))
    output_gap_fcst = np.zeros((h, n_samples))
    unemployment_fcst = np.zeros((h, n_samples))
    inflation_fcst = np.zeros((h, n_samples))

    # --- Forward sample for each posterior draw ---
    nairu_prev = nairu_T.copy()
    potential_prev = potential_T.copy()
    output_gap_prev = output_gap_T.copy()
    U_prev = np.full(n_samples, U_T)

    for t in range(h):
        # Save lags for error correction (before they get updated)
        nairu_lag = nairu_prev.copy()
        og_lag = output_gap_prev.copy()

        # Sample innovations
        eps_nairu = np.random.normal(0, sigma_nairu, n_samples)
        eps_potential = sample_skewnormal(skew_loc, skew_sigma, skew_alpha, n_samples)
        eps_is = np.random.normal(0, sigma_is)
        eps_okun = np.random.normal(0, sigma_okun)
        eps_pi = np.random.normal(0, sigma_pi)

        # NAIRU: random walk
        nairu_fcst[t] = nairu_prev + eps_nairu
        nairu_prev = nairu_fcst[t]

        # Potential: Cobb-Douglas drift + innovation
        potential_fcst[t] = potential_prev + potential_growth + eps_potential
        potential_prev = potential_fcst[t]

        # IS curve
        if t == 0:
            rate_gap_lag2 = rate_gap_Tm1
            delta_dsr_lag1 = delta_dsr_hist
            delta_hw_lag1 = delta_hw_hist
            import_price_effect = rho_pi * delta_import_price  # Historical import prices
        else:
            rate_gap_lag2 = rate_gap_hold
            # DSR pass-through (RBA calibration)
            delta_dsr_lag1 = rate_change * DSR_PASSTHROUGH_PER_100BP
            # Housing wealth pass-through (RBA calibration)
            delta_hw_lag1 = delta_hw_hist + rate_change * HOUSING_WEALTH_PASSTHROUGH_PER_100BP
            # FX channel effect on inflation (RBA calibration)
            import_price_effect = -rate_change * FX_PASSTHROUGH_PER_100BP

        output_gap_fcst[t] = (
            rho_is * output_gap_prev
            - beta_is * rate_gap_lag2
            - delta_dsr * delta_dsr_lag1
            + eta_hw * delta_hw_lag1
            + eps_is
        )
        output_gap_prev = output_gap_fcst[t]

        # Okun's Law (error correction form):
        # ΔU = β × OG + α × (U_{t-1} - NAIRU_{t-1} - γ × OG_{t-1})
        # α < 0: when U > equilibrium, unemployment falls toward it
        equilibrium_error = U_prev - nairu_lag - gamma_okun * og_lag
        delta_U = (
            beta_okun * output_gap_fcst[t] * DEMAND_MULTIPLIER
            + alpha_okun * equilibrium_error
            + eps_okun
        )
        unemployment_fcst[t] = U_prev + delta_U
        U_prev = unemployment_fcst[t]

        # Phillips curve (model estimates + import price effect)
        u_gap = (unemployment_fcst[t] - nairu_fcst[t]) / unemployment_fcst[t]
        inflation_fcst[t] = pi_anchor_qtr + gamma_pi_covid * u_gap + import_price_effect + eps_pi

    # --- Package results ---
    cols = [f"sample_{i}" for i in range(n_samples)]

    return BayesianScenarioResults(
        scenario_name=scenario_name,
        cash_rate=cash_rate_T,
        forecast_index=forecast_index,
        obs_index=obs_index,
        nairu_samples=pd.DataFrame(nairu_fcst, index=forecast_index, columns=cols),
        potential_samples=pd.DataFrame(potential_fcst, index=forecast_index, columns=cols),
        output_gap_samples=pd.DataFrame(output_gap_fcst, index=forecast_index, columns=cols),
        unemployment_samples=pd.DataFrame(unemployment_fcst, index=forecast_index, columns=cols),
        inflation_samples=pd.DataFrame(inflation_fcst, index=forecast_index, columns=cols),
        nairu_final=float(np.median(nairu_T)),
        unemployment_final=U_T,
        potential_final=float(np.median(potential_T)),
        log_gdp_final=log_gdp_T,
    )


def run_bayesian_scenarios(
    results: NAIRUResults,
    scenarios: dict[str, float] | None = None,
    n_samples: int = N_SAMPLES,
) -> dict[str, BayesianScenarioResults]:
    """Run multiple policy scenarios with Bayesian forward sampling.

    Args:
        results: NAIRUResults from stage2
        scenarios: Dict of {name: bp_change} from current rate
        n_samples: Number of samples per scenario

    Returns:
        Dict of {scenario_name: BayesianScenarioResults}

    """
    if scenarios is None:
        scenarios = DEFAULT_POLICY_SCENARIOS

    current_rate = results.obs["cash_rate"][-1]

    scenario_results = {}
    for name, bp_change in scenarios.items():
        new_rate = current_rate + bp_change
        scenario_results[name] = bayesian_forward_sample(
            results,
            cash_rate_override=new_rate,
            scenario_name=name,
            n_samples=n_samples,
        )

    return scenario_results


def print_bayesian_comparison(
    scenario_results: dict[str, BayesianScenarioResults],
    current_rate: float,
) -> None:
    """Print comparison table across Bayesian scenarios."""
    print("\n" + "=" * 80)
    print("BAYESIAN SCENARIO COMPARISON (with sampled shocks)")
    print(f"Current cash rate in model: {current_rate:.2f}%")
    print("=" * 80)

    first_result = next(iter(scenario_results.values()))
    periods = first_result.forecast_index

    for period in periods:
        print(f"\n{period}:")
        print("-" * 80)
        print(f"{'Scenario':>8} {'Cash Rate':>10} {'Output Gap':>12} {'U':>8} {'U Gap':>8} {'π (ann)':>10}")
        print(f"{'':>8} {'':>10} {'[90% HDI]':>12} {'[90%]':>8} {'[90%]':>8} {'[90% HDI]':>10}")

        for name in ["+200bp", "+100bp", "+50bp", "+25bp", "hold", "-25bp", "-50bp", "-100bp", "-200bp"]:
            if name not in scenario_results:
                continue
            result = scenario_results[name]
            idx = result.forecast_index.get_loc(period)

            og = result.output_gap_samples.iloc[idx]
            u = result.unemployment_samples.iloc[idx]
            nairu = result.nairu_samples.iloc[idx]
            u_gap = u - nairu
            pi_ann = annualize(result.inflation_samples.iloc[idx])

            print(
                f"{name:>8} {result.cash_rate:>9.2f}% "
                f"{og.median():>6.3f} [{og.quantile(0.05):>5.2f},{og.quantile(0.95):>5.2f}] "
                f"{u.median():>5.2f}% "
                f"{u_gap.median():>+5.2f} "
                f"{pi_ann.median():>5.2f}% [{pi_ann.quantile(0.05):>4.1f},{pi_ann.quantile(0.95):>4.1f}]"
            )

    print("\n" + "=" * 80)


def plot_bayesian_scenario_inflation(
    scenario_results: dict[str, BayesianScenarioResults],
    n_history: int = 4,
    chart_dir: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot Bayesian inflation scenarios with uncertainty bands."""
    if chart_dir is None:
        chart_dir = DEFAULT_CHART_DIR
    chart_dir = Path(chart_dir)
    mg.set_chart_dir(str(chart_dir))

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

    # Plot historical
    from src.data.inflation import get_inflation_qrtly
    cpi = annualize(get_inflation_qrtly().data)
    model_end = scenario_results["hold"].obs_index[-1]
    hist_actual = cpi.loc[cpi.index >= model_end - n_history + 1]
    hist_actual.name = "Actual"
    ax = mg.line_plot(hist_actual, color="black", width=2)

    # Plot each scenario median
    for name in scenario_order:
        if name in scenario_results:
            result = scenario_results[name]
            inflation_annual = annualize(result.inflation_samples)
            median = inflation_annual.median(axis=1)
            median.name = name
            mg.line_plot(median, ax=ax, color=colors[name], width=2 if name == "hold" else 1.5)

    # Add 90% bands for hold scenario
    hold = scenario_results.get("hold")
    if hold is not None and ax is not None:
        inflation_annual = annualize(hold.inflation_samples)
        lower = inflation_annual.quantile(0.05, axis=1)
        upper = inflation_annual.quantile(0.95, axis=1)
        ax.fill_between(
            [p.ordinal for p in hold.forecast_index],
            lower.values,
            upper.values,
            alpha=0.15,
            color="grey",
            label="90% HDI (hold)",
        )

    if ax is not None:
        # Target band (2-3%)
        ax.axhspan(2.0, 3.0, color="green", alpha=0.1, label="Target band (2-3%)")
        ax.axhline(y=2.5, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="Target (2.5%)")

        current_rate = hold.cash_rate if hold else None
        rate_str = f" (from {current_rate:.2f}%)" if current_rate is not None else ""

        mg.finalise_plot(
            ax,
            title=f"Inflation: Policy Rate Scenarios{rate_str}",
            ylabel="Per cent per annum",
            legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
            lheader="Trimmed mean, annualised",
            lfooter="Australia. NAIRU assumed fixed over scenario horizon.",
            rfooter="Bayesian sampling. RBA-calibrated transmission.",
            show=show,
        )


def plot_bayesian_scenario_unemployment(
    scenario_results: dict[str, BayesianScenarioResults],
    obs: dict | None = None,
    obs_index: pd.PeriodIndex | None = None,
    n_history: int = 4,
    chart_dir: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot Bayesian unemployment scenarios with uncertainty bands."""
    if chart_dir is None:
        chart_dir = DEFAULT_CHART_DIR
    chart_dir = Path(chart_dir)
    mg.set_chart_dir(str(chart_dir))

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

    # Plot historical
    ax = None
    if obs is not None and obs_index is not None:
        unemployment_hist = pd.Series(obs["U"], index=obs_index)
        hist_recent = unemployment_hist.iloc[-n_history:]
        hist_recent.name = "Actual"
        ax = mg.line_plot(hist_recent, color="black", width=2)

    # Plot each scenario median
    for name in scenario_order:
        if name in scenario_results:
            result = scenario_results[name]
            median = result.unemployment_samples.median(axis=1)
            median.name = name
            if ax is None:
                ax = mg.line_plot(median, color=colors[name], width=2 if name == "hold" else 1.5)
            else:
                mg.line_plot(median, ax=ax, color=colors[name], width=2 if name == "hold" else 1.5)

    # Add 90% bands for hold scenario
    hold = scenario_results.get("hold")
    if hold is not None and ax is not None:
        lower = hold.unemployment_samples.quantile(0.05, axis=1)
        upper = hold.unemployment_samples.quantile(0.95, axis=1)
        ax.fill_between(
            [p.ordinal for p in hold.forecast_index],
            lower.values,
            upper.values,
            alpha=0.15,
            color="grey",
            label="90% HDI (hold)",
        )

        # NAIRU reference (median across samples at final period)
        nairu_median = hold.nairu_samples.iloc[-1].median()
        ax.axhline(
            y=nairu_median,
            color="grey",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"NAIRU ({nairu_median:.1f}%)",
        )

        current_rate = hold.cash_rate
        rate_str = f" (from {current_rate:.2f}%)"

        mg.finalise_plot(
            ax,
            title=f"Unemployment: Policy Rate Scenarios{rate_str}",
            ylabel="Per cent",
            legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
            lheader="Unemployment rate. Responds to policy more slowly and over a longer time horizon than inflation.",
            lfooter="Australia. NAIRU assumed fixed over scenario horizon.",
            rfooter="Bayesian sampling. RBA-calibrated transmission.",
            show=show,
        )


def plot_bayesian_output_gap(
    scenario_results: dict[str, BayesianScenarioResults],
    obs: dict | None = None,
    obs_index: pd.PeriodIndex | None = None,
    n_history: int = 8,
    chart_dir: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot output gap scenarios."""
    if chart_dir is None:
        chart_dir = DEFAULT_CHART_DIR
    chart_dir = Path(chart_dir)
    mg.set_chart_dir(str(chart_dir))

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

    # Get hold result for historical potential
    hold = scenario_results.get("hold")
    if hold is None:
        return

    # Plot historical output gap (need to compute from obs)
    ax = None
    if obs is not None and obs_index is not None:
        # Historical output gap = log_gdp - potential (from stage2 results)
        # We'll approximate using the final output gap and work backwards
        # For simplicity, just show the scenario projections
        pass

    # Plot each scenario median
    for name in scenario_order:
        if name in scenario_results:
            result = scenario_results[name]
            median = result.output_gap_samples.median(axis=1)
            median.name = name
            if ax is None:
                ax = mg.line_plot(median, color=colors[name], width=2 if name == "hold" else 1.5)
            else:
                mg.line_plot(median, ax=ax, color=colors[name], width=2 if name == "hold" else 1.5)

    # Add 90% bands for hold scenario
    if hold is not None and ax is not None:
        lower = hold.output_gap_samples.quantile(0.05, axis=1)
        upper = hold.output_gap_samples.quantile(0.95, axis=1)
        ax.fill_between(
            [p.ordinal for p in hold.forecast_index],
            lower.values,
            upper.values,
            alpha=0.15,
            color="grey",
            label="90% HDI (hold)",
        )

        # Zero reference line
        ax.axhline(y=0, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="Potential (OG=0)")

        current_rate = hold.cash_rate
        rate_str = f" (from {current_rate:.2f}%)"

        mg.finalise_plot(
            ax,
            title=f"Output Gap: Policy Rate Scenarios{rate_str}",
            ylabel="Per cent of potential",
            legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
            lheader="Output gap = (GDP - Potential) / Potential × 100",
            rheader="Positive output gap is inflationary.",
            lfooter="Australia. Potential growth exogenous over scenario horizon.",
            rfooter="Bayesian sampling. RBA-calibrated transmission.",
            show=show,
        )


def plot_bayesian_output_vs_potential(
    scenario_results: dict[str, BayesianScenarioResults],
    obs: dict | None = None,
    obs_index: pd.PeriodIndex | None = None,
    n_history: int = 8,
    chart_dir: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot GDP vs potential output scenarios."""
    if chart_dir is None:
        chart_dir = DEFAULT_CHART_DIR
    chart_dir = Path(chart_dir)
    mg.set_chart_dir(str(chart_dir))

    # Just plot hold scenario with uncertainty, plus potential
    hold = scenario_results.get("hold")
    if hold is None:
        return

    # Historical GDP
    ax = None
    if obs is not None and obs_index is not None:
        log_gdp_hist = pd.Series(obs["log_gdp"], index=obs_index)
        hist_recent = log_gdp_hist.iloc[-n_history:]
        hist_recent.name = "GDP (actual)"
        ax = mg.line_plot(hist_recent, color="black", width=2)

    # Potential (median from hold scenario - same for all scenarios)
    potential_median = hold.potential_samples.median(axis=1)
    # Prepend final historical potential for continuity
    potential_with_hist = pd.concat([
        pd.Series([hold.potential_final], index=[hold.obs_index[-1]]),
        potential_median
    ])
    potential_with_hist.name = "Potential"
    if ax is None:
        ax = mg.line_plot(potential_with_hist, color="grey", width=2)
    else:
        mg.line_plot(potential_with_hist, ax=ax, color="grey", width=2)
    # Make potential line dashed
    if ax is not None:
        ax.lines[-1].set_linestyle("--")

    # GDP scenarios (just show +200bp, hold, -200bp for clarity)
    key_scenarios = ["+200bp", "hold", "-200bp"]
    colors = {"+200bp": "crimson", "hold": "black", "-200bp": "royalblue"}

    for name in key_scenarios:
        if name in scenario_results:
            result = scenario_results[name]
            gdp_median = result.output_samples().median(axis=1)
            # Prepend final historical GDP for continuity
            gdp_with_hist = pd.concat([
                pd.Series([result.log_gdp_final], index=[result.obs_index[-1]]),
                gdp_median
            ])
            gdp_with_hist.name = f"GDP ({name})"
            mg.line_plot(gdp_with_hist, ax=ax, color=colors[name], width=1.5)

    # Add 90% bands for hold GDP
    if ax is not None:
        gdp_hold = hold.output_samples()
        lower = gdp_hold.quantile(0.05, axis=1)
        upper = gdp_hold.quantile(0.95, axis=1)
        ax.fill_between(
            [p.ordinal for p in hold.forecast_index],
            lower.values,
            upper.values,
            alpha=0.15,
            color="grey",
            label="90% HDI (hold)",
        )

        current_rate = hold.cash_rate
        rate_str = f" (from {current_rate:.2f}%)"

        mg.finalise_plot(
            ax,
            title=f"GDP vs Potential: Policy Rate Scenarios{rate_str}",
            ylabel="Log GDP (×100)",
            legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
            lheader="GDP and potential output (log scale).",
            rheader="GDP above potential is inflationary.",
            lfooter="Australia. Potential growth exogenous over scenario horizon.",
            rfooter="Bayesian sampling. RBA-calibrated transmission.",
            show=show,
        )


def run_stage3_bayesian(
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
    n_samples: int = N_SAMPLES,
    plot_scenarios: bool = True,
    chart_dir: Path | str | None = None,
    verbose: bool = False,
) -> dict[str, BayesianScenarioResults]:
    """Run Stage 3 with Bayesian forward sampling.

    Args:
        output_dir: Directory containing saved results from Stage 1
        prefix: Filename prefix used when saving
        n_samples: Number of posterior samples to use
        plot_scenarios: Generate scenario plots
        chart_dir: Directory to save charts
        verbose: Print detailed output

    Returns:
        Dict of {scenario_name: BayesianScenarioResults}

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

    print(f"Running Bayesian forward sampling with {n_samples} samples...")
    scenario_results = run_bayesian_scenarios(results, n_samples=n_samples)

    # Print comparison
    print_bayesian_comparison(scenario_results, current_rate)

    # Plot
    if plot_scenarios:
        plot_bayesian_scenario_inflation(scenario_results, chart_dir=chart_dir, show=False)
        plot_bayesian_scenario_unemployment(
            scenario_results, obs=obs, obs_index=obs_index, chart_dir=chart_dir, show=False
        )
        plot_bayesian_output_gap(
            scenario_results, obs=obs, obs_index=obs_index, chart_dir=chart_dir, show=False
        )
        plot_bayesian_output_vs_potential(
            scenario_results, obs=obs, obs_index=obs_index, chart_dir=chart_dir, show=False
        )

    return scenario_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run NAIRU + Output Gap Bayesian forward sampling (Stage 3)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory with saved results")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Number of posterior samples")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    run_stage3_bayesian(
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        plot_scenarios=not args.no_plots,
        verbose=args.verbose,
    )
