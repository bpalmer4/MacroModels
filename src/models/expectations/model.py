"""Inflation expectations signal extraction model.

Extracts latent long-run inflation expectations from multiple survey and
market-based measures, following the approach in Cusbert (2017).

State equation (random walk with fat tails):
    πᵉ_t = πᵉ_{t-1} + ε_t,    ε_t ~ StudentT(ν=4, σ=0.075)

Observation equations:
    survey_m,t = πᵉ_t + α_m + λ_m × π_{t-1} + ε_m,t

Where:
    - α_m: Series-specific level effect (market_yoy is reference)
    - λ_m: Backward-looking bias coefficient
    - π_{t-1}: Lagged actual inflation

Additional observations for early period anchoring:
    - Lagged trimmed mean/weighted median inflation (full sample)
    - Lagged headline CPI (pre-1993 only)
    - Nominal 10y bond yields (pre-1986 only, with real rate offset)

Reference:
    Cusbert T (2017), "Estimating the NAIRU and the Unemployment Gap",
    RBA Bulletin, June, pp 13-22.
"""

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import mgplot as mg
import numpy as np
import pandas as pd
import pymc as pm
from src.data.bonds import get_breakeven_inflation, get_nominal_10y
from src.data.capital import get_capital_growth_qrtly
from src.data.expectations import get_expectations_surveys
from src.data.hourly_coe import get_hourly_coe_growth_annual, get_hourly_coe_growth_qrtly
from src.data.inflation import get_headline_annual, get_trimmed_mean_annual, get_weighted_median_annual
from src.data.labour_force import get_hours_growth_qrtly
from src.data.productivity import compute_mfp_trend_floored
from src.data.ulc import get_ulc_growth_qrtly
from src.models.common.diagnostics import check_model_diagnostics
from src.utilities.rate_conversion import annualize


# --- Configuration ---

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output" / "expectations"

# Series to use in signal extraction
# Only true forward-looking measures:
# - market_1y: 1-year ahead from professional forecasters
# - breakeven: 10-year market-implied from indexed bonds
# Excluded:
# - business: 3-month own-price expectations (not general CPI)
# - market_yoy: year-on-year nowcast (not forward-looking)
SURVEY_SERIES_LONG = ["market_1y", "breakeven"]  # Long-run expectations
SURVEY_SERIES_SHORT = ["market_1y"]  # Short-run (wage-relevant) expectations

# Sampler settings
DEFAULT_DRAWS = 20000
DEFAULT_TUNE = 4000
DEFAULT_CHAINS = 4

# Target anchoring parameters
ANCHOR_TARGET = 2.5  # Inflation target (%)
ANCHOR_SIGMA = 0.3   # Observation noise for target anchoring

# Model display names
MODEL_NAMES = {
    "long": "TARGET ANCHORED",
    "short": "SHORT RUN (1 Year)",
    "market": "LONG RUN (10-Year Bond)",
}


# --- Results Container ---


@dataclass
class ExpectationsResults:
    """Container for signal extraction results."""

    trace: az.InferenceData
    index: pd.PeriodIndex
    measures: pd.DataFrame
    inflation: pd.Series
    model_type: str = "long"

    def _get_posterior(self, var_name: str) -> pd.DataFrame:
        """Get full posterior samples of a latent variable.

        Returns:
            DataFrame with shape (n_periods, n_samples)
        """
        samples = self.trace.posterior[var_name].values
        # Shape: (chains, draws, time) -> (time, chains*draws)
        n_chains, n_draws, n_time = samples.shape
        flat = samples.reshape(n_chains * n_draws, n_time).T
        return pd.DataFrame(flat, index=self.index)

    def expectations_posterior(self, kind: str = "long") -> pd.DataFrame:
        """Get full posterior samples of latent expectations.

        Args:
            kind: Unused - kept for API compatibility. Single model always uses pi_exp.

        Returns:
            DataFrame with shape (n_periods, n_samples)
        """
        return self._get_posterior("pi_exp")

    def expectations_median(self, kind: str = "long") -> pd.Series:
        """Get posterior median of latent expectations."""
        return self.expectations_posterior(kind).median(axis=1)

    def expectations_hdi(self, prob: float = 0.9, kind: str = "long") -> pd.DataFrame:
        """Get HDI bounds for expectations."""
        post = self.expectations_posterior(kind)
        alpha = (1 - prob) / 2
        return pd.DataFrame(
            {
                "lower": post.quantile(alpha, axis=1),
                "median": post.median(axis=1),
                "upper": post.quantile(1 - alpha, axis=1),
            },
            index=self.index,
        )

    def save(self, output_dir: Path | None = None) -> None:
        """Save results to disk."""
        output_dir = output_dir or OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save trace with model type suffix
        self.trace.to_netcdf(output_dir / f"expectations_{self.model_type}_trace.nc")

        # Save HDI estimates
        hdi = self.expectations_hdi()
        hdi.to_parquet(output_dir / f"expectations_{self.model_type}_hdi.parquet")
        hdi.to_csv(output_dir / f"expectations_{self.model_type}_hdi.csv")


# --- Data Loading ---


def load_data(
    start: str = "1983Q1",
) -> tuple[pd.DataFrame, pd.Series, pd.PeriodIndex]:
    """Load and align expectations measures and inflation.

    Args:
        start: Start period for estimation

    Returns:
        Tuple of (measures DataFrame, inflation Series, common index)
    """
    start_period = pd.Period(start)

    # Load survey expectations - always load market_1y
    surveys = get_expectations_surveys()
    measures = pd.DataFrame(
        {name: surveys[name].data for name in SURVEY_SERIES_SHORT if name in surveys}
    )

    # Add breakeven from bonds (for long-run)
    breakeven = get_breakeven_inflation()
    measures["breakeven"] = breakeven.data

    # Interpolate through market_1y observations distorted by GST anticipation.
    # The 10% GST was introduced July 1, 2000. Market economists' 1-year ahead
    # expectations in 1999Q3-2000Q3 reflected the anticipated one-off price level
    # jump, not a change in long-run inflation expectations.
    if "market_1y" in measures.columns:
        measures.loc[pd.Period("1999Q3"), "market_1y"] = 2.6
        measures.loc[pd.Period("1999Q4"), "market_1y"] = 2.5
        measures.loc[pd.Period("2000Q1"), "market_1y"] = 2.5
        measures.loc[pd.Period("2000Q2"), "market_1y"] = 2.5
        measures.loc[pd.Period("2000Q3"), "market_1y"] = 2.5

    # Load actual inflation for backward-looking bias control
    # Average of trimmed mean and weighted median for robustness
    trimmed = get_trimmed_mean_annual().data
    weighted = get_weighted_median_annual().data
    inflation = (trimmed + weighted) / 2

    # Align to common index - start at requested period even if data is sparse
    # Model will extrapolate through missing observations with wider uncertainty
    common_end = min(measures.index.max(), inflation.index.max())
    common_index = pd.period_range(start_period, common_end, freq="Q")

    measures = measures.reindex(common_index)
    inflation = inflation.reindex(common_index)

    return measures, inflation, common_index


# --- Model Building ---


def build_model(
    measures: pd.DataFrame,
    inflation: pd.Series,
    index: pd.PeriodIndex,
    model_type: str = "long",
) -> pm.Model:
    """Build signal extraction model for a specific expectation type.

    Args:
        measures: DataFrame of expectations measures (columns = series)
        inflation: Series of actual inflation (for bias control)
        index: PeriodIndex for the estimation sample
        model_type: "long" (anchored), "short" (wage-relevant), or "market" (breakeven-based)

    Returns:
        PyMC model
    """
    # Lagged inflation for backward-looking bias
    inflation_lag = inflation.shift(1).bfill().values
    inflation_lagged = inflation.shift(1).values
    inflation_mask = ~np.isnan(inflation_lagged)

    # Regime break for innovation variance (inflation targeting bedded down)
    regime_break = pd.Period("1994Q1")
    n_early = int((index < regime_break).sum())
    n_late = int((index >= regime_break).sum())

    # Innovation scales: larger early (volatile), smaller late (anchored)
    sigma_early = 0.12
    sigma_late = 0.075

    # Configure observations based on model type
    if model_type == "long":
        survey_series = ["market_1y", "breakeven"]
        use_anchor = True
        use_headline = True
        use_nominal = True
        use_hcoe = True
        use_inflation = True
        inflation_sigma_prior = 1.5
        tie_inflation_sigma = False
    elif model_type == "short":
        survey_series = ["market_1y"]
        use_anchor = False
        use_headline = True
        use_nominal = True
        use_hcoe = True
        use_inflation = True
        inflation_sigma_prior = 1.0  # Will be tied to survey sigma
        tie_inflation_sigma = True
    else:  # market
        survey_series = ["breakeven"]
        use_anchor = False
        use_headline = False
        use_nominal = True  # Backcast using nominal bonds before breakeven exists
        use_hcoe = False
        use_inflation = False  # Current inflation irrelevant to 10-year expectations
        inflation_sigma_prior = 2.0
        tie_inflation_sigma = False

    n_series = len(survey_series)
    init_inflation = inflation.iloc[0] if not np.isnan(inflation.iloc[0]) else 5.0

    with pm.Model() as model:
        # --- Priors ---
        alpha = pm.Normal("alpha", mu=0, sigma=0.5, shape=n_series)
        lambda_bias = pm.Normal("lambda_bias", mu=0.1, sigma=0.15, shape=n_series)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0, shape=n_series)

        # --- State Equation (two-regime random walk) ---
        pi_exp_early = pm.RandomWalk(
            "pi_exp_early",
            innovation_dist=pm.StudentT.dist(mu=0, sigma=sigma_early, nu=4),
            init_dist=pm.Normal.dist(mu=init_inflation, sigma=2.0),
            steps=n_early - 1,
        )
        pi_exp_late_raw = pm.RandomWalk(
            "pi_exp_late_raw",
            innovation_dist=pm.StudentT.dist(mu=0, sigma=sigma_late, nu=4),
            init_dist=pm.Normal.dist(mu=0, sigma=0.01),
            steps=n_late - 1,
        )
        pi_exp_late = pi_exp_late_raw - pi_exp_late_raw[0] + pi_exp_early[-1]
        pi_exp = pm.Deterministic(
            "pi_exp",
            pm.math.concatenate([pi_exp_early, pi_exp_late])
        )

        # --- Survey/Market Observations ---
        for i, col in enumerate(survey_series):
            if col not in measures.columns:
                continue
            obs_data = measures[col].values
            mask = ~np.isnan(obs_data)
            if mask.sum() > 0:
                mu = pi_exp[mask] + alpha[i] + lambda_bias[i] * inflation_lag[mask]
                pm.Normal(f"obs_{col}", mu=mu, sigma=sigma_obs[i], observed=obs_data[mask])

        # --- Inflation Observation ---
        if use_inflation:
            if tie_inflation_sigma:
                # Short-run: tie inflation sigma to survey sigma for equal weight
                pm.Normal("obs_inflation", mu=pi_exp[inflation_mask], sigma=sigma_obs[0],
                          observed=inflation_lagged[inflation_mask])
            else:
                sigma_inflation = pm.HalfNormal("sigma_inflation", sigma=inflation_sigma_prior)
                pm.Normal("obs_inflation", mu=pi_exp[inflation_mask], sigma=sigma_inflation,
                          observed=inflation_lagged[inflation_mask])

        # --- Headline CPI (pre-1993) ---
        if use_headline:
            headline = get_headline_annual().data.reindex(index).shift(1)
            pre_targeting = index < pd.Period("1993Q1")
            headline_mask = ~np.isnan(headline.values) & pre_targeting
            if headline_mask.sum() > 0:
                sigma_headline = pm.HalfNormal("sigma_headline", sigma=2.0)
                pm.Normal("obs_headline", mu=pi_exp[headline_mask], sigma=sigma_headline,
                          observed=headline.values[headline_mask])

        # --- Nominal 10y Bonds (pre-breakeven, multiplicative Fisher) ---
        if use_nominal:
            nominal = get_nominal_10y().data.reindex(index)
            pre_breakeven = index < pd.Period("1993Q3")
            nominal_mask = ~np.isnan(nominal.values) & pre_breakeven
            if nominal_mask.sum() > 0:
                real_rate = pm.Normal("real_rate", mu=5.0, sigma=1.5)
                sigma_nominal = pm.HalfNormal("sigma_nominal", sigma=2.0)
                # Multiplicative Fisher: nominal = pi_exp + real + (pi_exp × real / 100)
                pi_masked = pi_exp[nominal_mask]
                mu_nominal = pi_masked + real_rate + (pi_masked * real_rate / 100)
                pm.Normal("obs_nominal", mu=mu_nominal, sigma=sigma_nominal,
                          observed=nominal.values[nominal_mask])

        # --- HCOE Growth ---
        if use_hcoe:
            hcoe = get_hourly_coe_growth_annual().data.reindex(index)
            ulc_q = get_ulc_growth_qrtly().data
            hcoe_q = get_hourly_coe_growth_qrtly().data
            capital_q = get_capital_growth_qrtly().data
            hours_q = get_hours_growth_qrtly().data
            mfp_trend_q = compute_mfp_trend_floored(ulc_q, hcoe_q, capital_q, hours_q, alpha=0.3).data
            mfp = annualize(mfp_trend_q).reindex(index)
            hcoe_mask = ~np.isnan(hcoe.values) & ~np.isnan(mfp.values)
            if hcoe_mask.sum() > 0:
                hcoe_adjustment = pm.Normal("hcoe_adjustment", mu=0.0, sigma=0.5)
                sigma_hcoe = pm.HalfNormal("sigma_hcoe", sigma=2.0)
                pm.Normal("obs_hcoe", mu=pi_exp[hcoe_mask] + mfp.values[hcoe_mask] + hcoe_adjustment,
                          sigma=sigma_hcoe, observed=hcoe.values[hcoe_mask])

        # --- Target Anchoring (long-run only) ---
        if use_anchor:
            post_anchored = index >= pd.Period("1998Q4")
            if post_anchored.sum() > 0:
                pm.Normal("obs_target", mu=pi_exp[post_anchored], sigma=ANCHOR_SIGMA,
                          observed=np.full(post_anchored.sum(), ANCHOR_TARGET))

    return model


# --- Estimation ---


def run_model(
    model_type: str = "long",
    start: str = "1984Q1",
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    verbose: bool = True,
) -> ExpectationsResults:
    """Run signal extraction model for a specific expectation type.

    Args:
        model_type: "long" (anchored), "short" (wage-relevant), or "market" (breakeven-based)
        start: Start period for estimation
        draws: Number of posterior draws per chain
        tune: Number of tuning samples
        chains: Number of MCMC chains
        verbose: Print progress

    Returns:
        ExpectationsResults container with posteriors
    """
    model_desc = {
        "long": "Long-run (market_1y + breakeven + anchor)",
        "short": "Short-run (market_1y, no anchor)",
        "market": "Market (breakeven only, no anchor)",
    }

    if verbose:
        print(f"\nLoading data for {model_type} model...")
    measures, inflation, index = load_data(start)

    if verbose:
        print(f"Sample: {index[0]} to {index[-1]} ({len(index)} quarters)")
        print(f"Model: {model_desc.get(model_type, model_type)}")

    if verbose:
        print("Building model...")
    model = build_model(measures, inflation, index, model_type=model_type)

    if verbose:
        print("Sampling posterior...")
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=chains,
            nuts_sampler="numpyro",
            random_seed=42,
            progressbar=verbose,
        )

    return ExpectationsResults(
        trace=trace,
        index=index,
        measures=measures,
        inflation=inflation,
        model_type=model_type,
    )


def run_all_models(
    start: str = "1984Q1",
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    verbose: bool = True,
) -> dict[str, ExpectationsResults]:
    """Run all three expectation models sequentially.

    Estimates three separate latent expectations series:
    - Long-run: market_1y + breakeven + anchor (assumed anchored)
    - Short-run: market_1y only, no anchor (wage-relevant)
    - Market: breakeven only, no anchor (what market believes)

    Args:
        start: Start period for estimation
        draws: Number of posterior draws per chain
        tune: Number of tuning samples
        chains: Number of MCMC chains
        verbose: Print progress

    Returns:
        Dict mapping model_type to ExpectationsResults
    """
    results = {}
    for model_type in ["long", "short", "market"]:
        if verbose:
            print("\n" + "=" * 60)
            print(f"Running {MODEL_NAMES[model_type]} model")
            print("=" * 60)
        results[model_type] = run_model(
            model_type=model_type,
            start=start,
            draws=draws,
            tune=tune,
            chains=chains,
            verbose=verbose,
        )
        if verbose:
            print(f"\nMCMC Diagnostics ({MODEL_NAMES[model_type]}):")
            check_model_diagnostics(results[model_type].trace)
            print(f"\nParameter Estimates ({MODEL_NAMES[model_type]}):")
            print(az.summary(results[model_type].trace, var_names=["alpha", "lambda_bias", "sigma_obs"]))
    return results


# --- CLI ---

if __name__ == "__main__":
    import argparse

    from src.models.common.timeseries import plot_posterior_timeseries

    parser = argparse.ArgumentParser(description="Run expectations signal extraction")
    parser.add_argument("--start", default="1983Q1", help="Start period")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bar")

    args = parser.parse_args()

    # Run all three models sequentially
    print("=" * 60)
    print("INFLATION EXPECTATIONS MODELS")
    print("  Target Anchored: market_1y + breakeven + anchor")
    print("  Short Run (1 Year): market_1y only, no anchor")
    print("  Long Run (10-Year Bond): breakeven only, no anchor")
    print("=" * 60)

    all_results = run_all_models(
        start=args.start,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        verbose=not args.quiet,
    )

    # Save all results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    for model_type, results in all_results.items():
        print(f"Saving {model_type} results...")
        results.save()

    # Print HDI summaries
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for model_type, results in all_results.items():
        print(f"\n{MODEL_NAMES[model_type]}:")
        print(results.expectations_hdi().tail(4))

    # Set up chart directory
    chart_dir = Path(__file__).parent.parent.parent.parent / "charts" / "expectations"
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    # Use long-run results as reference for index/measures
    results_long = all_results["long"]
    results_short = all_results["short"]
    results_market = all_results["market"]

    # Prepare overlay data
    trimmed = get_trimmed_mean_annual().data
    trimmed = trimmed[trimmed.index >= results_long.index[0]]

    market_1y = results_long.measures["market_1y"].dropna()
    market_1y.name = "Market Economists (1yr)"
    breakeven_series = results_long.measures["breakeven"].dropna()
    breakeven_series.name = "Breakeven (10yr)"

    plot_kwargs = {
        "axhspan": {"ymin": 2, "ymax": 3, "color": "red", "alpha": 0.1, "zorder": -1},
        "axhline": {"y": 2.5, "color": "black", "linestyle": "dashed", "linewidth": 0.75},
        "legend": {"loc": "best", "fontsize": "x-small"},
    }

    # Get posteriors from each model
    posterior_long = results_long.expectations_posterior()
    posterior_short = results_short.expectations_posterior()
    posterior_market = results_market.expectations_posterior()

    # Chart 1: Target-anchored expectations
    ax = plot_posterior_timeseries(data=posterior_long, legend_stem="Target Anchored", finalise=False)
    trimmed.name = "Trimmed Mean Inflation"
    mg.line_plot(trimmed, ax=ax, color="darkorange", width=2, annotate=False, zorder=5)
    mg.finalise_plot(
        ax,
        title="Target Anchored Inflation Expectations",
        lfooter="Australia. Model: market_1y + breakeven + anchor.",
        rfooter=f"Sample: {results_long.index[0]} to {results_long.index[-1]}",
        **plot_kwargs,
    )

    # Chart 2: Short-run (1-year) expectations
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

    # Chart 3: Long-run (10-year bond informed) expectations
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

    # Chart 4: All three expectations on same chart with distributions
    ax = plot_posterior_timeseries(data=posterior_long, legend_stem="Target Anchored",
                                   color="steelblue", finalise=False)
    ax = plot_posterior_timeseries(data=posterior_short, legend_stem="Short Run (1yr)",
                                   color="darkorange", ax=ax, finalise=False)
    ax = plot_posterior_timeseries(data=posterior_market, legend_stem="Long Run (10yr)",
                                   color="darkgreen", ax=ax, finalise=False)
    mg.finalise_plot(
        ax,
        title="Inflation Expectations: Three Measures",
        lfooter="Australia. Blue=target anchored, orange=short run (1yr), green=long run (10yr bond).",
        rfooter=f"Sample: {results_long.index[0]} to {results_long.index[-1]}",
        **plot_kwargs,
    )

    # Chart 5: Medians only comparison
    median_long = posterior_long.median(axis=1)
    median_long.name = "Target Anchored"
    median_short = posterior_short.median(axis=1)
    median_short.name = "Short Run (1yr)"
    median_market = posterior_market.median(axis=1)
    median_market.name = "Long Run (10yr)"

    ax = mg.line_plot(median_long, color="steelblue", width=2, annotate=False)
    mg.line_plot(median_short, ax=ax, color="darkorange", width=2, annotate=False)
    mg.line_plot(median_market, ax=ax, color="darkgreen", width=2, annotate=False)
    mg.finalise_plot(
        ax,
        title="Inflation Expectations: Median Comparison",
        lfooter="Australia. Blue=target anchored, orange=short run (1yr), green=long run (10yr bond).",
        rfooter=f"Sample: {results_long.index[0]} to {results_long.index[-1]}",
        **plot_kwargs,
    )
