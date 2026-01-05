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
from src.data.expectations import get_expectations_surveys
from src.data.inflation import get_headline_annual, get_trimmed_mean_annual, get_weighted_median_annual


# --- Configuration ---

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output" / "expectations"

# Series to use in signal extraction
SURVEY_SERIES = ["business", "market_1y", "market_yoy"]

# Sampler settings
DEFAULT_DRAWS = 10000
DEFAULT_TUNE = 4000
DEFAULT_CHAINS = 4


# --- Results Container ---


@dataclass
class ExpectationsResults:
    """Container for signal extraction results."""

    trace: az.InferenceData
    index: pd.PeriodIndex
    measures: pd.DataFrame
    inflation: pd.Series

    def expectations_posterior(self) -> pd.DataFrame:
        """Get full posterior samples of latent expectations.

        Returns:
            DataFrame with shape (n_periods, n_samples)
        """
        samples = self.trace.posterior["pi_exp"].values
        # Shape: (chains, draws, time) -> (time, chains*draws)
        n_chains, n_draws, n_time = samples.shape
        flat = samples.reshape(n_chains * n_draws, n_time).T
        return pd.DataFrame(flat, index=self.index)

    def expectations_median(self) -> pd.Series:
        """Get posterior median of latent expectations."""
        return self.expectations_posterior().median(axis=1)

    def expectations_hdi(self, prob: float = 0.9) -> pd.DataFrame:
        """Get HDI bounds for expectations."""
        post = self.expectations_posterior()
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

        # Save trace
        self.trace.to_netcdf(output_dir / "expectations_trace.nc")

        # Save point estimates
        hdi = self.expectations_hdi()
        hdi.to_parquet(output_dir / "expectations_hdi.parquet")
        hdi.to_csv(output_dir / "expectations_hdi.csv")


# --- Data Loading ---


def load_data(
    start: str = "1983Q1",
    use_breakeven: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.PeriodIndex]:
    """Load and align expectations measures and inflation.

    Args:
        start: Start period for estimation
        use_breakeven: Whether to include bond breakeven inflation

    Returns:
        Tuple of (measures DataFrame, inflation Series, common index)
    """
    start_period = pd.Period(start)

    # Load survey expectations
    surveys = get_expectations_surveys()
    measures = pd.DataFrame(
        {name: surveys[name].data for name in SURVEY_SERIES if name in surveys}
    )

    # Add breakeven if requested (insert before market_yoy so it stays as reference)
    if use_breakeven:
        breakeven = get_breakeven_inflation()
        cols = list(measures.columns)
        cols.insert(-1, "breakeven")  # Insert before last column
        measures["breakeven"] = breakeven.data
        measures = measures[cols]  # Reorder

    # Interpolate through market_yoy observations distorted by GST anticipation.
    # The 10% GST was introduced July 1, 2000. Market economists' year-on-year
    # expectations in 1999Q3-Q4 (4.2%, 4.9%) reflected the anticipated one-off
    # price level jump, not a change in long-run inflation expectations.
    # Interpolate from 1998Q4 (2.5%) to 2000Q3 (2.3%).
    measures.loc[pd.Period("1999Q3"), "market_yoy"] = 2.45
    measures.loc[pd.Period("1999Q4"), "market_yoy"] = 2.35

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
) -> pm.Model:
    """Build the signal extraction model.

    Args:
        measures: DataFrame of expectations measures (columns = series)
        inflation: Series of actual inflation (for bias control)
        index: PeriodIndex for the estimation sample

    Returns:
        PyMC model
    """
    n_obs = len(index)
    n_measures = len(measures.columns)

    # Lagged inflation for backward-looking bias
    inflation_lag = inflation.shift(1).bfill().values

    # Regime break for innovation variance (inflation targeting adoption)
    regime_break = pd.Period("1993Q1")
    n_early = int((index < regime_break).sum())
    n_late = int((index >= regime_break).sum())

    # Innovation scales: larger early (volatile), original late (anchored)
    sigma_early = 0.12
    sigma_late = 0.075

    with pm.Model() as model:
        # --- Priors ---

        # Series-specific level effect (e.g., business reads low)
        # Last series (market_yoy) is reference, others estimated relative to it
        alpha = pm.Normal("alpha", mu=0, sigma=0.5, shape=n_measures - 1)

        # Backward-looking bias coefficients (one per measure)
        lambda_bias = pm.Normal(
            "lambda_bias",
            mu=0.1,
            sigma=0.15,
            shape=n_measures,
        )

        # Observation noise (one per measure)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0, shape=n_measures)

        # --- State Equation ---
        # Two-regime random walk: volatile pre-targeting, anchored post-targeting
        init_inflation = inflation.iloc[0] if not np.isnan(inflation.iloc[0]) else 5.0

        # Early period (pre-targeting): larger innovations
        pi_exp_early = pm.RandomWalk(
            "pi_exp_early",
            innovation_dist=pm.StudentT.dist(mu=0, sigma=sigma_early, nu=4),
            init_dist=pm.Normal.dist(mu=init_inflation, sigma=2.0),
            steps=n_early - 1,
        )

        # Late period (post-targeting): smaller innovations, continues from early
        pi_exp_late_raw = pm.RandomWalk(
            "pi_exp_late_raw",
            innovation_dist=pm.StudentT.dist(mu=0, sigma=sigma_late, nu=4),
            init_dist=pm.Normal.dist(mu=0, sigma=0.01),
            steps=n_late - 1,
        )
        # Shift to continue from where early ended
        pi_exp_late = pi_exp_late_raw - pi_exp_late_raw[0] + pi_exp_early[-1]

        # Concatenate into single series
        pi_exp = pm.Deterministic(
            "pi_exp",
            pm.math.concatenate([pi_exp_early, pi_exp_late])
        )

        # --- Observation Equations ---
        for i, col in enumerate(measures.columns):
            obs_data = measures[col].values
            mask = ~np.isnan(obs_data)

            if mask.sum() > 0:
                # Expected value: latent + series effect + backward-looking bias
                # Last series (market_yoy) has no alpha (reference category)
                if i < n_measures - 1:
                    mu = pi_exp[mask] + alpha[i] + lambda_bias[i] * inflation_lag[mask]
                else:
                    mu = pi_exp[mask] + lambda_bias[i] * inflation_lag[mask]

                pm.Normal(
                    f"obs_{col}",
                    mu=mu,
                    sigma=sigma_obs[i],
                    observed=obs_data[mask],
                )

        # Actual inflation as observation (lagged - publication shapes expectations)
        # Expectations at t are informed by inflation published from t-1
        sigma_inflation = pm.HalfNormal("sigma_inflation", sigma=1.5)
        inflation_lagged = inflation.shift(1).values  # inflation_{t-1} at position t
        inflation_mask = ~np.isnan(inflation_lagged)
        pm.Normal(
            "obs_inflation",
            mu=pi_exp[inflation_mask],
            sigma=sigma_inflation,
            observed=inflation_lagged[inflation_mask],
        )

        # Headline CPI as additional early observation (pre-1993, pre-targeting era)
        # When expectations were more adaptive, headline CPI helps anchor estimates
        headline = get_headline_annual().data.reindex(index).shift(1)  # Lagged
        pre_targeting = index < pd.Period("1993Q1")
        headline_mask = ~np.isnan(headline.values) & pre_targeting
        if headline_mask.sum() > 0:
            sigma_headline = pm.HalfNormal("sigma_headline", sigma=2.0)
            pm.Normal(
                "obs_headline",
                mu=pi_exp[headline_mask],
                sigma=sigma_headline,
                observed=headline.values[headline_mask],
            )

        # Nominal 10y bond yields as early observation (pre-1986, before indexed bonds)
        # Nominal yield ≈ real rate + inflation expectations + term premium
        # Early 80s had high real rates (Volcker era) - nominal 13-15%, inflation 8-10%
        nominal = get_nominal_10y().data.reindex(index)
        pre_indexed = index < pd.Period("1986Q3")
        nominal_mask = ~np.isnan(nominal.values) & pre_indexed
        if nominal_mask.sum() > 0:
            # Real rate offset (nominal = expectations + real_rate)
            real_rate = pm.Normal("real_rate", mu=5.0, sigma=1.5)
            sigma_nominal = pm.HalfNormal("sigma_nominal", sigma=2.0)
            pm.Normal(
                "obs_nominal",
                mu=pi_exp[nominal_mask] + real_rate,
                sigma=sigma_nominal,
                observed=nominal.values[nominal_mask],
            )

    return model


# --- Estimation ---


def run_model(
    start: str = "1984Q1",
    use_breakeven: bool = True,
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    verbose: bool = True,
) -> ExpectationsResults:
    """Run the signal extraction model.

    Args:
        start: Start period for estimation
        use_breakeven: Whether to include bond breakeven inflation
        draws: Number of posterior draws per chain
        tune: Number of tuning samples
        chains: Number of MCMC chains
        verbose: Print progress

    Returns:
        ExpectationsResults container
    """
    if verbose:
        print("Loading data...")
    measures, inflation, index = load_data(start, use_breakeven)

    if verbose:
        print(f"Sample: {index[0]} to {index[-1]} ({len(index)} quarters)")
        print(f"Measures: {list(measures.columns)}")
        for col in measures.columns:
            n = measures[col].notna().sum()
            print(f"  {col}: {n} obs")

    if verbose:
        print("\nBuilding model...")
    model = build_model(measures, inflation, index)

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

    results = ExpectationsResults(
        trace=trace,
        index=index,
        measures=measures,
        inflation=inflation,
    )

    if verbose:
        print("\nSaving results...")
    results.save()

    return results


# --- CLI ---

if __name__ == "__main__":
    import argparse

    from src.models.common.diagnostics import check_for_zero_coeffs, check_model_diagnostics
    from src.models.common.timeseries import plot_posterior_timeseries

    parser = argparse.ArgumentParser(description="Run expectations signal extraction")
    parser.add_argument("--start", default="1983Q1", help="Start period")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--no-breakeven", action="store_true", help="Exclude bond breakeven")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bar")

    args = parser.parse_args()

    results = run_model(
        start=args.start,
        use_breakeven=not args.no_breakeven,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        verbose=not args.quiet,
    )

    print("\nResults summary:")
    hdi = results.expectations_hdi()
    print(hdi.tail(8))

    # Diagnostics
    print("\n" + "=" * 60)
    print("MCMC Diagnostics")
    print("=" * 60)
    check_model_diagnostics(results.trace)

    print("\n" + "=" * 60)
    print("Coefficient Significance")
    print("=" * 60)
    coef_check = check_for_zero_coeffs(results.trace)
    if not coef_check.empty:
        print(coef_check)
    else:
        print("(No scalar parameters to check)")

    print("\n" + "=" * 60)
    print("Parameter Estimates")
    print("=" * 60)
    print(az.summary(results.trace, var_names=["alpha", "lambda_bias", "sigma_obs"]))

    # Set up chart directory
    chart_dir = Path(__file__).parent.parent.parent.parent / "charts" / "expectations"
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    # Plot
    ax = plot_posterior_timeseries(
        data=results.expectations_posterior(),
        legend_stem="Expectations",
        title="Long-Run Inflation Expectations",
        finalise=False,
    )

    # Add trimmed mean inflation (white background for visibility, then orange)
    trimmed = get_trimmed_mean_annual().data
    trimmed = trimmed[trimmed.index >= results.index[0]]
    trimmed.name = "_"
    mg.line_plot(trimmed, ax=ax, color="white", width=3, zorder=4)
    trimmed.name = "Trimmed Mean Inflation"
    mg.line_plot(trimmed, ax=ax, color="darkorange", width=2, zorder=5)

    mg.finalise_plot(
        ax,
        title="Long-Run Inflation Expectations",
        lfooter="Australia. Informed by Cusbert (2017) signal extraction.",
        rfooter=f"Sample: {results.index[0]} to {results.index[-1]}",
        axhspan={"ymin": 2, "ymax": 3, "color": "red", "alpha": 0.1, "zorder": -1},
        axhline={"y": 2.5, "color": "black", "linestyle": "dashed", "linewidth": 0.75},
        legend={"loc": "best", "fontsize": "x-small"},
    )
