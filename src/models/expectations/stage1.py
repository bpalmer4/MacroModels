"""Inflation expectations Stage 1: Model building and sampling.

This module handles:
- Loading data
- Building the PyMC model
- Sampling the posterior
- Saving results (trace, measures, index) to disk

Run with: uv run python -m src.models.expectations.stage1
"""

import pickle

import arviz as az
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
from src.models.expectations.common import (
    ANCHOR_SIGMA,
    ANCHOR_TARGET,
    DEFAULT_CHAINS,
    DEFAULT_DRAWS,
    DEFAULT_TUNE,
    MODEL_NAMES,
    MODEL_TYPES,
    OUTPUT_DIR,
)
from src.utilities.rate_conversion import annualize

# --- Model Configurations ---

MODEL_CONFIGS = {
    "target": {
        "survey_series": ["market_1y", "breakeven", "business", "market_yoy"],
        "use_anchor": True,
        "use_headline": True,
        "use_nominal": True,
        "nominal_cutoff": "1993Q3",  # 7yr overlap with breakeven for r* identification
        "use_hcoe": True,
        "use_inflation": True,
        "use_survey_bias": True,
        "inflation_sigma_prior": 1.5,
        "tie_inflation_sigma": False,
        "estimate_innovation": True,
        "sigma_early": 0.12,  # Prior center (estimated)
        "sigma_late": 0.075,  # Prior center (estimated)
    },
    "short": {
        "survey_series": ["market_1y"],
        "use_anchor": False,
        "use_headline": True,
        "use_nominal": False,
        "use_hcoe": False,
        "use_inflation": True,
        "use_survey_bias": False,
        "inflation_sigma_prior": 1.5,
        "tie_inflation_sigma": True,
        "estimate_innovation": True,
        "sigma_early": 0.12,
        "sigma_late": 0.075,
    },
    "market": {
        "survey_series": ["breakeven"],
        "use_anchor": False,
        "use_headline": False,
        "use_nominal": True,
        "use_hcoe": False,
        "use_inflation": False,
        "use_survey_bias": False,
        "inflation_sigma_prior": 2.0,
        "tie_inflation_sigma": False,
        "estimate_innovation": False,
        "sigma_early": 0.12,
        "sigma_late": 0.075,
    },
}

# Unanchored is target without the anchor (fixed innovation to avoid funnel geometry)
MODEL_CONFIGS["unanchored"] = {
    **MODEL_CONFIGS["target"],
    "use_anchor": False,
    "estimate_innovation": False,
    "sigma_early": 0.30,
    "sigma_late": 0.07,
}


# --- Data Loading ---


def load_data(
    start: str = "1983Q1",
) -> tuple[pd.DataFrame, pd.Series, pd.PeriodIndex]:
    """Load and align expectations measures and inflation."""
    start_period = pd.Period(start)

    # Load survey expectations
    surveys = get_expectations_surveys()
    measures = pd.DataFrame(
        {name: surveys[name].data for name in ["market_1y", "business", "market_yoy"] if name in surveys}
    )

    # Add breakeven from bonds
    breakeven = get_breakeven_inflation()
    measures["breakeven"] = breakeven.data

    # Interpolate through GST-distorted observations
    if "market_1y" in measures.columns:
        measures.loc[pd.Period("1999Q3"), "market_1y"] = 2.6
        measures.loc[pd.Period("1999Q4"), "market_1y"] = 2.5
        measures.loc[pd.Period("2000Q1"), "market_1y"] = 2.5
        measures.loc[pd.Period("2000Q2"), "market_1y"] = 2.5
        measures.loc[pd.Period("2000Q3"), "market_1y"] = 2.5
    if "market_yoy" in measures.columns:
        measures.loc[pd.Period("1999Q3"), "market_yoy"] = 2.6
        measures.loc[pd.Period("1999Q4"), "market_yoy"] = 2.5
        measures.loc[pd.Period("2000Q1"), "market_yoy"] = 2.5
        measures.loc[pd.Period("2000Q2"), "market_yoy"] = 2.5
        measures.loc[pd.Period("2000Q3"), "market_yoy"] = 2.5

    # Load actual inflation (average of trimmed mean and weighted median)
    trimmed = get_trimmed_mean_annual().data
    weighted = get_weighted_median_annual().data
    inflation = (trimmed + weighted) / 2

    # Align to common index
    common_end = min(measures.index.max(), inflation.index.max())
    common_index = pd.period_range(start_period, common_end, freq="Q")

    measures = measures.reindex(common_index)
    inflation = inflation.reindex(common_index)

    return measures, inflation, common_index


# --- Observation Equations ---


def _add_survey_obs(pi_exp, measures: pd.DataFrame, inflation_lag: np.ndarray, **kwargs) -> None:
    """Add survey/market expectation observations."""
    survey_series = kwargs.get("survey_series", [])
    use_bias = kwargs.get("use_survey_bias", False)

    n_series = len(survey_series)
    if n_series == 0:
        return

    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0, shape=n_series)

    if use_bias:
        alpha = pm.Normal("alpha", mu=0, sigma=0.5, shape=n_series)
        lambda_bias = pm.Normal("lambda_bias", mu=0.1, sigma=0.15, shape=n_series)

    for i, col in enumerate(survey_series):
        if col not in measures.columns:
            continue
        obs_data = measures[col].values
        mask = ~np.isnan(obs_data)
        if mask.sum() > 0:
            mu = pi_exp[mask] + alpha[i] + lambda_bias[i] * inflation_lag[mask] if use_bias else pi_exp[mask]
            pm.Normal(f"obs_{col}", mu=mu, sigma=sigma_obs[i], observed=obs_data[mask])


def _add_inflation_obs(pi_exp, inflation_mask: np.ndarray, inflation_lagged: np.ndarray, **kwargs) -> None:
    """Add actual inflation observation."""
    if not kwargs.get("use_inflation", False):
        return

    if kwargs.get("tie_inflation_sigma", False):
        # Use sigma_obs[0] from survey - must be called after _add_survey_obs
        sigma = pm.modelcontext(None)["sigma_obs"][0]
    else:
        sigma = pm.HalfNormal("sigma_inflation", sigma=kwargs.get("inflation_sigma_prior", 1.5))

    pm.Normal("obs_inflation", mu=pi_exp[inflation_mask], sigma=sigma, observed=inflation_lagged[inflation_mask])


def _add_headline_obs(pi_exp, index: pd.PeriodIndex, **kwargs) -> None:
    """Add pre-1993 headline CPI observation."""
    if not kwargs.get("use_headline", False):
        return

    headline = get_headline_annual().data.reindex(index).shift(1)
    pre_targeting = index < pd.Period("1993Q1")
    mask = ~np.isnan(headline.values) & pre_targeting

    if mask.sum() > 0:
        sigma = pm.HalfNormal("sigma_headline", sigma=2.0)
        pm.Normal("obs_headline", mu=pi_exp[mask], sigma=sigma, observed=headline.values[mask])


def _add_nominal_obs(pi_exp, index: pd.PeriodIndex, **kwargs) -> None:
    """Add nominal bond observation for pre-breakeven period."""
    if not kwargs.get("use_nominal", False):
        return

    nominal = get_nominal_10y().data.reindex(index)
    cutoff = kwargs.get("nominal_cutoff", "1988Q3")  # 2yr overlap with breakeven (starts 1986Q3)
    pre_breakeven = index < pd.Period(cutoff)
    mask = ~np.isnan(nominal.values) & pre_breakeven

    if mask.sum() > 0:
        fixed_rate = kwargs.get("fixed_real_rate")
        real_rate = fixed_rate if fixed_rate is not None else pm.Normal("real_rate", mu=5.0, sigma=1.5)
        sigma = pm.HalfNormal("sigma_nominal", sigma=2.0)
        pi_masked = pi_exp[mask]
        mu = pi_masked + real_rate + (pi_masked * real_rate / 100)
        pm.Normal("obs_nominal", mu=mu, sigma=sigma, observed=nominal.values[mask])


def _add_hcoe_obs(pi_exp, index: pd.PeriodIndex, **kwargs) -> None:
    """Add hourly compensation observation."""
    if not kwargs.get("use_hcoe", False):
        return

    hcoe = get_hourly_coe_growth_annual().data.reindex(index)
    ulc_q = get_ulc_growth_qrtly().data
    hcoe_q = get_hourly_coe_growth_qrtly().data
    capital_q = get_capital_growth_qrtly().data
    hours_q = get_hours_growth_qrtly().data
    mfp_trend_q = compute_mfp_trend_floored(ulc_q, hcoe_q, capital_q, hours_q, alpha=0.3).data
    mfp = annualize(mfp_trend_q).reindex(index)
    mask = ~np.isnan(hcoe.values) & ~np.isnan(mfp.values)

    if mask.sum() > 0:
        adjustment = pm.Normal("hcoe_adjustment", mu=0.0, sigma=0.5)
        sigma = pm.HalfNormal("sigma_hcoe", sigma=2.0)
        pm.Normal("obs_hcoe", mu=pi_exp[mask] + mfp.values[mask] + adjustment, sigma=sigma,
                  observed=hcoe.values[mask])


def _add_target_anchor(pi_exp, index: pd.PeriodIndex, **kwargs) -> None:
    """Add target anchor observation post-1998."""
    if not kwargs.get("use_anchor", False):
        return

    post_anchored = index >= pd.Period("1998Q4")
    if post_anchored.sum() > 0:
        pm.Normal("obs_target", mu=pi_exp[post_anchored], sigma=ANCHOR_SIGMA,
                  observed=np.full(post_anchored.sum(), ANCHOR_TARGET))


# --- Model Building ---


def _build_pymc_model(measures: pd.DataFrame, inflation: pd.Series, index: pd.PeriodIndex, **kwargs) -> pm.Model:
    """Build the PyMC model with the given configuration."""
    # Prepare data
    inflation_lag = inflation.shift(1).bfill().values
    inflation_lagged = inflation.shift(1).values
    inflation_mask = ~np.isnan(inflation_lagged)

    # Regime break for innovation variance
    regime_break = pd.Period("1994Q1")
    n_early = int((index < regime_break).sum())
    n_late = int((index >= regime_break).sum())

    init_inflation = inflation.iloc[0] if not np.isnan(inflation.iloc[0]) else 5.0

    # Innovation variance config
    estimate_innovation = kwargs.get("estimate_innovation", False)
    sigma_early = kwargs.get("sigma_early", 0.12)
    sigma_late = kwargs.get("sigma_late", 0.075)

    with pm.Model() as model:
        # --- State Equation ---
        if estimate_innovation:
            sigma_early = pm.HalfNormal("sigma_early", sigma=0.15)
            sigma_late = pm.HalfNormal("sigma_late", sigma=0.1)

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
        pi_exp = pm.Deterministic("pi_exp", pm.math.concatenate([pi_exp_early, pi_exp_late]))

        # --- Observation Equations ---
        _add_survey_obs(pi_exp, measures, inflation_lag, **kwargs)
        _add_inflation_obs(pi_exp, inflation_mask, inflation_lagged, **kwargs)
        _add_headline_obs(pi_exp, index, **kwargs)
        _add_nominal_obs(pi_exp, index, **kwargs)
        _add_hcoe_obs(pi_exp, index, **kwargs)
        _add_target_anchor(pi_exp, index, **kwargs)

    return model


def build_model(
    measures: pd.DataFrame,
    inflation: pd.Series,
    index: pd.PeriodIndex,
    model_type: str = "target",
) -> pm.Model:
    """Build signal extraction model for a specific expectation type."""
    match model_type:
        case "target" | "unanchored" | "short" | "market":
            config = MODEL_CONFIGS[model_type]
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    return _build_pymc_model(measures, inflation, index, **config)


# --- Estimation ---


def run_model(
    model_type: str = "target",
    start: str = "1983Q1",
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    verbose: bool = True,
) -> tuple[az.InferenceData, pd.DataFrame, pd.Series, pd.PeriodIndex]:
    """Run model and return trace + data."""
    model_desc = {
        "target": "Target-anchored (all surveys + anchor)",
        "unanchored": "Unanchored (all surveys, no anchor)",
        "short": "Short-run (market_1y, no anchor)",
        "market": "Market (breakeven only, no anchor)",
    }

    if verbose:
        print(f"\nLoading data for {model_type} model...")
    measures, inflation, index = load_data(start)

    if verbose:
        print(f"Sample: {index[0]} to {index[-1]} ({len(index)} quarters)")
        print(f"Model: {model_desc.get(model_type, model_type)}")
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

    return trace, measures, inflation, index


def save_results(
    model_type: str,
    trace: az.InferenceData,
    measures: pd.DataFrame,
    inflation: pd.Series,
    index: pd.PeriodIndex,
    output_dir: Path | None = None,
) -> None:
    """Save trace and metadata to disk."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trace
    trace.to_netcdf(output_dir / f"expectations_{model_type}_trace.nc")

    # Save metadata
    metadata = {
        "measures": measures,
        "inflation": inflation,
        "index": index,
        "model_type": model_type,
    }
    with open(output_dir / f"expectations_{model_type}_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # Save HDI for quick access
    samples = trace.posterior["pi_exp"].values
    n_chains, n_draws, n_time = samples.shape
    flat = samples.reshape(n_chains * n_draws, n_time).T
    post = pd.DataFrame(flat, index=index)
    hdi = pd.DataFrame({
        "lower": post.quantile(0.05, axis=1),
        "median": post.median(axis=1),
        "upper": post.quantile(0.95, axis=1),
    }, index=index)
    hdi.to_parquet(output_dir / f"expectations_{model_type}_hdi.parquet")
    hdi.to_csv(output_dir / f"expectations_{model_type}_hdi.csv")


# --- CLI ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expectations Stage 1: Sampling")
    parser.add_argument("--start", default="1983Q1", help="Start period")
    parser.add_argument("--model", choices=MODEL_TYPES, help="Run single model (default: all)")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODEL_TYPES

    print("=" * 60)
    print("EXPECTATIONS STAGE 1: SAMPLING")
    print("=" * 60)

    for model_type in models_to_run:
        print("\n" + "=" * 60)
        print(f"Running {MODEL_NAMES[model_type]} model")
        print("=" * 60)

        trace, measures, inflation, index = run_model(
            model_type=model_type,
            start=args.start,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            verbose=not args.quiet,
        )

        print(f"\nMCMC Diagnostics ({MODEL_NAMES[model_type]}):")
        check_model_diagnostics(trace)

        print(f"\nSaving {model_type} results...")
        save_results(model_type, trace, measures, inflation, index)

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE - Run stage2 for diagnostics and plots")
    print("=" * 60)
