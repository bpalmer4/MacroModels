"""Inflation expectations Stage 1: Model building and sampling.

This module handles:
- Loading data
- Building the PyMC model
- Sampling the posterior
- Saving results (trace, measures, index) to disk

Run with: uv run python -m src.models.expectations.stage1
"""

import pickle
from pathlib import Path  # noqa: TC003 — used in annotations
from typing import TypedDict

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

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


class ModelConfigDict(TypedDict, total=False):
    """Configuration options for expectation model variants."""

    survey_series: list[str]
    use_anchor: bool
    use_headline: bool
    use_nominal: bool
    nominal_cutoff: str
    use_hcoe: bool
    use_inflation: bool
    use_survey_bias: bool
    inflation_sigma_prior: float
    tie_inflation_sigma: bool
    estimate_innovation: bool
    sigma_early: float
    sigma_late: float
    fixed_real_rate: float
    start: str
    nominal_full_sample: bool
    quarterly: bool


MODEL_CONFIGS: dict[str, ModelConfigDict] = {
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
        "quarterly": True,  # Run quarterly — too sparse for monthly identification
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


def _to_period(period_str: str, monthly: bool) -> pd.Period:
    """Convert a quarterly period string to the appropriate frequency.

    Args:
        period_str: Period string in quarterly format (e.g. "1999Q3").
        monthly: If True, convert to the last month of that quarter.

    """
    q = pd.Period(period_str, freq="Q")
    if not monthly:
        return q
    # Last month of the quarter: Q1->Mar, Q2->Jun, Q3->Sep, Q4->Dec
    return q.asfreq("M", how="end")


# GST-distorted periods and replacement values
_GST_OVERRIDES: dict[str, float] = {
    "1999Q3": 2.6,
    "1999Q4": 2.5,
    "2000Q1": 2.5,
    "2000Q2": 2.5,
    "2000Q3": 2.5,
}


def _quarter_end_mask(index: pd.PeriodIndex) -> np.ndarray:
    """Boolean mask for quarter-end months (Mar/Jun/Sep/Dec)."""
    return np.isin(index.month, [3, 6, 9, 12])


def load_data(
    start: str = "1983Q1",
    *,
    monthly: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.PeriodIndex]:
    """Load and align expectations measures and inflation.

    Args:
        start: Start period in quarterly format (e.g. "1983Q1").
        monthly: If True, build a monthly index. Survey/bond data load at monthly
                 frequency; quarterly-native series (inflation) appear only at
                 quarter-end months (Mar/Jun/Sep/Dec) with NaN elsewhere.

    """
    freq = "M" if monthly else "Q"
    start_period = pd.Period(start, freq="Q").asfreq(freq, how="start") if monthly else pd.Period(start)

    # Load survey expectations
    surveys = get_expectations_surveys(monthly=monthly)
    measures = pd.DataFrame(
        {name: surveys[name].data for name in ["market_1y", "business", "market_yoy"] if name in surveys}
    )

    # Add breakeven from bonds
    breakeven = get_breakeven_inflation(monthly=monthly)
    measures["breakeven"] = breakeven.data

    # Interpolate through GST-distorted observations
    for col in ("market_1y", "market_yoy"):
        if col in measures.columns:
            for qtr, val in _GST_OVERRIDES.items():
                measures.loc[_to_period(qtr, monthly), col] = val

    # Load actual inflation (average of trimmed mean and weighted median)
    # These are quarterly-native from ABS — on a monthly grid they appear only
    # at quarter-end months; other months are NaN (handled by observation masks).
    trimmed = get_trimmed_mean_annual().data
    weighted = get_weighted_median_annual().data
    inflation = (trimmed + weighted) / 2

    if monthly:
        # Convert quarterly PeriodIndex to the last month of each quarter
        inflation.index = inflation.index.asfreq("M", how="end")

    # Align to common index
    common_end = min(measures.index.max(), inflation.index.max())
    common_index = pd.period_range(start_period, common_end, freq=freq)

    measures = measures.reindex(common_index)
    inflation = inflation.reindex(common_index)

    return measures, inflation, common_index


# --- Observation Equations ---


def _add_survey_obs(
    pi_exp: pt.TensorVariable,
    measures: pd.DataFrame,
    inflation_lag: np.ndarray,
    index: pd.PeriodIndex,
    config: ModelConfigDict,
) -> None:
    """Add survey/market expectation observations.

    For monthly models, the inflation-lag bias term is only applied at
    quarter-end months (Mar/Jun/Sep/Dec) where a fresh inflation reading
    exists. Non-quarter observations inform pi_exp without the bias correction.
    """
    survey_series = config.get("survey_series", [])
    use_bias = config.get("use_survey_bias", False)
    monthly = index.freqstr.startswith("M")

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
        obs_data = measures[col].to_numpy()
        mask = ~np.isnan(obs_data)
        if mask.sum() == 0:
            continue

        if use_bias and monthly:
            # Split into quarter-end (with bias) and non-quarter (without bias)
            qtr_end = _quarter_end_mask(index)
            mask_qtr = mask & qtr_end
            mask_nonqtr = mask & ~qtr_end

            if mask_qtr.sum() > 0:
                mu_qtr = pi_exp[mask_qtr] + alpha[i] + lambda_bias[i] * inflation_lag[mask_qtr]
                pm.Normal(f"obs_{col}_qtr", mu=mu_qtr, sigma=sigma_obs[i], observed=obs_data[mask_qtr])
            if mask_nonqtr.sum() > 0:
                mu_nonqtr = pi_exp[mask_nonqtr] + alpha[i]
                pm.Normal(f"obs_{col}", mu=mu_nonqtr, sigma=sigma_obs[i], observed=obs_data[mask_nonqtr])
        elif use_bias:
            mu = pi_exp[mask] + alpha[i] + lambda_bias[i] * inflation_lag[mask]
            pm.Normal(f"obs_{col}", mu=mu, sigma=sigma_obs[i], observed=obs_data[mask])
        else:
            pm.Normal(f"obs_{col}", mu=pi_exp[mask], sigma=sigma_obs[i], observed=obs_data[mask])


def _add_inflation_obs(
    pi_exp: pt.TensorVariable, inflation_mask: np.ndarray, inflation_lagged: np.ndarray, config: ModelConfigDict,
) -> None:
    """Add actual inflation observation."""
    if not config.get("use_inflation", False):
        return

    if config.get("tie_inflation_sigma", False):
        # Use sigma_obs[0] from survey - must be called after _add_survey_obs
        sigma = pm.modelcontext(None)["sigma_obs"][0]
    else:
        sigma = pm.HalfNormal("sigma_inflation", sigma=config.get("inflation_sigma_prior", 1.5))

    pm.Normal("obs_inflation", mu=pi_exp[inflation_mask], sigma=sigma, observed=inflation_lagged[inflation_mask])


def _add_headline_obs(pi_exp: pt.TensorVariable, index: pd.PeriodIndex, config: ModelConfigDict) -> None:
    """Add pre-1993 headline CPI observation."""
    if not config.get("use_headline", False):
        return

    headline_raw = get_headline_annual().data
    monthly = index.freqstr.startswith("M")
    if monthly:
        headline_raw.index = headline_raw.index.asfreq("M", how="end")
    headline = headline_raw.reindex(index).shift(1)
    pre_targeting = index < _to_period("1993Q1", monthly)
    mask = ~np.isnan(headline.to_numpy()) & pre_targeting

    if mask.sum() > 0:
        sigma = pm.HalfNormal("sigma_headline", sigma=2.0)
        pm.Normal("obs_headline", mu=pi_exp[mask], sigma=sigma, observed=headline.to_numpy()[mask])


def _add_nominal_obs(pi_exp: pt.TensorVariable, index: pd.PeriodIndex, config: ModelConfigDict) -> None:
    """Add nominal bond observation.

    Two modes:
    - Default: pre-breakeven only (cutoff period), used by target/unanchored
    - Full sample: all available observations with an inflation risk premium,
      used by the market model to improve identification.
    """
    if not config.get("use_nominal", False):
        return

    monthly = index.freqstr.startswith("M")
    nominal = get_nominal_10y(monthly=monthly).data.reindex(index)
    full_sample = config.get("nominal_full_sample", False)

    if full_sample:
        mask = ~np.isnan(nominal.to_numpy())
    else:
        cutoff = config.get("nominal_cutoff", "1988Q3")  # 2yr overlap with breakeven (starts 1986Q3)
        pre_breakeven = index < _to_period(cutoff, monthly)
        mask = ~np.isnan(nominal.to_numpy()) & pre_breakeven

    if mask.sum() > 0:
        fixed_rate = config.get("fixed_real_rate")
        real_rate = fixed_rate if fixed_rate is not None else pm.Normal("real_rate", mu=5.0, sigma=1.5)
        sigma = pm.HalfNormal("sigma_nominal", sigma=2.0)
        pi_masked = pi_exp[mask]
        mu = pi_masked + real_rate + (pi_masked * real_rate / 100)
        if full_sample:
            # Regime-varying inflation risk premium (same regimes as NAIRU
            # Phillips curves): pre-GFC, GFC-to-COVID, post-COVID.
            # Premium was likely higher pre-GFC (inflation uncertainty),
            # compressed during low-vol era, and potentially shifted post-COVID.
            irp_pre_gfc = pm.Normal("irp_pre_gfc", mu=0.5, sigma=0.3)
            irp_gfc = pm.Normal("irp_gfc", mu=0.2, sigma=0.3)
            irp_covid = pm.Normal("irp_covid", mu=0.3, sigma=0.3)

            gfc_start = _to_period("2008Q4", monthly)
            covid_start = _to_period("2021Q1", monthly)
            regime_pre_gfc = (index < gfc_start).astype(float)
            regime_gfc = ((index >= gfc_start) & (index < covid_start)).astype(float)
            regime_covid = (index >= covid_start).astype(float)

            irp = (irp_pre_gfc * regime_pre_gfc[mask]
                   + irp_gfc * regime_gfc[mask]
                   + irp_covid * regime_covid[mask])
            mu = mu + irp
        pm.Normal("obs_nominal", mu=mu, sigma=sigma, observed=nominal.to_numpy()[mask])


def _add_hcoe_obs(pi_exp: pt.TensorVariable, index: pd.PeriodIndex, config: ModelConfigDict) -> None:
    """Add hourly compensation observation."""
    if not config.get("use_hcoe", False):
        return

    monthly = index.freqstr.startswith("M")

    hcoe_raw = get_hourly_coe_growth_annual().data
    ulc_q = get_ulc_growth_qrtly().data
    hcoe_q = get_hourly_coe_growth_qrtly().data
    capital_q = get_capital_growth_qrtly().data
    hours_q = get_hours_growth_qrtly().data
    mfp_trend_q = compute_mfp_trend_floored(ulc_q, hcoe_q, capital_q, hours_q, alpha=0.3).data
    mfp_annual = annualize(mfp_trend_q)

    if monthly:
        hcoe_raw.index = hcoe_raw.index.asfreq("M", how="end")
        mfp_annual.index = mfp_annual.index.asfreq("M", how="end")

    hcoe = hcoe_raw.reindex(index)
    mfp = mfp_annual.reindex(index)
    mask = ~np.isnan(hcoe.to_numpy()) & ~np.isnan(mfp.to_numpy())

    if mask.sum() > 0:
        adjustment = pm.Normal("hcoe_adjustment", mu=0.0, sigma=0.5)
        sigma = pm.HalfNormal("sigma_hcoe", sigma=2.0)
        pm.Normal("obs_hcoe", mu=pi_exp[mask] + mfp.to_numpy()[mask] + adjustment, sigma=sigma,
                  observed=hcoe.to_numpy()[mask])


def _add_target_anchor(pi_exp: pt.TensorVariable, index: pd.PeriodIndex, config: ModelConfigDict) -> None:
    """Add target anchor observation post-1998."""
    if not config.get("use_anchor", False):
        return

    monthly = index.freqstr.startswith("M")
    post_anchored = index >= _to_period("1998Q4", monthly)
    if post_anchored.sum() > 0:
        pm.Normal("obs_target", mu=pi_exp[post_anchored], sigma=ANCHOR_SIGMA,
                  observed=np.full(post_anchored.sum(), ANCHOR_TARGET))


# --- Model Building ---


def _build_pymc_model(
    measures: pd.DataFrame, inflation: pd.Series, index: pd.PeriodIndex, config: ModelConfigDict,
) -> pm.Model:
    """Build the PyMC model with the given configuration."""
    monthly = index.freqstr.startswith("M")

    # Prepare data — for monthly, inflation is NaN at non-quarter months;
    # forward-fill the lag so survey months still have an inflation_lag value.
    inflation_lag = inflation.shift(1).ffill().bfill().to_numpy()
    inflation_lagged = inflation.shift(1).to_numpy()
    inflation_mask = ~np.isnan(inflation_lagged)

    # Regime break for innovation variance
    regime_break = _to_period("1994Q1", monthly)
    n_early = int((index < regime_break).sum())
    n_late = int((index >= regime_break).sum())

    init_inflation = inflation.dropna().iloc[0] if inflation.dropna().shape[0] > 0 else 5.0

    # Innovation variance config — scale for monthly steps (σ_m ≈ σ_q / √3)
    estimate_innovation = config.get("estimate_innovation", False)
    scale = 1 / np.sqrt(3) if monthly else 1.0
    sigma_early = config.get("sigma_early", 0.12) * scale
    sigma_late = config.get("sigma_late", 0.075) * scale

    with pm.Model() as model:
        # --- State Equation (hybrid parameterisation) ---
        #
        # The early regime (pre-1994) uses a CENTERED random walk with FIXED
        # sigma. Observations are sparse here (no surveys until 1986-93) so the
        # data strongly constrain the levels — centered works well, and fixing
        # sigma avoids the funnel geometry entirely.
        #
        # The late regime (1994+) uses a NON-CENTERED random walk with ESTIMATED
        # sigma. Dense monthly survey data identifies sigma_late well. The
        # non-centered form (sample iid z's, then x = cumsum(z * sigma)) breaks
        # the correlation between sigma and the walk levels that causes
        # divergences in the centered form at monthly frequency.
        #

        # Early regime: centered random walk, fixed innovation variance.
        # Centered is fine here — sigma is fixed so there's no funnel.
        pi_exp_early = pm.RandomWalk(
            "pi_exp_early",
            innovation_dist=pm.StudentT.dist(mu=0, sigma=sigma_early, nu=4),
            init_dist=pm.Normal.dist(mu=init_inflation, sigma=2.0),
            steps=n_early - 1,
        )

        # Late regime: non-centered random walk.
        # At monthly frequency, the long walk (~380 steps post-1994) creates
        # strong level correlations that hurt mixing. Non-centering helps
        # regardless of whether sigma is estimated or fixed.
        if estimate_innovation:
            sigma_late = pm.HalfNormal("sigma_late", sigma=0.1 * scale)

        raw_late = pm.StudentT("raw_late", mu=0, sigma=1, nu=4, shape=n_late - 1)
        pi_exp_late = pm.Deterministic(
            "pi_exp_late",
            pt.concatenate([pi_exp_early[-1:], pi_exp_early[-1] + pt.cumsum(raw_late * sigma_late)]),
        )

        pi_exp = pm.Deterministic("pi_exp", pt.concatenate([pi_exp_early, pi_exp_late]))

        # --- Observation Equations ---
        _add_survey_obs(pi_exp, measures, inflation_lag, index, config)
        _add_inflation_obs(pi_exp, inflation_mask, inflation_lagged, config)
        _add_headline_obs(pi_exp, index, config)
        _add_nominal_obs(pi_exp, index, config)
        _add_hcoe_obs(pi_exp, index, config)
        _add_target_anchor(pi_exp, index, config)

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

    return _build_pymc_model(measures, inflation, index, config)


# --- Estimation ---


def run_model(
    model_type: str = "target",
    start: str = "1983Q1",
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    verbose: bool = True,
    *,
    monthly: bool = True,
) -> tuple[az.InferenceData, pd.DataFrame, pd.Series, pd.PeriodIndex]:
    """Run model and return trace + data."""
    model_desc = {
        "target": "Target-anchored (all surveys + anchor)",
        "unanchored": "Expectations (all surveys, no anchor)",
        "short": "Short-run (market_1y, no anchor)",
        "market": "Market (breakeven only, no anchor)",
    }

    # Per-model overrides
    config = MODEL_CONFIGS.get(model_type, {})
    effective_start = config.get("start", start)
    # Per-model quarterly override (e.g. market model too sparse for monthly)
    if config.get("quarterly", False):
        monthly = False
    freq_label = "monthly" if monthly else "quarterly"

    if verbose:
        print(f"\nLoading data for {model_type} model ({freq_label})...")
    measures, inflation, index = load_data(effective_start, monthly=monthly)

    if verbose:
        print(f"Sample: {index[0]} to {index[-1]} ({len(index)} periods, {freq_label})")
        print(f"Model: {model_desc.get(model_type, model_type)}")
        print("Building model...")

    model = build_model(measures, inflation, index, model_type=model_type)

    if verbose:
        print("Sampling posterior...")
    # Target model needs higher target_accept for remaining divergences
    # from the hybrid parameterisation (estimated sigma_late + non-centered walk)
    target_accept = {"target": 0.95, "market": 0.9}.get(model_type, 0.8)

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=chains,
            nuts_sampler="numpyro",
            target_accept=target_accept,
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

    monthly = index.freqstr.startswith("M")

    # Save trace
    trace.to_netcdf(output_dir / f"expectations_{model_type}_trace.nc")

    # Save metadata
    metadata = {
        "measures": measures,
        "inflation": inflation,
        "index": index,
        "model_type": model_type,
        "monthly": monthly,
    }
    meta_path = output_dir / f"expectations_{model_type}_metadata.pkl"
    with meta_path.open("wb") as f:
        pickle.dump(metadata, f)

    # Save HDI for quick access
    samples = trace.posterior["pi_exp"].to_numpy()
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

    # For monthly models, also save quarterly extracts (Mar/Jun/Sep/Dec)
    if monthly:
        qtr_mask = _quarter_end_mask(index)
        hdi_q = hdi.loc[qtr_mask].copy()
        hdi_q.index = hdi_q.index.to_timestamp().to_period("Q")
        hdi_q.to_parquet(output_dir / f"expectations_{model_type}_hdi_quarterly.parquet")
        hdi_q.to_csv(output_dir / f"expectations_{model_type}_hdi_quarterly.csv")


# --- CLI ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expectations Stage 1: Sampling")
    parser.add_argument("--start", default="1983Q1", help="Start period (quarterly format, e.g. 1983Q1)")
    parser.add_argument("--model", choices=MODEL_TYPES, help="Run single model (default: all)")
    parser.add_argument("--quarterly", action="store_true", help="Run at quarterly frequency (default is monthly)")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODEL_TYPES

    monthly = not args.quarterly
    freq_label = "MONTHLY" if monthly else "QUARTERLY"
    print("=" * 60)
    print(f"EXPECTATIONS STAGE 1: SAMPLING ({freq_label})")
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
            monthly=monthly,
        )

        print(f"\nMCMC Diagnostics ({MODEL_NAMES[model_type]}):")
        check_model_diagnostics(trace)

        print(f"\nSaving {model_type} results...")
        save_results(model_type, trace, measures, inflation, index)

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE - Run stage2 for diagnostics and plots")
    print("=" * 60)
