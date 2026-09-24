"""Inflation expectations Stage 1: Model building and sampling.

This module handles:
- Loading data
- Building the PyMC model
- Sampling the posterior
- Saving results (trace, measures, index) to disk

Run with: uv run python -m src.models.expectations.stage1
"""

import pickle
from pathlib import Path
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
from src.data.inflation import (
    get_headline_qrtly,
    get_trimmed_mean_annual,
    get_trimmed_mean_qrtly,
    get_weighted_median_annual,
    get_weighted_median_qrtly,
)
from src.data.labour_force import get_hours_growth_qrtly
from src.data.productivity import compute_mfp_trend_floored
from src.data.ulc import get_ulc_growth_qrtly
from src.models.common.diagnostics import check_model_diagnostics, save_diagnostics
from src.models.expectations.common import (
    CHART_DIR,
    DEFAULT_CHAINS,
    DEFAULT_DRAWS,
    DEFAULT_TUNE,
    MODEL_NAMES,
    MODEL_TYPES,
    OUTPUT_DIR,
)
from src.utilities.rate_conversion import annualize

# --- Inflation Gap ---

# Inflation is observed as a quarterly annualised rate, and its deviation from
# expectations (the gap) follows an AR(1) across quarters, after Chan, Clark
# and Koop (2018). Beta(2, 2) on the persistence b: centred on 0.5, weight off
# both bounds, so the gap is always stationary and never a random walk.
GAP_PERSISTENCE_ALPHA = 2.0
GAP_PERSISTENCE_BETA = 2.0

MONTHS_PER_QUARTER = 3

# An AR(1) gap needs a previous quarter to condition on.
MIN_GAP_OBSERVATIONS = 2

# --- Model Configurations ---


class ModelConfigDict(TypedDict, total=False):
    """Configuration options for expectation model variants."""

    survey_series: list[str]
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
    "unanchored": {
        "survey_series": ["market_1y", "breakeven", "business", "market_yoy"],
        "use_headline": True,
        "use_nominal": True,
        "nominal_cutoff": "1993Q3",  # 7yr overlap with breakeven for r* identification
        "use_hcoe": True,
        "use_inflation": True,
        "use_survey_bias": True,
        "inflation_sigma_prior": 1.5,
        # Fixed innovation variance, to avoid funnel geometry in the posterior
        "estimate_innovation": False,
        "sigma_early": 0.30,
        "sigma_late": 0.07,
    },
    "short": {
        "survey_series": ["market_1y"],
        "use_headline": True,
        "use_nominal": False,
        "use_hcoe": False,
        "use_inflation": True,
        "use_survey_bias": False,
        "inflation_sigma_prior": 1.5,
        # One survey and a free sigma_obs lets the survey noise collapse and the
        # walk pin to it; sharing sigma with the inflation gap holds it apart.
        "tie_inflation_sigma": True,
        "estimate_innovation": True,
        "sigma_early": 0.12,
        "sigma_late": 0.075,
    },
    "market": {
        "survey_series": ["breakeven"],
        "use_headline": False,
        "use_nominal": True,
        "use_hcoe": False,
        "use_inflation": False,
        "use_survey_bias": False,
        "inflation_sigma_prior": 2.0,
        "estimate_innovation": False,
        "sigma_early": 0.12,
        "sigma_late": 0.075,
        "quarterly": True,  # Run quarterly — too sparse for monthly identification
    },
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
    monthly: bool = False,
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

    # Surveys and the breakeven in one frame, built together so its index is
    # every date any of them has. Assigning the breakeven as a later column
    # would align it to the surveys' dates and drop it before they start.
    surveys = get_expectations_surveys(monthly=monthly)
    measures = pd.DataFrame({
        **{name: surveys[name].data for name in ["market_1y", "business", "market_yoy"] if name in surveys},
        "breakeven": get_breakeven_inflation(monthly=monthly).data,
    })

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


def _lagged_on_index(quarterly: pd.Series, index: pd.PeriodIndex) -> np.ndarray:
    """Place a quarterly series on the model's grid, lagged one period.

    On a monthly grid the quarter sits at its last month. Works on a copy: the
    loaders cache their results, and re-indexing in place would hand every
    later caller a monthly index.
    """
    series = quarterly.copy()
    if index.freqstr.startswith("M"):
        series.index = series.index.asfreq("M", how="end")
    return series.reindex(index).shift(1).to_numpy()


def _add_ar1_gap_obs(
    name: str, pi_exp: pt.TensorVariable, observed: np.ndarray, mask: np.ndarray, *, index: pd.PeriodIndex,
    sigma_prior: float, sigma: pt.TensorVariable | None = None,
) -> None:
    """Observe inflation as expectations plus a gap that is AR(1) across quarters.

    gap_k = y_k - pi_exp_k,  gap_k = b * gap_{k-1} + e_k,  e_k ~ N(0, sigma).
    Conditioning on the previous observed gap gives the exact likelihood, with
    no extra latent states; the first quarter takes the stationary variance.
    A caller passing `sigma` shares an existing noise scale instead of
    estimating one here.
    """
    positions = np.flatnonzero(mask)
    step = MONTHS_PER_QUARTER if index.freqstr.startswith("M") else 1
    if positions.size < MIN_GAP_OBSERVATIONS or not np.all(np.diff(positions) == step):
        raise ValueError(f"{name}: the AR(1) gap needs consecutive quarterly observations with no holes")

    y = observed[positions]
    level = pi_exp[positions]
    b = pm.Beta(f"b_{name}", alpha=GAP_PERSISTENCE_ALPHA, beta=GAP_PERSISTENCE_BETA)
    if sigma is None:
        sigma = pm.HalfNormal(f"sigma_{name}", sigma=sigma_prior)

    pm.Normal(f"obs_{name}_first", mu=level[0], sigma=sigma / pt.sqrt(1 - b**2), observed=y[0])
    pm.Normal(f"obs_{name}", mu=level[1:] + b * (y[:-1] - level[:-1]), sigma=sigma, observed=y[1:])


def _add_inflation_obs(pi_exp: pt.TensorVariable, index: pd.PeriodIndex, config: ModelConfigDict) -> None:
    """Add underlying inflation, quarterly annualised, with an AR(1) gap."""
    if not config.get("use_inflation", False):
        return

    underlying = annualize((get_trimmed_mean_qrtly().data + get_weighted_median_qrtly().data) / 2)
    observed = _lagged_on_index(underlying, index)
    # The survey's sigma_obs exists because survey observations are added first.
    shared = pm.modelcontext(None)["sigma_obs"][0] if config.get("tie_inflation_sigma", False) else None
    _add_ar1_gap_obs("inflation", pi_exp, observed, ~np.isnan(observed), index=index,
                     sigma_prior=config.get("inflation_sigma_prior", 1.5), sigma=shared)


def _add_headline_obs(pi_exp: pt.TensorVariable, index: pd.PeriodIndex, config: ModelConfigDict) -> None:
    """Add pre-1993 headline CPI, quarterly annualised, with its own AR(1) gap."""
    if not config.get("use_headline", False):
        return

    observed = _lagged_on_index(annualize(get_headline_qrtly().data), index)
    pre_targeting = index < _to_period("1993Q1", index.freqstr.startswith("M"))
    _add_ar1_gap_obs("headline", pi_exp, observed, ~np.isnan(observed) & pre_targeting, index=index,
                     sigma_prior=2.0)


def _add_nominal_obs(pi_exp: pt.TensorVariable, index: pd.PeriodIndex, config: ModelConfigDict) -> None:
    """Add nominal bond observation.

    Two modes:
    - Default: pre-breakeven only (cutoff period), used by unanchored
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


# --- Model Building ---


def _build_pymc_model(
    measures: pd.DataFrame, inflation: pd.Series, index: pd.PeriodIndex, config: ModelConfigDict,
) -> pm.Model:
    """Build the PyMC model with the given configuration."""
    monthly = index.freqstr.startswith("M")

    # Prepare data — for monthly, inflation is NaN at non-quarter months;
    # forward-fill the lag so survey months still have an inflation_lag value.
    inflation_lag = inflation.shift(1).ffill().bfill().to_numpy()

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
            # The config value becomes the prior scale. Large sigma_late lets
            # the walk chase inflation surges into a ridge against the survey
            # noise, so the prior keeps weight off that tail.
            sigma_late = pm.HalfNormal("sigma_late", sigma=sigma_late)

        raw_late = pm.StudentT("raw_late", mu=0, sigma=1, nu=4, shape=n_late - 1)
        pi_exp_late = pm.Deterministic(
            "pi_exp_late",
            pt.concatenate([pi_exp_early[-1:], pi_exp_early[-1] + pt.cumsum(raw_late * sigma_late)]),
        )

        pi_exp = pm.Deterministic("pi_exp", pt.concatenate([pi_exp_early, pi_exp_late]))

        # --- Observation Equations ---
        _add_survey_obs(pi_exp, measures, inflation_lag, index, config)
        _add_inflation_obs(pi_exp, index, config)
        _add_headline_obs(pi_exp, index, config)
        _add_nominal_obs(pi_exp, index, config)
        _add_hcoe_obs(pi_exp, index, config)

    return model


def build_model(
    measures: pd.DataFrame,
    inflation: pd.Series,
    index: pd.PeriodIndex,
    model_type: str = "unanchored",
) -> pm.Model:
    """Build signal extraction model for a specific expectation type."""
    match model_type:
        case "unanchored" | "short" | "market":
            config = MODEL_CONFIGS[model_type]
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    return _build_pymc_model(measures, inflation, index, config)


# --- Estimation ---


def run_model(
    model_type: str = "unanchored",
    *,
    start: str = "1983Q1",
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    verbose: bool = True,
    monthly: bool = False,
) -> tuple[az.InferenceData, pd.DataFrame, pd.Series, pd.PeriodIndex]:
    """Run model and return trace + data."""
    model_desc = {
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
    target_accept = {"market": 0.9}.get(model_type, 0.8)

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
    *,
    output_dir: Path | None = None,
) -> None:
    """Save trace and metadata to disk."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly = index.freqstr.startswith("M")

    # Save trace
    trace.to_netcdf(output_dir / f"expectations_{model_type}_trace.nc")

    # Into the chart directory, not beside the trace: this model has no separate
    # analysis step to write it from, so it goes out here.
    save_diagnostics(trace, CHART_DIR, f"expectations_{model_type}", model="expectations")

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

    # The quarterly file downstream models read. Written at either frequency:
    # the loader prefers it, so a run that skipped it would leave an older
    # run's file to be read in its place.
    if monthly:
        hdi_q = hdi.loc[_quarter_end_mask(index)].copy()
        hdi_q.index = hdi_q.index.to_timestamp().to_period("Q")
    else:
        hdi_q = hdi
    hdi_q.to_parquet(output_dir / f"expectations_{model_type}_hdi_quarterly.parquet")
    hdi_q.to_csv(output_dir / f"expectations_{model_type}_hdi_quarterly.csv")


# --- CLI ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expectations Stage 1: Sampling")
    parser.add_argument("--start", default="1983Q1", help="Start period (quarterly format, e.g. 1983Q1)")
    parser.add_argument("--model", choices=MODEL_TYPES, help="Run single model (default: all)")
    parser.add_argument("--monthly", action="store_true", help="Run at monthly frequency (default is quarterly)")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODEL_TYPES

    monthly = args.monthly
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
