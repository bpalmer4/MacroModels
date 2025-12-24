"""NAIRU + Output Gap Stage 1: Data preparation, model building, and sampling.

This module handles:
- Loading and preparing observation data from ABS/RBA
- Building the PyMC model
- Sampling the posterior
- Saving results (observations, trace) to disk
"""

import pickle
from pathlib import Path
from typing import Any, cast

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from statsmodels.tsa.filters.hp_filter import hpfilter

from src.data import (
    get_capital_growth_qrtly,
    get_cash_rate_qrtly,
    get_dfd_deflator_growth_annual,
    get_gscpi_qrtly,
    get_hourly_coe_growth_qrtly,
    get_import_price_growth_annual,
    get_inflation_annual,
    get_inflation_qrtly,
    get_labour_force_growth_qrtly,
    get_log_gdp,
    get_oil_change_annual,
    get_participation_rate_change_qrtly,
    get_twi_change_annual,
    get_twi_change_qrtly,
    get_ulc_growth_qrtly,
    get_unemployment_rate_qrtly,
    hma,
)
from src.data.abs_loader import load_series
from src.data.gdp import get_gdp
from src.data.gov_spending import get_fiscal_impulse_qrtly
from src.data.rba_loader import get_inflation_anchor
from src.data.series_specs import HOURS_WORKED_INDEX
from src.equations import (
    REGIME_COVID_START,
    REGIME_GFC_START,
    exchange_rate_equation,
    hourly_coe_equation,
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

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "model_outputs"


# --- Data Preparation ---


def _derive_mfp_from_wages(
    ulc_growth: pd.Series,
    hcoe_growth: pd.Series,
    capital_growth: pd.Series,
    hours_growth: pd.Series,
    alpha: float = ALPHA,
) -> pd.Series:
    """Derive MFP growth from wage data using Solow residual identity.

    Labour productivity growth = Δhcoe - Δulc (from wage equation identity)
    MFP growth = labour_productivity - α × capital_deepening

    Where capital_deepening = g_K - g_L (capital growth minus hours growth)

    This replaces external MFP data (ABS 5204) with an internally consistent
    estimate derived from the model's wage equations.

    Args:
        ulc_growth: Unit labour cost growth (quarterly)
        hcoe_growth: Hourly COE growth (quarterly)
        capital_growth: Capital stock growth (quarterly, smoothed)
        hours_growth: Hours worked growth (quarterly)
        alpha: Capital share (default 0.3)

    Returns:
        Derived MFP growth (quarterly)

    """
    # Labour productivity growth from wage data
    # ULC = COE/GDP, HCOE = COE/Hours
    # So: Δulc - Δhcoe = Δhours - Δgdp = -labour_productivity_growth
    # Therefore: labour_productivity_growth = Δhcoe - Δulc
    labour_productivity = hcoe_growth - ulc_growth

    # Capital deepening = capital growth - hours growth
    capital_deepening = capital_growth - hours_growth

    # MFP as Solow residual: g_MFP = g_LP - α × g_KL
    mfp_growth = labour_productivity - alpha * capital_deepening

    # HP filter to extract smooth trend (λ=1600 for quarterly data)
    # Raw MFP is noisy - use trend for potential output calculation
    mfp_clean = mfp_growth.dropna()
    _, mfp_trend = hpfilter(mfp_clean, lamb=1600)
    return pd.Series(mfp_trend, index=mfp_clean.index).reindex(mfp_growth.index)


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


def _prepare_hours_growth() -> pd.Series:
    """Prepare hours growth with COVID smoothing."""
    hours_index = load_series(HOURS_WORKED_INDEX).data
    hours_growth_raw = np.log(hours_index).diff(1) * 100
    hours_growth_smoothed = hma(hours_growth_raw.dropna(), HMA_TERM)
    hours_growth_smoothed = hours_growth_smoothed.reindex(hours_growth_raw.index)

    # Replace COVID period with smoothed values
    covid_period = pd.period_range("2020Q1", "2023Q2", freq="Q")
    return hours_growth_raw.where(
        ~hours_growth_raw.index.isin(covid_period),
        other=hours_growth_smoothed,
    )


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
    U_1 = U.shift(1)
    ΔU_1_over_U = ΔU.shift(1) / U  # Speed limit term for wage equations

    # Participation rate
    Δpr = get_participation_rate_change_qrtly().data

    # GDP
    log_gdp = get_log_gdp().data

    # Production inputs (with model-specific smoothing)
    capital_growth = _prepare_capital_growth()
    lf_growth = _prepare_labour_force_growth()

    # Hours growth (for capital deepening in MFP derivation)
    hours_growth = _prepare_hours_growth()

    # Inflation
    π = get_inflation_qrtly().data
    π4 = get_inflation_annual().data
    π_anchor = get_inflation_anchor().data

    # Interest rates
    cash_rate = get_cash_rate_qrtly().data

    # Unit labour costs
    Δulc = get_ulc_growth_qrtly().data
    Δulc_1 = Δulc.shift(1)

    # Hourly compensation of employees (COE/hours - cleaner wage signal)
    Δhcoe = get_hourly_coe_growth_qrtly().data
    Δhcoe_1 = Δhcoe.shift(1)

    # Derive MFP from wage data (replaces external ABS 5204 MFP)
    # MFP = (Δhcoe - Δulc) - α × (g_K - g_L)
    mfp_growth = _derive_mfp_from_wages(
        ulc_growth=Δulc,
        hcoe_growth=Δhcoe,
        capital_growth=capital_growth,
        hours_growth=hours_growth,
    )

    # Domestic final demand deflator (demand channel for wages)
    Δ4dfd = get_dfd_deflator_growth_annual().data

    # Import prices
    Δ4ρm = get_import_price_growth_annual().data
    Δ4ρm_1 = Δ4ρm.shift(1)

    # GSCPI (COVID supply chain pressure, masked and lagged)
    ξ_2 = _prepare_gscpi()

    # TWI (Trade-Weighted Index)
    Δtwi = get_twi_change_qrtly().data
    Δtwi_1 = Δtwi.shift(1)
    Δ4twi_1 = get_twi_change_annual().data.shift(1)

    # Oil prices (AUD)
    Δ4oil_1 = get_oil_change_annual().data.shift(1)

    # Fiscal impulse (lagged)
    fiscal_impulse_1 = get_fiscal_impulse_qrtly().data.shift(1)

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
        "ΔU_1_over_U": ΔU_1_over_U,
        # Participation
        "Δpr": Δpr,
        # GDP
        "log_gdp": log_gdp,
        # Production inputs
        "capital_growth": capital_growth,
        "lf_growth": lf_growth,
        "hours_growth": hours_growth,
        "mfp_growth": mfp_growth,
        # Rates
        "cash_rate": cash_rate,
        "det_r_star": r_star,
        # Unit labor costs
        "Δulc": Δulc,
        "Δulc_1": Δulc_1,
        # Hourly compensation
        "Δhcoe": Δhcoe,
        "Δhcoe_1": Δhcoe_1,
        # Demand deflator
        "Δ4dfd": Δ4dfd,
        # Import prices and supply shocks
        "Δ4ρm": Δ4ρm,
        "Δ4ρm_1": Δ4ρm_1,
        "ξ_2": ξ_2,
        # Exchange rates
        "Δtwi": Δtwi,
        "Δtwi_1": Δtwi_1,
        "Δ4twi_1": Δ4twi_1,
        "r_gap_1": r_gap_1,
        # Oil prices
        "Δ4oil_1": Δ4oil_1,
        # Fiscal impulse
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

    # Add Phillips curve regime indicators (3 regimes)
    # Regime 1: Pre-GFC (before 2008Q4) - moderate slope
    # Regime 2: Post-GFC (2008Q4 - 2020Q4) - flat
    # Regime 3: Post-COVID (2021Q1+) - steep
    idx = observed.index
    observed["regime_pre_gfc"] = (idx < REGIME_GFC_START).astype(float)
    observed["regime_gfc"] = ((idx >= REGIME_GFC_START) & (idx < REGIME_COVID_START)).astype(float)
    observed["regime_covid"] = (idx >= REGIME_COVID_START).astype(float)

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
    hourly_coe_const: dict[str, Any] | None = None,
    include_exchange_rate: bool = True,
    include_import_price: bool = True,
    include_participation: bool = True,
    include_hourly_coe: bool = True,
) -> pm.Model:
    """Build the joint NAIRU + Output Gap model.

    Args:
        obs: Observation dictionary from build_observations()
        nairu_const: Fixed values for NAIRU equation
        potential_const: Fixed values for potential output equation
        exchange_rate_const: Fixed values for exchange rate equation
        import_price_const: Fixed values for import price equation
        participation_const: Fixed values for participation rate equation
        hourly_coe_const: Fixed values for hourly COE equation
        include_exchange_rate: Whether to include TWI equation (default True)
        include_import_price: Whether to include import price pass-through (default True)
        include_participation: Whether to include participation rate equation (default True)
        include_hourly_coe: Whether to include hourly COE wage equation (default True)

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

    # Hourly COE wage equation (optional, default on)
    if include_hourly_coe:
        hourly_coe_equation(obs, model, nairu, constant=hourly_coe_const)

    # Labour supply equation (optional)
    if include_participation:
        participation_equation(obs, model, nairu, constant=participation_const)

    # Open economy equations (optional)
    if include_exchange_rate:
        exchange_rate_equation(obs, model, constant=exchange_rate_const)
    if include_import_price:
        import_price_equation(obs, model, constant=import_price_const)

    return model


# --- Save/Load Functions ---


def save_results(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
) -> Path:
    """Save model results to disk.

    Args:
        trace: ArviZ InferenceData from sampling
        obs: Observation dictionary
        obs_index: Period index for observations
        output_dir: Directory to save to (default: model_outputs/)
        prefix: Filename prefix

    Returns:
        Path to output directory

    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trace as NetCDF
    trace_path = output_dir / f"{prefix}_trace.nc"
    trace.to_netcdf(str(trace_path))
    print(f"Saved trace to: {trace_path}")

    # Save observations and index as pickle
    obs_path = output_dir / f"{prefix}_obs.pkl"
    with open(obs_path, "wb") as f:
        pickle.dump({"obs": obs, "obs_index": obs_index}, f)
    print(f"Saved observations to: {obs_path}")

    return output_dir


# --- Main Entry Point ---


def run_stage1(
    start: str | None = "1980Q1",
    end: str | None = None,
    config: SamplerConfig | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
    verbose: bool = False,
) -> tuple[az.InferenceData, dict[str, np.ndarray], pd.PeriodIndex]:
    """Run Stage 1: Build observations, sample model, and save results.

    Args:
        start: Start period
        end: End period
        config: Sampler configuration
        output_dir: Directory to save results
        prefix: Filename prefix for saved files
        verbose: Print progress messages

    Returns:
        Tuple of (trace, obs, obs_index)

    """
    if config is None:
        config = SamplerConfig(
            draws=10_000,
            tune=3_500,
            chains=5,
            cores=5,
            target_accept=0.90,
        )

    # Build observations
    print("Building observations...")
    obs, obs_index = build_observations(start=start, end=end, verbose=verbose)

    # Build model
    print("Building model...")
    model = build_model(obs)

    # Sample
    print("\nSampling...")
    trace = sample_model(model, config)
    print("\n")

    # Save results
    save_results(trace, obs, obs_index, output_dir=output_dir, prefix=prefix)

    return trace, obs, obs_index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NAIRU + Output Gap model (Stage 1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--start", type=str, default="1980Q1", help="Start period")
    parser.add_argument("--end", type=str, default=None, help="End period")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_stage1(
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
