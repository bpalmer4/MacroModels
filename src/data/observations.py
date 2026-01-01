"""Observation matrix building for NAIRU + Output Gap model.

Collates data from various data modules into a single observation dictionary
ready for model estimation.
"""

from typing import cast

import numpy as np
import pandas as pd

from src.data import (
    compute_r_star,
    get_capital_growth_qrtly,
    get_capital_share,
    get_cash_rate_qrtly,
    get_dfd_deflator_growth_annual,
    get_employment_growth_lagged_qrtly,
    get_employment_growth_qrtly,
    get_fiscal_impulse_lagged_qrtly,
    get_gscpi_covid_lagged_qrtly,
    get_hourly_coe_growth_lagged_qrtly,
    get_hourly_coe_growth_qrtly,
    get_hours_growth_qrtly,
    get_import_price_growth_annual,
    get_import_price_growth_lagged_annual,
    get_inflation_annual,
    get_inflation_qrtly,
    get_labour_force_growth_qrtly,
    get_log_gdp,
    get_mfp_trend_floored,
    get_net_exports_ratio_change_qrtly,
    get_oil_change_lagged_annual,
    get_participation_rate_change_qrtly,
    get_real_wage_gap,
    get_twi_change_annual,
    get_twi_change_lagged_annual,
    get_twi_change_lagged_qrtly,
    get_twi_change_qrtly,
    get_ulc_growth_lagged_qrtly,
    get_ulc_growth_qrtly,
    get_unemployment_change_qrtly,
    get_unemployment_rate_lagged_qrtly,
    get_unemployment_rate_qrtly,
    get_unemployment_speed_limit_qrtly,
    hma,
)
from src.data.rba_loader import get_inflation_anchor
from src.models.nairu.equations import REGIME_COVID_START, REGIME_GFC_START

# --- Constants ---

HMA_TERM = 13  # Henderson MA smoothing term


# --- Data Preparation Helpers ---


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
    hours_growth_raw = get_hours_growth_qrtly().data
    hours_growth_smoothed = hma(hours_growth_raw.dropna(), HMA_TERM)
    hours_growth_smoothed = hours_growth_smoothed.reindex(hours_growth_raw.index)

    # Replace COVID period with smoothed values
    covid_period = pd.period_range("2020Q1", "2023Q2", freq="Q")
    return hours_growth_raw.where(
        ~hours_growth_raw.index.isin(covid_period),
        other=hours_growth_smoothed,
    )


# --- Main Function ---


def build_observations(
    start: str | None = None,
    end: str | None = None,
    hma_term: int = HMA_TERM,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex]:
    """Build observation dictionary for model.

    Loads all data from library, applies model-specific transformations,
    aligns to common sample, and returns as numpy arrays.

    Capital share (alpha) is loaded from ABS national accounts data:
    α = GOS / (GOS + COE), time-varying.

    Args:
        start: Start period (e.g., "1980Q1")
        end: End period
        hma_term: Henderson MA smoothing term (default 13)
        verbose: Print sample info

    Returns:
        Tuple of (observations dict, period index)

    """
    # --- Load data from library ---

    # Unemployment
    U = get_unemployment_rate_qrtly().data
    U_1 = get_unemployment_rate_lagged_qrtly().data
    ΔU = get_unemployment_change_qrtly().data
    ΔU_1_over_U = get_unemployment_speed_limit_qrtly().data

    # Participation rate
    Δpr = get_participation_rate_change_qrtly().data

    # GDP
    log_gdp = get_log_gdp().data

    # Production inputs (with model-specific smoothing)
    capital_growth = _prepare_capital_growth()
    lf_growth = _prepare_labour_force_growth()
    hours_growth = _prepare_hours_growth()

    # Capital share from national accounts (time-varying)
    alpha = get_capital_share().data

    # Inflation
    π = get_inflation_qrtly().data
    π4 = get_inflation_annual().data
    π_anchor = get_inflation_anchor().data

    # Interest rates
    cash_rate = get_cash_rate_qrtly().data

    # Unit labour costs
    Δulc = get_ulc_growth_qrtly().data
    Δulc_1 = get_ulc_growth_lagged_qrtly().data

    # Hourly compensation of employees (COE/hours - cleaner wage signal)
    Δhcoe = get_hourly_coe_growth_qrtly().data
    Δhcoe_1 = get_hourly_coe_growth_lagged_qrtly().data

    # Derive MFP from wage data (replaces external ABS 5204 MFP)
    mfp_growth = get_mfp_trend_floored(
        ulc_growth=Δulc,
        hcoe_growth=Δhcoe,
        capital_growth=capital_growth,
        hours_growth=hours_growth,
        alpha=alpha,
    ).data

    # Domestic final demand deflator (demand channel for wages)
    Δ4dfd = get_dfd_deflator_growth_annual().data

    # Import prices
    Δ4ρm = get_import_price_growth_annual().data
    Δ4ρm_1 = get_import_price_growth_lagged_annual().data

    # GSCPI (COVID supply chain pressure, masked and lagged)
    ξ_2 = get_gscpi_covid_lagged_qrtly().data

    # TWI (Trade-Weighted Index)
    Δtwi = get_twi_change_qrtly().data
    Δtwi_1 = get_twi_change_lagged_qrtly().data
    Δ4twi_1 = get_twi_change_lagged_annual().data

    # Oil prices (AUD)
    Δ4oil_1 = get_oil_change_lagged_annual().data

    # Fiscal impulse (lagged)
    fiscal_impulse_1 = get_fiscal_impulse_lagged_qrtly().data

    # Employment growth (for employment equation)
    emp_growth = get_employment_growth_qrtly().data
    emp_growth_1 = get_employment_growth_lagged_qrtly().data

    # Real wage gap: ULC growth minus MFP growth
    real_wage_gap = get_real_wage_gap(Δulc, mfp_growth).data

    # Net exports ratio change (for net exports equation)
    Δnx_ratio = get_net_exports_ratio_change_qrtly().data

    # Compute r*
    r_star = compute_r_star(
        capital_growth, lf_growth, mfp_growth, alpha, hma_term
    ).data

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
        "alpha_capital": alpha,
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
        # Employment (for employment equation)
        "emp_growth": emp_growth,
        "emp_growth_1": emp_growth_1,
        "real_wage_gap": real_wage_gap,
        # Net exports (for net exports equation)
        "Δnx_ratio": Δnx_ratio,
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
