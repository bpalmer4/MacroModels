"""Observation matrix building for NAIRU + Output Gap model.

Collates data from various data modules into a single observation dictionary
ready for model estimation.
"""

from typing import Literal, cast

import numpy as np
import pandas as pd

# --- Anchor Mode ---

AnchorMode = Literal["expectations", "target", "rba"]

# Inflation target (RBA's 2-3% band midpoint)
INFLATION_TARGET = 2.5

# Phasing period for target anchor mode
PHASE_START = pd.Period("1992Q4")  # Last quarter using pure expectations
PHASE_END = pd.Period("1998Q4")    # First quarter using pure target

# Labels for chart annotations
ANCHOR_LABELS = {
    "expectations": "Anchor: Estimated expectations",
    "target": "Anchor: Expectations → 1993 → phased → 1998 → Target",
    "rba": "Anchor: RBA PIE_RBAQ → 1993 → phased → 1998 → Target",
}

from src.data import (
    compute_mfp_trend_floored,
    compute_r_star,
    get_capital_growth_qrtly,
    get_capital_share,
    get_cash_rate_qrtly,
    get_dfd_deflator_growth_annual,
    get_dsr_change_lagged_qrtly,
    get_employment_growth_lagged_qrtly,
    get_employment_growth_qrtly,
    get_fiscal_impulse_lagged_qrtly,
    get_gscpi_covid_lagged_qrtly,
    get_hourly_coe_growth_lagged_qrtly,
    get_hourly_coe_growth_qrtly,
    get_hours_growth_qrtly,
    get_housing_wealth_growth_lagged_qrtly,
    get_import_price_growth_annual,
    get_import_price_growth_lagged_annual,
    get_trimmed_mean_annual,
    get_trimmed_mean_qrtly,
    get_labour_force_growth_qrtly,
    get_log_gdp,
    get_net_exports_ratio_change_qrtly,
    get_oil_change_lagged_annual,
    get_participation_rate_change_qrtly,
    get_real_wage_gap,
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
from src.data.expectations_model import get_model_expectations
from src.data.expectations_rba import get_rba_expectations
from src.models.nairu.equations import REGIME_COVID_START, REGIME_GFC_START

# --- Constants ---

HMA_TERM = 13  # Henderson MA smoothing term


# --- Anchor Mode Helpers ---


def apply_anchor_mode(
    expectations: pd.Series,
    anchor_mode: AnchorMode,
) -> pd.Series:
    """Apply anchor mode to expectations series.

    Args:
        expectations: Raw expectations series from signal extraction model
        anchor_mode: How to anchor expectations
            - "expectations": Use full series as-is
            - "target": Use expectations to PHASE_START, phase to target by PHASE_END,
                        then use target from PHASE_END onwards

    Returns:
        Anchored expectations series
    """
    if anchor_mode == "expectations":
        return expectations

    # Target mode: phase from expectations to target
    result = expectations.copy()

    # Calculate phase weights (linear interpolation)
    # 0 at PHASE_START, 1 at PHASE_END
    phase_periods = pd.period_range(PHASE_START, PHASE_END, freq="Q")
    n_periods = len(phase_periods)

    for i, period in enumerate(phase_periods):
        if period in result.index:
            # Weight increases linearly from 0 to 1
            weight = i / (n_periods - 1)
            exp_value = expectations.loc[period] if period in expectations.index else INFLATION_TARGET
            result.loc[period] = (1 - weight) * exp_value + weight * INFLATION_TARGET

    # After PHASE_END: use target
    post_phase = result.index > PHASE_END
    result.loc[post_phase] = INFLATION_TARGET

    return result


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
    """Prepare capital growth (raw, no smoothing needed for stock variable)."""
    return get_capital_growth_qrtly().data


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
    anchor_mode: AnchorMode = "rba",
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex, str]:
    """Build observation dictionary for model.

    Loads all data from library, applies model-specific transformations,
    aligns to common sample, and returns as numpy arrays.

    Capital share (alpha) is loaded from ABS national accounts data:
    α = GOS / (GOS + COE), time-varying.

    Args:
        start: Start period (e.g., "1980Q1")
        end: End period
        hma_term: Henderson MA smoothing term (default 13)
        anchor_mode: How to anchor expectations
            - "expectations": Use full estimated expectations series
            - "target": Phase from expectations to 2.5% target (1993-1998)
            - "rba": Use RBA PIE_RBAQ with same phase-in to target
        verbose: Print sample info

    Returns:
        Tuple of (observations dict, period index, anchor label)

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
    π = get_trimmed_mean_qrtly().data
    π4 = get_trimmed_mean_annual().data

    # Inflation expectations
    if anchor_mode == "rba":
        # Use RBA PIE_RBAQ with target phase-in
        π_exp = get_rba_expectations().data
    else:
        # Use model expectations (with optional target phase-in)
        π_exp_raw = get_model_expectations().data
        π_exp = apply_anchor_mode(π_exp_raw, anchor_mode)

    # Interest rates
    cash_rate = get_cash_rate_qrtly().data

    # Unit labour costs
    Δulc = get_ulc_growth_qrtly().data
    Δulc_1 = get_ulc_growth_lagged_qrtly().data

    # Hourly compensation of employees (COE/hours - cleaner wage signal)
    Δhcoe = get_hourly_coe_growth_qrtly().data
    Δhcoe_1 = get_hourly_coe_growth_lagged_qrtly().data

    # Derive MFP from wage data (replaces external ABS 5204 MFP)
    mfp_growth = compute_mfp_trend_floored(
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

    # Debt servicing ratio change (lagged, for IS curve credit channel)
    Δdsr_1 = get_dsr_change_lagged_qrtly().data

    # Housing wealth growth (lagged, for IS curve wealth channel)
    Δhw_1 = get_housing_wealth_growth_lagged_qrtly().data

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
    r_gap = cash_rate - π_exp - r_star
    r_gap_1 = r_gap.shift(1)

    # Check expectations end date vs GDP
    gdp_end = log_gdp.last_valid_index()
    exp_end = π_exp.last_valid_index()
    if exp_end < gdp_end:
        print(
            f"WARNING: Expectations end ({exp_end}) is before GDP end ({gdp_end}). "
            "Re-run expectations model to update."
        )

    # Build DataFrame
    observed = pd.DataFrame({
        # Inflation
        "π": π,
        "π4": π4,
        "π_exp": π_exp,
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
        # Debt servicing (for IS curve credit channel)
        "Δdsr_1": Δdsr_1,
        # Housing wealth (for IS curve wealth channel)
        "Δhw_1": Δhw_1,
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

    # Get anchor label for chart annotations
    anchor_label = ANCHOR_LABELS[anchor_mode]

    return obs_dict, obs_index, anchor_label
