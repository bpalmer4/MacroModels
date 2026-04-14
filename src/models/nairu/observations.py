"""Observation matrix building for NAIRU + Output Gap model.

Collates data from various data modules into a single observation dictionary
ready for model estimation.
"""

from typing import Literal, cast

import numpy as np
import pandas as pd

# --- Anchor Mode ---

AnchorMode = Literal["expectations", "target", "rba", "unanchored", "unanchored_raw"]

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
    "unanchored": "Anchor: RBA PIE_RBAQ → 1993 → phased → 1998 → Unanchored",
    "unanchored_raw": "Anchor: Unanchored model expectations (no phase-in)",
}

from src.data import (
    DataSeries,
    compute_mfp_trend_hma,
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
    get_labour_force_growth_qrtly,
    get_log_gdp,
    get_net_exports_ratio_change_qrtly,
    get_oil_change_lagged_annual,
    get_participation_rate_change_qrtly,
    get_real_wage_gap,
    get_trimmed_mean_annual,
    get_trimmed_mean_qrtly,
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
from src.data.expectations_model import get_model_expectations, get_model_expectations_unanchored
from src.data.expectations_rba import get_rba_expectations
from src.data.rba_loader import get_inflation_expectations as get_rba_raw_expectations
from src.models.nairu.config import REGIME_COVID_START, REGIME_GFC_START

# --- Constants ---

HMA_TERM = 13  # Henderson MA smoothing term
_NAME_WIDTH = 18  # Column width for variable names in loading output


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
    target_series = pd.Series(INFLATION_TARGET, index=result.index, dtype=float)
    return _phase_between(result, target_series)


def _phase_between(early: pd.Series, late: pd.Series) -> pd.Series:
    """Phase from `early` series (pre-PHASE_START) to `late` series (post-PHASE_END).

    Between PHASE_START and PHASE_END, uses linear blend: weight 0 → 1 on `late`.
    Both series must share the same PeriodIndex (quarterly).
    """
    result = early.copy()
    phase_periods = pd.period_range(PHASE_START, PHASE_END, freq="Q")
    n_periods = len(phase_periods)

    for i, period in enumerate(phase_periods):
        if period not in result.index:
            continue
        weight = i / (n_periods - 1)
        e = early.loc[period] if period in early.index else np.nan
        l = late.loc[period] if period in late.index else np.nan
        if pd.isna(e) or pd.isna(l):
            continue
        result.loc[period] = (1 - weight) * e + weight * l

    post_phase = result.index > PHASE_END
    result.loc[post_phase] = late.reindex(result.index[post_phase]).to_numpy()
    return result


def _build_rba_to_unanchored() -> pd.Series:
    """Build RBA-raw → Unanchored phased series.

    Pre-1993: RBA PIE_RBAQ raw (no target lock)
    1993-1998: linear phase from RBA raw to unanchored model median
    Post-1998: unanchored model median
    """
    rba_raw = get_rba_raw_expectations().data
    unanchored = get_model_expectations_unanchored().data

    # Align on a common quarterly index spanning both series
    start = min(rba_raw.index.min(), unanchored.index.min())
    end = max(rba_raw.index.max(), unanchored.index.max())
    full_index = pd.period_range(start, end, freq="Q")

    early = rba_raw.reindex(full_index).astype(float)
    late = unanchored.reindex(full_index).astype(float)

    # Pre-PHASE_START: use RBA raw
    result = pd.Series(index=full_index, dtype=float)
    pre_phase = full_index <= PHASE_START
    result.loc[pre_phase] = early.loc[pre_phase]

    # PHASE_START → PHASE_END: blend
    phase_periods = pd.period_range(PHASE_START, PHASE_END, freq="Q")
    n_periods = len(phase_periods)
    for i, period in enumerate(phase_periods):
        if period not in full_index:
            continue
        weight = i / (n_periods - 1)
        e = early.loc[period]
        l = late.loc[period]
        if pd.isna(e) and pd.isna(l):
            continue
        if pd.isna(e):
            result.loc[period] = l
        elif pd.isna(l):
            result.loc[period] = e
        else:
            result.loc[period] = (1 - weight) * e + weight * l

    # Post-PHASE_END: use unanchored
    post_phase = full_index > PHASE_END
    result.loc[post_phase] = late.loc[post_phase]
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


def build_observations(  # noqa: PLR0915 — flat data-loading sequence, not genuinely complex
    start: str | None = None,
    end: str | None = None,
    hma_term: int = HMA_TERM,
    anchor_mode: AnchorMode = "unanchored",
    verbose: bool = False,  # noqa: ARG001 — reserved for future use
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex, str, pd.DataFrame]:
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
            - "expectations": Use full estimated (target-anchored) expectations series
            - "target": Phase from expectations to 2.5% target (1993-1998)
            - "rba": RBA PIE_RBAQ, phased to 2.5% target (policy-counterfactual)
            - "unanchored" (default): RBA PIE_RBAQ phased to unanchored model post-1998
            - "unanchored_raw": unanchored model series, no phase-in
        verbose: Print sample info

    Returns:
        Tuple of (observations dict, period index, anchor label, chart DataFrame).
        The chart DataFrame has the same start date but extends to the latest
        available data for each series (no row-wise dropna).

    """
    # --- Load data from library ---
    print("Loading observation series:")

    def _load(var_name: str, ds: DataSeries) -> pd.Series:
        """Extract data from DataSeries, printing variable name and description."""
        print(f"  {var_name:<{_NAME_WIDTH}s}{ds.description or '(no description)'}")
        return ds.data

    # Unemployment
    U = _load("U", get_unemployment_rate_qrtly())
    U_1 = _load("U_1", get_unemployment_rate_lagged_qrtly())
    ΔU = _load("ΔU", get_unemployment_change_qrtly())
    ΔU_1_over_U = _load("ΔU_1_over_U", get_unemployment_speed_limit_qrtly())

    # Participation rate
    Δpr = _load("Δpr", get_participation_rate_change_qrtly())

    # GDP
    log_gdp = _load("log_gdp", get_log_gdp())

    # Production inputs (with model-specific smoothing)
    capital_growth = _prepare_capital_growth()
    print(f"  {'capital_growth':<{_NAME_WIDTH}s}Capital stock growth (quarterly)")
    lf_growth = _prepare_labour_force_growth()
    print(f"  {'lf_growth':<{_NAME_WIDTH}s}Labour force growth (quarterly, COVID-smoothed)")
    hours_growth = _prepare_hours_growth()
    print(f"  {'hours_growth':<{_NAME_WIDTH}s}Hours worked growth (quarterly, COVID-smoothed)")

    # Capital share from national accounts (time-varying)
    alpha = _load("alpha", get_capital_share())

    # Inflation
    π = _load("π", get_trimmed_mean_qrtly())
    π4 = _load("π4", get_trimmed_mean_annual())

    # Inflation expectations
    if anchor_mode == "rba":
        π_exp = _load("π_exp", get_rba_expectations())
    elif anchor_mode == "unanchored":
        print(f"  {'π_exp':<{_NAME_WIDTH}s}RBA PIE_RBAQ → phased 1993-1998 → Unanchored model")
        π_exp = _build_rba_to_unanchored()
    elif anchor_mode == "unanchored_raw":
        π_exp = _load("π_exp", get_model_expectations_unanchored())
    else:
        π_exp_raw = _load("π_exp", get_model_expectations())
        π_exp = apply_anchor_mode(π_exp_raw, anchor_mode)

    # Interest rates
    cash_rate = _load("cash_rate", get_cash_rate_qrtly())

    # Unit labour costs
    Δulc = _load("Δulc", get_ulc_growth_qrtly())
    Δulc_1 = _load("Δulc_1", get_ulc_growth_lagged_qrtly())

    # Hourly compensation of employees
    Δhcoe = _load("Δhcoe", get_hourly_coe_growth_qrtly())
    Δhcoe_1 = _load("Δhcoe_1", get_hourly_coe_growth_lagged_qrtly())

    # Derive MFP from wage data, HMA(101) smoothed
    mfp_growth = _load("mfp_growth", compute_mfp_trend_hma(
        ulc_growth=Δulc,
        hcoe_growth=Δhcoe,
        capital_growth=capital_growth,
        hours_growth=hours_growth,
        alpha=alpha,
    ))

    # Domestic final demand deflator
    Δ4dfd = _load("Δ4dfd", get_dfd_deflator_growth_annual())

    # Import prices
    Δ4ρm = _load("Δ4ρm", get_import_price_growth_annual())
    Δ4ρm_1 = _load("Δ4ρm_1", get_import_price_growth_lagged_annual())

    # GSCPI
    ξ_2 = _load("ξ_2", get_gscpi_covid_lagged_qrtly())

    # TWI
    Δtwi = _load("Δtwi", get_twi_change_qrtly())
    Δtwi_1 = _load("Δtwi_1", get_twi_change_lagged_qrtly())
    Δ4twi_1 = _load("Δ4twi_1", get_twi_change_lagged_annual())

    # Oil prices (AUD)
    Δ4oil_1 = _load("Δ4oil_1", get_oil_change_lagged_annual())

    # Fiscal impulse
    fiscal_impulse_1 = _load("fiscal_impulse_1", get_fiscal_impulse_lagged_qrtly())

    # Debt servicing ratio change
    Δdsr_1 = _load("Δdsr_1", get_dsr_change_lagged_qrtly())

    # Housing wealth growth
    Δhw_1 = _load("Δhw_1", get_housing_wealth_growth_lagged_qrtly())

    # Employment growth
    emp_growth = _load("emp_growth", get_employment_growth_qrtly())
    emp_growth_1 = _load("emp_growth_1", get_employment_growth_lagged_qrtly())

    # Real wage gap
    real_wage_gap = _load("real_wage_gap", get_real_wage_gap(Δulc, mfp_growth))

    # Net exports ratio change
    Δnx_ratio = _load("Δnx_ratio", get_net_exports_ratio_change_qrtly())

    # Compute r*
    r_star = _load("r_star", compute_r_star(
        capital_growth, lf_growth, mfp_growth, alpha, hma_term,
    ))

    # Real rate gap
    r_gap = cash_rate - π_exp - r_star
    r_gap_1 = r_gap.shift(1)
    print(f"  {'r_gap_1':<{_NAME_WIDTH}s}Lagged real rate gap (cash_rate - π_exp - r*)")

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

    # Forward-fill housing wealth growth (5232.0 lags other releases)
    # Δhw_1 is only used in stage3 forward sampling, not in estimation
    observed["Δhw_1"] = observed["Δhw_1"].ffill()

    # Charting version: same start date, extends to latest available per series
    chart_obs = observed.copy()

    # Drop missing (sampling version: complete cases only)
    observed = observed.dropna()

    print(f"Observations: {observed.index.min()} to {observed.index.max()} ({len(observed)} periods)")

    # Report chart data extent (latest period with any data)
    chart_last = chart_obs.apply(lambda s: s.last_valid_index()).max()
    if chart_last > observed.index.max():
        print(f"Chart data extends to: {chart_last}")

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

    return obs_dict, obs_index, anchor_label, chart_obs
