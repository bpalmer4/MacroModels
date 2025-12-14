"""NAIRU + Output Gap joint estimation model.

Bayesian state-space model that jointly estimates:
- NAIRU (Non-Accelerating Inflation Rate of Unemployment)
- Potential output (via Cobb-Douglas production function)
- Output gap and unemployment gap

Equations:
1. NAIRU: Gaussian random walk (state equation)
2. Potential output: Cobb-Douglas with time-varying drift (state equation)
3. Okun's Law: Links output gap to unemployment changes
4. Phillips Curve: Links inflation to unemployment gap
5. Wage Phillips Curve: Links wage growth to unemployment gap
6. IS Curve: Links output gap to real interest rate gap

Data sources:
- ABS: GDP, unemployment, CPI, hours worked, capital stock, MFP
- RBA: Cash rate, inflation expectations
"""

from dataclasses import dataclass
from typing import Any, cast

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.data import load_series, get_abs_data, hma, DataSeries, splice_series
from src.data.series_specs import (
    CPI_TRIMMED_MEAN_QUARTERLY,
    CPI_TRIMMED_MEAN_ANNUAL,
    GDP_CVM,
    HOURS_WORKED_INDEX,
    CAPITAL_STOCK,
    HOURS_WORKED,
    LABOUR_FORCE_TOTAL,
    UNEMPLOYED_TOTAL,
    MFP_HOURS_WORKED,
    COMPENSATION_OF_EMPLOYEES,
    HISTORICAL_RATE_FILE,
)
from src.data.rba_loader import (
    get_cash_rate as rba_get_cash_rate,
    get_historical_interbank_rate,
    get_inflation_anchor as rba_get_inflation_anchor,
    PI_TARGET,
    PI_TARGET_START,
    PI_TARGET_FULL,
)
from src.equations import (
    nairu_equation,
    potential_output_equation,
    okun_law_equation,
    price_inflation_equation,
    wage_growth_equation,
    is_equation,
)
from src.models.base import SamplerConfig, sample_model
from src.analysis import (
    check_for_zero_coeffs,
    check_model_diagnostics,
    plot_posteriors_bar,
    plot_posteriors_kde,
    posterior_predictive_checks,
    residual_autocorrelation_analysis,
)


# --- Constants ---

ALPHA = 0.3  # Capital share for Cobb-Douglas
HMA_TERM = 13  # Henderson MA smoothing term

# Plotting constants (from notebook)
MODEL_NAME = "Joint NAIRU + Output Gap Model"
RFOOTER_OUTPUT = "Joint NAIRU + Output Gap Model"
START = pd.Period("1985Q1", freq="Q")

# NAIRU warning region (before inflation target fully anchored)
NAIRU_WARN = {
    "axvspan": {
        "xmin": START.ordinal,
        "xmax": PI_TARGET_FULL.ordinal,
        "label": r"NAIRU ($U^*$) WRT $\pi^e$ (before inflation target fully anchored)",
        "color": "goldenrod",
        "alpha": 0.2,
        "zorder": -2,
    }
}

# Inflation target ranges for plots
QUARTERLY_RANGE = {
    "axhspan": {
        "ymin": (pow(1.02, 0.25) - 1) * 100,
        "ymax": (pow(1.03, 0.25) - 1) * 100,
        "color": "#ffdddd",
        "label": "Quarterly growth consistent with 2-3% annual inflation target",
        "zorder": -1,
    }
}

ANNUAL_RANGE = {
    "axhspan": {
        "ymin": 2,
        "ymax": 3,
        "color": "#dddddd",
        "label": "2-3% annual inflation target range",
        "zorder": -1,
    }
}

ANNUAL_TARGET = {
    "axhline": {
        "y": 2.5,
        "linestyle": "dashed",
        "linewidth": 0.75,
        "color": "darkred",
        "label": "2.5% annual inflation target",
    }
}


# --- Data Preparation ---


def get_unemployment_data() -> dict[str, pd.Series]:
    """Load unemployment rate and compute derived series.

    Uses Modellers Database (1364.0.15.003) which provides quarterly data
    covering total population including defence/non-civilian employment.
    Unemployment rate is calculated as: unemployed / labour_force * 100
    """
    data = get_abs_data({
        "LF": LABOUR_FORCE_TOTAL,
        "Unemp": UNEMPLOYED_TOTAL,
    })

    lf = data["LF"].data
    unemp = data["Unemp"].data

    # Calculate unemployment rate
    U = (unemp / lf) * 100

    ΔU = U.diff(1)
    ΔU_1 = ΔU.shift(1)

    return {
        "U": U,
        "ΔU": ΔU,
        "ΔU_1": ΔU_1,
        "ΔU_1_over_U": ΔU_1 / U,
    }


def get_gdp_data() -> dict[str, pd.Series]:
    """Load GDP and compute log levels and growth."""
    gdp = load_series(GDP_CVM)
    gdp_series = gdp.data

    log_gdp = np.log(gdp_series) * 100  # Scale for numerical stability
    gdp_growth = log_gdp.diff(1)

    return {
        "log_gdp": log_gdp,
        "gdp_growth": gdp_growth,
    }


def get_productivity_trend(trend_weight: float = 0.75) -> pd.Series:
    """Get labour productivity trend for backfilling MFP.

    Calculates GDP per hour worked and blends with linear trend.
    Used to extend MFP series before ABS 5204 data is available.

    Args:
        trend_weight: Weight on linear trend (default 0.75)

    Returns:
        Blended quarterly productivity growth
    """
    data = get_abs_data({
        "GDP": GDP_CVM,
        "Hours": HOURS_WORKED_INDEX,
    })

    gdp = data["GDP"].data
    hours = data["Hours"].data

    productivity_index = gdp / hours
    log_productivity = np.log(productivity_index) * 100
    productivity_growth = log_productivity.diff(1).dropna()

    # Fit linear trend
    x = np.arange(len(productivity_growth))
    slope, intercept = np.polyfit(x, productivity_growth.values, 1)
    linear_trend = pd.Series(intercept + slope * x, index=productivity_growth.index)

    # Blend: trend_weight × linear + (1 - trend_weight) × raw
    productivity_blend = trend_weight * linear_trend + (1 - trend_weight) * productivity_growth

    return productivity_blend


def get_production_inputs() -> dict[str, pd.Series]:
    """Load production function inputs: capital, labor, MFP."""
    import readabs as ra

    data = get_abs_data({
        "Capital": CAPITAL_STOCK,
        "Hours": HOURS_WORKED,
        "LF": LABOUR_FORCE_TOTAL,
    })

    # Hours worked - convert monthly to quarterly (sum of 3 months)
    hours_monthly = data["Hours"].data
    hours = ra.monthly_to_qtly(hours_monthly, q_ending="DEC", f="sum")

    # Capital growth (Henderson smoothed)
    capital = data["Capital"].data
    log_capital = np.log(capital) * 100
    capital_growth_raw = log_capital.diff(1)
    capital_growth = hma(capital_growth_raw.dropna(), HMA_TERM)
    # Reindex to original
    capital_growth = capital_growth.reindex(capital.index)

    # Labor force growth (Henderson smoothed during COVID)
    lf = data["LF"].data
    log_lf = np.log(lf) * 100
    lf_growth_raw = log_lf.diff(1)
    lf_growth_smoothed = hma(lf_growth_raw.dropna(), HMA_TERM)
    lf_growth_smoothed = lf_growth_smoothed.reindex(lf.index)

    # Replace COVID period with smoothed values to keep drift smooth
    covid_period = pd.period_range("2020Q1", "2023Q2", freq="Q")
    lf_growth = lf_growth_raw.where(
        ~lf_growth_raw.index.isin(covid_period),
        other=lf_growth_smoothed,
    )

    # Get productivity trend for backfilling MFP
    productivity_trend = get_productivity_trend()

    # MFP growth (from ABS productivity tables - annual data)
    mfp = load_series(MFP_HOURS_WORKED)
    mfp_annual = mfp.data
    # Smooth annual data
    mfp_smoothed = hma(mfp_annual.dropna(), 25)

    # Convert annual to quarterly contribution (÷4) and interpolate to quarterly frequency
    mfp_quarterly_rate = ((1 + mfp_smoothed / 100) ** 0.25 - 1) * 100
    # Resample to quarterly: convert to timestamp, resample, back to period
    mfp_quarterly_rate = (
        mfp_quarterly_rate.to_timestamp(how="end")
        .resample("QE-DEC")
        .last()
        .to_period("Q-DEC")
    ).interpolate()

    # Backfill MFP with productivity_trend where MFP is unavailable
    mfp_quarterly_rate = mfp_quarterly_rate.reindex(productivity_trend.index)
    mfp_final = mfp_quarterly_rate.where(mfp_quarterly_rate.notna(), other=productivity_trend)

    return {
        "capital_growth": capital_growth,
        "lf_growth": lf_growth,
        "mfp_growth": mfp_final,
    }


def get_inflation_data() -> dict[str, pd.Series]:
    """Load inflation and anchor series.

    Uses actual quarterly and annual trimmed mean inflation from ABS,
    NOT computed quarterly from annual.
    """
    # Load actual quarterly trimmed mean (percentage change from previous period)
    quarterly = load_series(CPI_TRIMMED_MEAN_QUARTERLY)
    π = quarterly.data

    # Load actual annual trimmed mean (percentage change from corresponding quarter)
    annual = load_series(CPI_TRIMMED_MEAN_ANNUAL)
    π4 = annual.data

    # Inflation anchor (expectations → target transition)
    anchor = rba_get_inflation_anchor()
    π_anchor = anchor.data

    return {
        "π": π,
        "π4": π4,
        "π_anchor": π_anchor,
    }


def get_cash_rate_data() -> dict[str, pd.Series]:
    """Load cash rate data, splicing OCR with historical interbank rate.

    Combines:
    - Modern OCR from RBA (1990-present)
    - Historical interbank overnight rate (pre-1990)

    Favours OCR where both have data. Quarterly is end-of-quarter value.
    Note: Early OCR records with range values are excluded.
    """
    import readabs as ra

    # Get raw data from loaders
    ocr = rba_get_cash_rate().data
    historical = get_historical_interbank_rate(HISTORICAL_RATE_FILE).data

    # Exclude early OCR records with non-numeric (range) values
    ocr = pd.to_numeric(ocr, errors="coerce").dropna()

    # Convert historical index to PeriodIndex if needed
    if not isinstance(historical.index, pd.PeriodIndex):
        historical.index = pd.PeriodIndex(historical.index, freq="M")

    # Splice: favour OCR where both have data
    monthly = splice_series(ocr, historical)

    # Convert to quarterly (end of quarter value)
    quarterly = monthly.sort_index().groupby(monthly.index.asfreq("Q")).last()

    return {
        "cash_rate": quarterly,
        "cash_rate_monthly": monthly,
    }


def get_ulc_data() -> dict[str, pd.Series]:
    """Load unit labor costs data.

    Calculates quarterly ULC growth from National Accounts:
    - GDP (Chain volume measures)
    - Compensation of employees
    ULC = compensation / GDP, then log difference for growth rate.
    """
    data = get_abs_data({
        "GDP": GDP_CVM,
        "CoE": COMPENSATION_OF_EMPLOYEES,
    })

    gdp = data["GDP"].data
    coe = data["CoE"].data

    ulc = coe / gdp
    log_ulc = np.log(ulc)
    delta_ulc = log_ulc.diff(1).dropna() * 100

    return {
        "Δulc": delta_ulc,
    }


def get_import_pricing() -> dict[str, pd.Series]:
    """Get lagged annual change in import prices.

    From ABS 6457.0 Import Price Index (series A2298279F).
    Returns 4-quarter log difference of import prices, lagged 1 and 2 periods.
    """
    from readabs import read_abs_series

    # Import Price Index by Balance of Payments, index, original
    trade, _trade_meta = read_abs_series(cat="6457.0", series_id="A2298279F")
    log_import_prices = trade["A2298279F"].apply(np.log)
    delta4_log_import_prices = log_import_prices.diff(periods=4).dropna() * 100
    dlip_1 = delta4_log_import_prices.shift(periods=1).dropna()
    dlip_2 = delta4_log_import_prices.shift(periods=2).dropna()

    return {
        "Δ4ρm_1": dlip_1,
        "Δ4ρm_2": dlip_2,
    }


def get_gscpi() -> dict[str, pd.Series]:
    """Global Supply Chain Price Index for COVID supply shock.

    From NY Fed: https://www.newyorkfed.org/research/policy/gscpi
    Only non-zero during COVID period (2020Q1-2023Q2).
    """
    from pathlib import Path
    import readabs as ra

    # Load GSCPI data from data directory
    gscpi_path = Path(__file__).parent.parent.parent / "data" / "gscpi_data.xls"

    gscpi = pd.read_excel(
        gscpi_path,
        sheet_name="GSCPI Monthly Data",
        index_col=0,
        parse_dates=True,
    )["GSCPI"]
    gscpi = ra.monthly_to_qtly(gscpi, q_ending="DEC", f="mean")
    gscpi.index = pd.PeriodIndex(gscpi.index, freq="Q")
    gscpi_1 = gscpi.shift(1)
    gscpi_2 = gscpi.shift(2)

    # Only use during COVID period (2020Q1-2023Q2), zero otherwise
    quarter = pd.Timestamp.today().to_period("Q")
    dummy = pd.Series(1, pd.period_range(start="1959Q1", end=quarter - 1, freq="Q"))
    mask = (dummy.index >= "2020Q1") & (dummy.index <= "2023Q2")
    dummy[mask] = 0  # key dates for the COVID period
    gscpi_1 = gscpi_1.where(dummy == 0, other=0).reindex(dummy.index).fillna(0)
    gscpi_2 = gscpi_2.where(dummy == 0, other=0).reindex(dummy.index).fillna(0)

    return {
        "ξ_1": gscpi_1,
        "ξ_2": gscpi_2,
    }


def compute_r_star(
    capital_growth: pd.Series,
    lf_growth: pd.Series,
    mfp_growth: pd.Series,
    alpha: float = ALPHA,
) -> pd.Series:
    """Compute deterministic r* as smoothed potential growth.

    r* ≈ α×g_K + (1-α)×g_L + g_MFP (annualized and smoothed)
    """
    # Quarterly potential growth
    quarterly_growth = (
        alpha * capital_growth
        + (1 - alpha) * lf_growth
        + mfp_growth
    )

    # Annualize (4Q rolling sum)
    annual_growth = quarterly_growth.rolling(4).sum()
    annual_growth = annual_growth.bfill()

    # Henderson smooth
    r_star = hma(annual_growth.dropna(), HMA_TERM)

    return r_star


def build_observations(
    start: str | None = None,
    end: str | None = None,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex]:
    """Build observation dictionary for model.

    Loads all data, aligns to common sample, and returns as numpy arrays.

    Args:
        start: Start period (e.g., "1980Q1")
        end: End period
        verbose: Print sample info

    Returns:
        Tuple of (observations dict, period index)
    """
    # Load all data
    unemployment = get_unemployment_data()
    gdp = get_gdp_data()
    production = get_production_inputs()
    inflation = get_inflation_data()
    rates = get_cash_rate_data()
    ulc = get_ulc_data()
    import_prices = get_import_pricing()
    gscpi = get_gscpi()

    # Compute r*
    r_star = compute_r_star(
        production["capital_growth"],
        production["lf_growth"],
        production["mfp_growth"],
    )

    # Build DataFrame
    observed = pd.DataFrame({
        # Inflation
        "π": inflation["π"],
        "π4": inflation["π4"],
        "π_anchor": inflation["π_anchor"],
        # Unemployment
        "U": unemployment["U"],
        "ΔU": unemployment["ΔU"],
        "ΔU_1": unemployment["ΔU_1"],
        "ΔU_1_over_U": unemployment["ΔU_1_over_U"],
        # GDP
        "log_gdp": gdp["log_gdp"],
        "gdp_growth": gdp["gdp_growth"],
        # Production inputs
        "capital_growth": production["capital_growth"],
        "lf_growth": production["lf_growth"],
        "mfp_growth": production["mfp_growth"],
        # Rates
        "cash_rate": rates["cash_rate"],
        "det_r_star": r_star,
        # Unit labor costs
        "Δulc": ulc["Δulc"],
        # Import prices and supply shocks
        "Δ4ρm_1": import_prices["Δ4ρm_1"],
        "ξ_2": gscpi["ξ_2"],
    })

    # Apply sample period
    if start:
        observed = observed[observed.index >= pd.Period(start)]
    if end:
        observed = observed[observed.index <= pd.Period(end)]

    # Drop missing
    observed = observed.dropna()

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

    if verbose:
        print(f"Sample: {observed.index[0]} to {observed.index[-1]} ({len(observed)} periods)")

    # Convert to dict of numpy arrays
    obs_dict = {col: observed[col].to_numpy() for col in observed.columns}
    obs_index = cast(pd.PeriodIndex, observed.index)

    return obs_dict, obs_index


# --- Model Assembly ---


def build_model(
    obs: dict[str, np.ndarray],
    nairu_const: dict[str, Any] | None = None,
    potential_const: dict[str, Any] | None = None,
) -> pm.Model:
    """Build the joint NAIRU + Output Gap model.

    Args:
        obs: Observation dictionary from build_observations()
        nairu_const: Fixed values for NAIRU equation
        potential_const: Fixed values for potential output equation

    Returns:
        PyMC Model ready for sampling
    """
    if nairu_const is None:
        nairu_const = {"nairu_innovation": 0.25}
    if potential_const is None:
        potential_const = {"potential_innovation": 0.2}

    model = pm.Model()

    # State equations
    nairu = nairu_equation(obs, model, constant=nairu_const)
    potential = potential_output_equation(obs, model, constant=potential_const)

    # Observation equations
    okun_law_equation(obs, model, nairu, potential)
    price_inflation_equation(obs, model, nairu)
    wage_growth_equation(obs, model, nairu)
    is_equation(obs, model, potential)

    return model


# --- Results Container ---


@dataclass
class NAIRUResults:
    """Results from NAIRU + Output Gap estimation."""

    trace: az.InferenceData
    obs: dict[str, np.ndarray]
    obs_index: pd.PeriodIndex
    model: pm.Model

    def nairu_posterior(self) -> pd.DataFrame:
        """Extract NAIRU posterior as DataFrame."""
        from src.analysis import get_vector_var
        samples = get_vector_var("nairu", self.trace)
        samples.index = self.obs_index
        return samples

    def potential_posterior(self) -> pd.DataFrame:
        """Extract potential output posterior as DataFrame."""
        from src.analysis import get_vector_var
        samples = get_vector_var("potential_output", self.trace)
        samples.index = self.obs_index
        return samples

    def nairu_median(self) -> pd.Series:
        """NAIRU point estimate (posterior median)."""
        return self.nairu_posterior().median(axis=1)

    def potential_median(self) -> pd.Series:
        """Potential output point estimate (posterior median)."""
        return self.potential_posterior().median(axis=1)

    def unemployment_gap(self) -> pd.Series:
        """Unemployment gap = U - NAIRU."""
        U = pd.Series(self.obs["U"], index=self.obs_index)
        return U - self.nairu_median()

    def output_gap(self) -> pd.Series:
        """Output gap = log(GDP) - log(potential)."""
        log_gdp = pd.Series(self.obs["log_gdp"], index=self.obs_index)
        return log_gdp - self.potential_median()


# --- Main Entry Point ---


def run_model(
    start: str | None = "1980Q1",
    end: str | None = None,
    config: SamplerConfig | None = None,
    verbose: bool = False,
) -> NAIRUResults:
    """Run the full NAIRU + Output Gap estimation.

    Args:
        start: Start period
        end: End period
        config: Sampler configuration
        verbose: Print progress messages

    Returns:
        NAIRUResults with trace and computed series
    """
    if config is None:
        config = SamplerConfig()

    # Build observations
    obs, obs_index = build_observations(start=start, end=end)

    # Build model
    model = build_model(obs)

    # Sample
    print("Sampling...")
    trace = sample_model(model, config)

    return NAIRUResults(
        trace=trace,
        obs=obs,
        obs_index=obs_index,
        model=model,
    )


# --- Plotting Functions ---


def plot_nairu(
    results: NAIRUResults,
    show: bool = False,
) -> None:
    """Plot the NAIRU with unemployment and inflation overlay."""
    import mgplot as mg
    from src.analysis import plot_timeseries

    # NAIRU with credible intervals
    ax = plot_timeseries(
        trace=results.trace,
        var="nairu",
        index=results.obs_index,
        legend_stem="NAIRU",
        color="blue",
        start=START,
    )

    # Unemployment and inflation overlay with white background trick
    U = pd.Series(results.obs["U"], index=results.obs_index)
    U = U[U.index >= START]
    π4 = pd.Series(results.obs["π4"], index=results.obs_index)
    π4 = π4[π4.index >= START]

    back, front = 3, 1.5
    for color, width, label in zip(["white", ""], [back, front], ["_", ""]):
        U.name = "Unemployment Rate" if not label else label
        mg.line_plot(U, ax=ax, color=color if color else "brown", width=width)
        π4.name = "Inflation rate" if not label else label
        mg.line_plot(π4, ax=ax, color=color if color else "darkorange", width=width)

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="NAIRU Estimate for Australia",
            ylabel="Per cent",
            legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
            lfooter=r"Australia. $NAIRU = U^*$ "
                    f"WRT inflation expectations → {PI_TARGET_START} → blended to target → {PI_TARGET_FULL} → inflation target. ",
            rheader=f"From {PI_TARGET_FULL} onwards, the model estimates NAIRU as the unemployment rate needed to hit the inflation target.",
            rfooter=RFOOTER_OUTPUT,
            **ANNUAL_RANGE,
            **ANNUAL_TARGET,
            **NAIRU_WARN,
            show=show,
        )


def plot_unemployment_gap(
    results: NAIRUResults,
    show: bool = False,
    verbose: bool = False,
) -> None:
    """Plot the unemployment gap (U - U*)."""
    import mgplot as mg
    from src.analysis import plot_timeseries, get_vector_var

    start = START

    # Get NAIRU samples and calculate unemployment gap for each sample
    nairu = get_vector_var("nairu", results.trace)
    nairu.index = results.obs_index
    U = pd.Series(results.obs["U"], index=results.obs_index)
    u_gap = nairu.apply(lambda col: U - col)
    if verbose:
        print("Last data point:", u_gap.index[-1])

    # Plot using shared time series function
    ax = plot_timeseries(
        data=u_gap,
        legend_stem="Unemployment Gap",
        color="darkred",
        start=start,
    )

    # Finalise plot
    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Unemployment Gap Estimate for Australia",
            ylabel="Percentage points (U - U*)",
            lfooter=r"Australia. $U\text{-}gap = U - U^*$. Positive = slack/disinflationary, Negative = tight/inflationary.",
            rfooter=RFOOTER_OUTPUT,
            legend={"loc": "best", "fontsize": "x-small"},
            y0=True,
            **NAIRU_WARN,
            show=show,
        )


def plot_output_gap(
    results: NAIRUResults,
    show: bool = False,
) -> None:
    """Plot the output gap as percentage deviation from potential."""
    import mgplot as mg
    from src.analysis import plot_timeseries, get_vector_var

    start = START

    # Get potential output samples and calculate output gap for each sample
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    # Calculate output gap: (Y - Y*)/Y* * 100
    actual_gdp = results.obs["log_gdp"]
    output_gap = (actual_gdp[:, np.newaxis] - potential.values) / potential.values * 100
    output_gap = pd.DataFrame(output_gap, index=results.obs_index)

    # Plot using shared time series function
    ax = plot_timeseries(
        data=output_gap,
        legend_stem="Output Gap",
        color="green",
        start=start,
    )

    # Finalise plot
    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Output Gap Estimate for Australia",
            ylabel="Per cent of potential GDP",
            legend={"loc": "best", "fontsize": "x-small"},
            lfooter="Australia. (log Y - log Y*) / log Y* × 100. Positive = overheating/inflationary.",
            rfooter=RFOOTER_OUTPUT,
            y0=True,
            show=show,
        )


def plot_gdp_vs_potential(
    results: NAIRUResults,
    show: bool = False,
) -> None:
    """Plot actual GDP against potential GDP estimates."""
    import mgplot as mg
    from src.analysis import plot_timeseries, get_vector_var

    # Get potential output samples
    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    # Plot potential GDP using shared time series function
    ax = plot_timeseries(
        data=potential,
        legend_stem="Potential GDP",
        color="green",
        start=pd.Period("1985Q1", freq="Q"),
    )

    # Plot actual GDP on top
    actual = pd.Series(results.obs["log_gdp"], index=results.obs_index)
    actual = actual.reindex(results.obs_index)
    actual.name = "Actual GDP"
    mg.line_plot(
        actual,
        ax=ax,
        color="black",
        width=1.5,
    )

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Actual vs Potential GDP",
            ylabel="Log GDP (scaled)",
            legend={"loc": "upper left", "fontsize": "x-small"},
            lfooter="Australia. Log real GDP scaled by 100. ",
            rfooter=RFOOTER_OUTPUT,
            show=show,
        )


def plot_potential_growth(
    results: NAIRUResults,
    r_star_trend_weight: float = 0.75,
    show: bool = False,
    verbose: bool = False,
) -> None:
    """Plot annual potential GDP growth (4Q difference of log potential).

    This serves as a proxy for r* (the natural rate of interest), based on
    the theoretical relationship r* ≈ trend real GDP growth.
    """
    import mgplot as mg
    from scipy import stats
    from src.analysis import plot_timeseries, get_vector_var

    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    # r* = annual potential growth
    r_star = potential.diff(4).dropna()

    # Plot 1: Potential growth with credible intervals
    ax = plot_timeseries(
        data=r_star,
        legend_stem="Potential Growth",
        color="purple",
        start=pd.Period("1985Q1", freq="Q"),
    )

    # Add trend line
    median = r_star.quantile(0.5, axis=1)
    x = np.arange(len(median))
    slope, intercept, *_ = stats.linregress(x, median.values)
    trend = pd.Series(intercept + slope * x, index=median.index)
    trend.name = f"Trend (slope: {slope * 4:.2f}pp/year)"
    mg.line_plot(trend, ax=ax, color="darkred", width=1.5, style="--")

    if verbose:
        print(f"Chart 1 - Potential Growth:")
        print(f"  Median endpoint: {median.iloc[-1]:.3f}% at {median.index[-1]}")
        print(f"  Trend endpoint: {trend.iloc[-1]:.3f}%")

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Potential GDP Growth Rate (proxy for $r^*$)",
            ylabel="Per cent per annum",
            legend={"loc": "upper right", "fontsize": "x-small"},
            lfooter="Australia. 4-quarter change in log potential GDP. r* ≈ trend growth.",
            rfooter=RFOOTER_OUTPUT,
            y0=True,
            show=show,
        )

    # Plot 2: r* smoothing comparison
    w = r_star_trend_weight
    hybrid = (1 - w) * median + w * trend

    if verbose:
        print(f"\nChart 2 - r* Comparison:")
        print(f"  Median (raw) endpoint: {median.iloc[-1]:.3f}%")
        print(f"  Trend endpoint: {trend.iloc[-1]:.3f}%")
        print(f"  Hybrid ({int(w*100)}% trend, {int((1-w)*100)}% raw) endpoint: {hybrid.iloc[-1]:.3f}%")
        print(f"  Check: {(1-w):.2f} × {median.iloc[-1]:.3f} + {w:.2f} × {trend.iloc[-1]:.3f} = {(1-w)*median.iloc[-1] + w*trend.iloc[-1]:.3f}%")

    median.name = "$r^*$ raw median (no smoothing)"
    trend.name = "Trend only"
    hybrid.name = f"Hybrid ({int(w*100)}% trend, {int((1-w)*100)}% raw)"

    ax = mg.line_plot(median, color="darkblue", width=1)
    mg.line_plot(trend, ax=ax, style="--", color="darkorange", width=1)
    mg.line_plot(hybrid, ax=ax, width=2, color="darkred", annotate=True)

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Natural Rate of Interest (r*) - Comparison",
            ylabel="Per cent per annum",
            legend={"loc": "upper right", "fontsize": "x-small"},
            lfooter="Australia. Raw model median vs linear trend vs hybrid.",
            rfooter=RFOOTER_OUTPUT,
            y0=True,
            show=show,
        )


def plot_taylor_rule(
    results: NAIRUResults,
    inflation_annual: pd.Series,
    cash_rate_monthly: pd.Series,
    pi_target: float = PI_TARGET,
    pi_coef_start: float = 1.6,
    pi_coef_end: float = 1.25,
    r_star_trend_weight: float = 0.75,
    show: bool = False,
) -> None:
    """Plot Taylor Rule prescribed rate vs actual RBA cash rate.

    Taylor Rule: i = r* + π_coef·π - 0.5·πᵗ + 0.5·y_gap
    where πᵗ is the inflation target (2.5%)
    """
    import mgplot as mg
    from scipy import stats
    from src.analysis import plot_timeseries, get_vector_var

    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    # r* = annual potential growth
    r_star = potential.diff(4).dropna()

    # Calculate raw median and trend for reporting
    median = r_star.quantile(0.5, axis=1)
    slope, intercept, *_ = stats.linregress(np.arange(len(median)), median.values)
    trend = intercept + slope * np.arange(len(median))

    # Current r* values for annotation
    r_star_raw = median.iloc[-1]
    r_star_trend = trend[-1]
    w = r_star_trend_weight
    r_star_hybrid = (1 - w) * r_star_raw + w * r_star_trend

    # Smooth r* toward trend
    if r_star_trend_weight > 0:
        r_star = r_star.multiply(1 - w).add(trend * w, axis=0)

    # Output gap: (Y - Y*)/Y* × 100
    log_gdp = pd.Series(results.obs["log_gdp"], index=results.obs_index)
    actual_gdp = log_gdp.reindex(results.obs_index).values
    output_gap = (actual_gdp[:, np.newaxis] - potential.values) / potential.values * 100
    output_gap = pd.DataFrame(output_gap, index=results.obs_index, columns=potential.columns)
    output_gap = output_gap.reindex(r_star.index)

    # Time-varying inflation coefficient
    pi = inflation_annual.reindex(r_star.index)
    pi_coef = pd.Series(
        np.linspace(pi_coef_start, pi_coef_end, len(r_star)),
        index=r_star.index
    )

    # Taylor Rule for each sample
    taylor = (
        r_star
        .add(pi_coef * pi, axis=0)
        .add(-0.5 * pi_target)
        .add(output_gap.multiply(0.5))
    ).dropna()

    # Convert to monthly for cash rate alignment
    monthly_idx = taylor.index.to_timestamp(how='end').to_period('M')
    taylor_monthly = taylor.copy()
    taylor_monthly.index = monthly_idx

    # Plot
    ax = plot_timeseries(
        data=taylor_monthly,
        legend_stem="Taylor Rule",
        color="darkblue",
        start=None,
    )

    cash_rate_monthly.name = "RBA Cash Rate"
    mg.line_plot(cash_rate_monthly, ax=ax, color="#dd0000", width=1,
                drawstyle="steps-post", annotate=True)

    if ax is not None:
        mg.finalise_plot(
            ax,
            title="Taylor Rule vs RBA Cash Rate",
            ylabel="Per cent per annum",
            legend={"loc": "upper right", "fontsize": "x-small"},
            lfooter=f"Australia. Taylor Rule: i = r* + π_coef·π - 0.5πᵗ + 0.5·y_gap; "
                    +f"π_coef={pi_coef_start}→{pi_coef_end}; πᵗ={pi_target}%",
            rfooter=f"Final r*={r_star_hybrid:.1f}% ({int(w*100)}% trend {r_star_trend:.1f}%, "
                    +f"{int((1-w)*100)}% raw {r_star_raw:.1f}%)",
            rheader=RFOOTER_OUTPUT,
            y0=True,
            show=show,
        )


def plot_equilibrium_rates(
    results: NAIRUResults,
    cash_rate_monthly: pd.Series,
    pi_target: float = PI_TARGET,
    show: bool = False,
) -> None:
    """Plot neutral interest rate vs actual RBA cash rate."""
    import mgplot as mg
    from scipy import stats
    from src.analysis import get_vector_var

    potential = get_vector_var("potential_output", results.trace)
    potential.index = results.obs_index

    # r* trend from potential growth
    r_star = potential.diff(4).dropna().quantile(0.5, axis=1)
    x = np.arange(len(r_star))
    slope, intercept, *_ = stats.linregress(x, r_star.values)
    trend = pd.Series(intercept + slope * x, index=r_star.index)

    # Neutral = trend r* + πᵗ
    neutral = trend + pi_target
    neutral.name = "Nominal Neutral Rate"

    # Convert to monthly
    neutral.index = neutral.index.to_timestamp(how='end').to_period('M')

    # Plot
    cash_rate_monthly.name = "RBA Cash Rate"
    ax = mg.line_plot(neutral, color="darkorange", width=2, annotate=True)
    ax = mg.line_plot(cash_rate_monthly, ax=ax, color="darkblue", width=1,
                      drawstyle="steps-post", annotate=True)

    mg.finalise_plot(
        ax,
        title="Neutral Interest Rate vs RBA Cash Rate",
        ylabel="Per cent per annum",
        legend={"loc": "upper right", "fontsize": "x-small"},
        lfooter=f"Australia. Neutral rate = trend r* + πᵗ (where πᵗ = {pi_target}%).",
        rfooter="Equilibrium rate when output gap = 0 and U = NAIRU: i = r* + πᵗ",
        rheader=RFOOTER_OUTPUT,
        y0=True,
        show=show,
    )


def plot_obs_grid(obs: dict[str, np.ndarray], obs_index: pd.PeriodIndex, show: bool = False) -> None:
    """Plot all observation variables in a grid for quick visual inspection."""
    import math
    import matplotlib.pyplot as plt
    import mgplot as mg

    n_vars = len(obs)
    n_cols = 4
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.0, 2.5 * n_rows))
    axes = axes.flatten()

    last_used = 0
    for i, (name, values) in enumerate(obs.items()):
        ax = axes[i]
        series = pd.Series(values, index=obs_index, name=name)
        mg.line_plot(series, ax=ax, width=1)
        mg.finalise_plot(
            ax,
            title=name,
            y0=True,
            dont_save=True,
            dont_close=True,
        )
        last_used = i

    # Hide unused subplots
    for j in range(last_used + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()

    from pathlib import Path
    chart_dir = Path(__file__).parent.parent.parent / "charts" / "nairu_output_gap"
    chart_dir.mkdir(parents=True, exist_ok=True)
    mg.set_chart_dir(str(chart_dir))

    # Finalise last used axes - title becomes filename, suptitle for figure heading
    mg.finalise_plot(
        axes[last_used],
        title="Model Input Variables",
        figsize=(14.0, 2.5 * n_rows),
        show=show,
    )


def test_theoretical_expectations(trace: az.InferenceData) -> pd.DataFrame:
    """Test whether parameters match theoretical expectations.

    For parameters expected to equal a value (α≈0.3), we test:
        - Probability that parameter differs from expected value
        - Whether the expected value falls within the 90% HDI

    For parameters expected to have a sign (β<0, γ<0), we test:
        - Probability that parameter has the expected sign
    """
    from src.analysis import get_scalar_var

    results = []

    # Define tests: (parameter, expected_value or 'negative'/'positive'/(low,high), description)
    tests = [
        ('alpha_capital', (0.20, 0.35), 'Capital share ∈ (0.20, 0.35)'),
        ('beta_okun', 'negative', 'Okun coefficient < 0'),
        ('gamma_pi', 'negative', 'Phillips curve slope < 0'),
        ('gamma_wg', 'negative', 'Wage Phillips curve slope < 0'),
        ('beta_is', 'positive', 'IS interest rate effect > 0'),
        ('rho_is', 'between_0_1', 'IS persistence ∈ (0,1)'),
    ]

    for param, expected, description in tests:
        try:
            samples = get_scalar_var(param, trace).values
        except KeyError:
            # Parameter not in model (e.g., IS equation not included)
            continue

        median = np.median(samples)
        hdi_90 = az.hdi(samples, hdi_prob=0.90)

        if isinstance(expected, tuple):
            # Test for value within range (low, high)
            low, high = expected
            prob_in_range = np.mean((samples >= low) & (samples <= high))

            results.append({
                'Parameter': param,
                'Hypothesis': description,
                'Median': f'{median:.3f}',
                '90% HDI': f'[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]',
                'Expected in HDI': '-',
                f'P({low} ≤ θ ≤ {high})': f'{prob_in_range:.1%}',
                'Result': 'PASS' if prob_in_range > 0.90 else ('WEAK' if prob_in_range > 0.50 else 'FAIL')
            })
        elif isinstance(expected, (int, float)):
            # Test for equality to expected value
            in_hdi = hdi_90[0] <= expected <= hdi_90[1]
            prob_above = np.mean(samples > expected)

            results.append({
                'Parameter': param,
                'Hypothesis': description,
                'Median': f'{median:.3f}',
                '90% HDI': f'[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]',
                'Expected in HDI': '✓' if in_hdi else '✗',
                'P(θ > expected)': f'{prob_above:.1%}',
                'Result': 'PASS' if in_hdi else 'FAIL'
            })
        elif expected == 'between_0_1':
            # Test for value between 0 and 1 (stable persistence)
            prob_valid = np.mean((samples > 0) & (samples < 1))

            results.append({
                'Parameter': param,
                'Hypothesis': description,
                'Median': f'{median:.3f}',
                '90% HDI': f'[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]',
                'Expected in HDI': '-',
                'P(0 < θ < 1)': f'{prob_valid:.1%}',
                'Result': 'PASS' if prob_valid > 0.99 else ('WEAK' if prob_valid > 0.90 else 'FAIL')
            })
        else:
            # Test for sign
            if expected == 'negative':
                prob_correct = np.mean(samples < 0)
            else:  # positive
                prob_correct = np.mean(samples > 0)

            results.append({
                'Parameter': param,
                'Hypothesis': description,
                'Median': f'{median:.3f}',
                '90% HDI': f'[{hdi_90[0]:.3f}, {hdi_90[1]:.3f}]',
                'Expected in HDI': '-',
                'P(correct sign)': f'{prob_correct:.1%}',
                'Result': 'PASS' if prob_correct > 0.99 else ('WEAK' if prob_correct > 0.90 else 'FAIL')
            })

    df = pd.DataFrame(results)
    return df


def plot_all(
    results: NAIRUResults,
    inflation_annual: pd.Series | None = None,
    cash_rate_monthly: pd.Series | None = None,
    show: bool = False,
) -> None:
    """Generate all standard plots."""
    plot_nairu(results, show=show)
    plot_unemployment_gap(results, show=show)
    plot_output_gap(results, show=show)
    plot_gdp_vs_potential(results, show=show)
    plot_potential_growth(results, show=show)
    if cash_rate_monthly is not None and inflation_annual is not None:
        plot_taylor_rule(results, inflation_annual, cash_rate_monthly, show=show)
        plot_equilibrium_rates(results, cash_rate_monthly, show=show)


# --- CLI Entry Point ---


def main(verbose: bool = False) -> None:
    """Run the full NAIRU + Output Gap estimation pipeline."""
    from pathlib import Path
    import mgplot as mg

    # Set output directory for charts
    chart_dir = Path(__file__).parent.parent.parent / "charts" / "nairu_output_gap"
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    print("Running NAIRU + Output Gap model...\n")

    # Sampling configuration (matches notebook)
    config = SamplerConfig(
        draws=10_000,
        tune=3_500,
        chains=5,
        cores=5,
        target_accept=0.90,
    )

    # Build observations
    obs, obs_index = build_observations()

    # Plot observation grid
    plot_obs_grid(obs, obs_index)

    # Build and sample model
    model = build_model(obs)

    print("\nSampling...")
    trace = sample_model(model, config)

    # Create results container
    results = NAIRUResults(
        trace=trace,
        obs=obs,
        obs_index=obs_index,
        model=model,
    )

    # Diagnostics
    check_model_diagnostics(results.trace)

    # Check for zero coefficients
    zero_check = check_for_zero_coeffs(
        results.trace,
        critical_params=["gamma_pi", "gamma_wg", "beta_okun"]
    )
    if verbose:
        print(zero_check.T)

    # Plot scalar posteriors (bar and KDE)
    plot_posteriors_bar(
        results.trace,
        model_name=MODEL_NAME,
        show=False,
    )
    plot_posteriors_kde(
        results.trace,
        model_name=MODEL_NAME,
        show=False,
    )

    # Posterior predictive checks and residual analysis
    obs_vars = {
        "okun_law": obs["ΔU"],
        "observed_price_inflation": obs["π"],
        "observed_wage_growth": obs["Δulc"],
    }
    var_labels = {
        "okun_law": "Change in Unemployment (pp)",
        "observed_price_inflation": "Quarterly Inflation (%)",
        "observed_wage_growth": "Unit Labour Cost Growth (%)",
    }

    ppc_data = posterior_predictive_checks(
        trace=results.trace,
        model=model,
        obs_vars=obs_vars,
        obs_index=obs_index,
        var_labels=var_labels,
        model_name=MODEL_NAME,
        rfooter=RFOOTER_OUTPUT,
        show=False,
    )

    residual_autocorrelation_analysis(
        ppc=ppc_data,
        obs_vars=obs_vars,
        obs_index=obs_index,
        var_labels=var_labels,
        model_name=MODEL_NAME,
        rfooter=RFOOTER_OUTPUT,
        show=False,
    )

    # Theoretical expectations tests
    hypothesis_results = test_theoretical_expectations(results.trace)
    print(hypothesis_results.to_string(index=False))

    # Print summary
    if verbose:
        print("\nRecent NAIRU estimates:")
        nairu = results.nairu_median()
        U = pd.Series(results.obs["U"], index=results.obs_index)
        summary = pd.DataFrame({
            "NAIRU": nairu,
            "U": U,
            "U_gap": U - nairu,
        })
        print(summary.tail(8).round(2))

        print("\nRecent output gap:")
        print(results.output_gap().tail(8).round(2))

    # Get cash rate and inflation data for Taylor rule plots
    cash_rate_monthly = get_cash_rate_data()["cash_rate_monthly"]
    π4 = pd.Series(results.obs["π4"], index=results.obs_index)

    # Generate all plots
    plot_all(
        results,
        inflation_annual=π4,
        cash_rate_monthly=cash_rate_monthly,
        show=False,
    )

    print(f"\nCharts saved to: {chart_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NAIRU + Output Gap model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()
    main(verbose=args.verbose)
