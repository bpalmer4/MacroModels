"""GDP nowcasting model using a Dynamic Factor Model (DFM).

Extracts common factors from a mixed-frequency panel of monthly and quarterly
indicators using the Kalman filter (statsmodels DynamicFactorMQ). The ragged
edge — where faster indicators have more recent observations than slower ones —
is handled natively by the state-space framework, eliminating the need for
SARIMA completion.

Based on:
    - Bańbura, Giannone, Reichlin (2011): "Nowcasting"
    - Bańbura, Modugno (2014): "Maximum likelihood estimation of factor models
      on datasets with arbitrary pattern of missing data"
    - Bok et al. (2017): "Macroeconomic Nowcasting and Forecasting with Big Data"
      (NY Fed Staff Nowcast)

Indicator panel:
    Monthly:
        1. Retail turnover (ABS 5682.0) — consumption proxy
        2. Building approvals (ABS 8731.0) — investment proxy
        3. Hours worked (ABS 6202.0) — labour input
        4. Employment (ABS 6202.0) — labour demand
        5. Goods trade balance (ABS 5368.0) — external sector
        6. NAB business conditions (RBA H3) — survey/soft data
        7. Westpac-MI consumer sentiment (RBA H3) — household soft data

    Quarterly (entered as quarterly series in the mixed-frequency model):
        7. GDP growth (ABS 5206.0) — target variable
        8. CPI trimmed mean (ABS 6401.0) — prices
        9. WPI growth (ABS 6345.0) — wages
       10. Company profits growth (ABS 5676.0) — business activity
       11. Construction work done growth (ABS 8755.0) — investment
       12. Private capex growth (ABS 5625.0) — investment

Usage:
    # Live nowcast
    uv run python -m src.models.gdp_nowcast_dfm.model

    # Programmatic use
    from src.models.gdp_nowcast_dfm.model import nowcast, DataAvailability
    result = nowcast(target_quarter=pd.Period("2025Q4", "Q-DEC"))
"""

import logging
import warnings
from dataclasses import dataclass, field

import mgplot as mg
import numpy as np
import pandas as pd
import readabs as ra
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

from src.data import (
    get_building_approvals_monthly,
    get_goods_balance_monthly,
    get_hours_worked_monthly,
    get_retail_turnover_monthly,
    get_trimmed_mean_qrtly,
)
from src.data.abs_loader import load_series
from src.data.business_indicators import (
    get_company_profits_growth_qrtly,
)
from src.data.capex import get_total_capex_growth_qrtly
from src.data.construction import get_total_construction_growth_qrtly
from src.data.dataseries import DataSeries
from src.data.gdp import get_gdp
from src.data.series_specs import EMPLOYMENT_PERSONS
from src.data.surveys import (
    get_consumer_sentiment_monthly,
    get_nab_business_conditions_monthly,
)
from src.data.wpi import get_wpi_growth_qrtly

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

# --- Constants ---

CHART_DIR = "./charts/GDP-Nowcast-DFM/"
SHOW = False

# Number of factors and factor VAR order
N_FACTORS = 2
FACTOR_ORDER = 2

# Minimum observations for model estimation
MIN_OBS = 40

# Sample start: truncate older data to avoid structural breaks
# (e.g. pre-inflation-targeting era, pre-floating dollar)
SAMPLE_START = pd.Period("1990Q1", freq="Q-DEC")

# COVID masking — disabled. Tried 2020Q2-2021Q4 (7 quarters) and 2020Q2-2020Q4
# (3 quarters). Both improved RMSE and bias but at the cost of crushing the
# correlation between nowcasts and actual GDP (from +0.64 down to 0.009 / -0.26),
# i.e. the RMSE improvement came from regressing toward the mean rather than
# from better predictions. The DFM keeps more signal when COVID quarters are
# left in the panel, even though they introduce some bias.
COVID_MASK_START: pd.Period | None = None
COVID_MASK_END: pd.Period | None = None

# Publication lags (months after reference month) — same as bridge model
PUBLICATION_LAGS = {
    "retail": 1,
    "building_approvals": 2,
    "hours_worked": 1,
    "employment": 1,
    "goods_balance": 2,
    "nab_conditions": 1,
    "consumer_sentiment": 1,  # Westpac-MI, mid-month for the same month
}


# --- Data availability ---


@dataclass
class DataAvailability:
    """Specifies which data is available for a nowcast.

    Monthly indicators: the last reference month available (None = not available).
    Quarterly indicators: whether published for the target quarter.
    """

    employment: pd.Period | None = None
    hours_worked: pd.Period | None = None
    retail: pd.Period | None = None
    building_approvals: pd.Period | None = None
    goods_balance: pd.Period | None = None
    nab_conditions: pd.Period | None = None
    consumer_sentiment: pd.Period | None = None
    cpi_quarterly: bool = False
    wpi: bool = False
    business_profits: bool = False
    construction: bool = False
    capex: bool = False

    @classmethod
    def from_live_data(
        cls,
        monthly_indicators: dict[str, DataSeries],
        quarterly_indicators: dict[str, pd.Series],
        target_quarter: pd.Period,
    ) -> "DataAvailability":
        """Detect availability from actual data."""
        avail = cls()
        for name, ds in monthly_indicators.items():
            last = ds.data.dropna().index[-1]
            setattr(avail, name, last)
        for q_name, series in quarterly_indicators.items():
            available = target_quarter in series.index and pd.notna(series.loc[target_quarter])
            setattr(avail, q_name, available)
        return avail

    @classmethod
    def at_t_minus_3m(cls, target_quarter: pd.Period) -> "DataAvailability":
        """~0 months of target quarter data available."""
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        pre_q = pd.Period(year=q_year, month=q_start_month, freq="M") - 1
        avail = cls()
        for name, lag in PUBLICATION_LAGS.items():
            setattr(avail, name, pre_q - lag)
        return avail

    @classmethod
    def at_t_minus_2m(cls, target_quarter: pd.Period) -> "DataAvailability":
        """~1 month of target quarter data available."""
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        month_1 = pd.Period(year=q_year, month=q_start_month, freq="M")
        avail = cls()
        for name, lag in PUBLICATION_LAGS.items():
            setattr(avail, name, month_1 - lag + 1)
        return avail

    @classmethod
    def at_t_minus_1m(cls, target_quarter: pd.Period) -> "DataAvailability":
        """~2 months of target quarter data available."""
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        month_2 = pd.Period(year=q_year, month=q_start_month + 1, freq="M")
        avail = cls()
        for name, lag in PUBLICATION_LAGS.items():
            setattr(avail, name, month_2 - lag + 1)
        return avail

    @classmethod
    def at_t_minus_0(cls, target_quarter: pd.Period) -> "DataAvailability":
        """All data available, just before GDP publication."""
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        month_3 = pd.Period(year=q_year, month=q_start_month + 2, freq="M")
        return cls(
            employment=month_3,
            hours_worked=month_3,
            retail=month_3,
            building_approvals=month_3,
            goods_balance=month_3,
            nab_conditions=month_3,
            consumer_sentiment=month_3,
            cpi_quarterly=True,
            wpi=True,
            business_profits=True,
            construction=True,
            capex=True,
        )

    def monthly_status(self, target_quarter: pd.Period) -> dict[str, str]:
        """Report data availability for each monthly indicator."""
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        months = [
            pd.Period(year=q_year, month=q_start_month + i, freq="M")
            for i in range(3)
        ]
        status = {}
        for name in PUBLICATION_LAGS:
            cutoff = getattr(self, name)
            if cutoff is None:
                status[name] = "0/3 months"
            else:
                n = sum(1 for m in months if m <= cutoff)
                status[name] = f"{n}/3 months"
        return status


# --- Result structures ---


@dataclass
class NowcastResult:
    """Combined nowcast result."""

    target_quarter: pd.Period
    gdp_qoq: float
    gdp_tty: float
    gdp_qoq_70: tuple[float, float]
    gdp_qoq_90: tuple[float, float]
    gdp_tty_70: tuple[float, float]
    gdp_tty_90: tuple[float, float]
    n_factors: int
    factor_order: int
    n_monthly: int
    n_quarterly: int
    data_status: dict[str, str]
    availability: DataAvailability | None = None
    factor_loadings: pd.DataFrame | None = None
    smoothed_factors: pd.DataFrame | None = None


# --- Data loading ---


def _load_monthly_indicators() -> dict[str, DataSeries]:
    """Load all monthly indicator series (full history)."""
    return {
        "retail": get_retail_turnover_monthly(),
        "building_approvals": get_building_approvals_monthly(),
        "hours_worked": get_hours_worked_monthly(),
        "employment": load_series(EMPLOYMENT_PERSONS),
        "goods_balance": get_goods_balance_monthly(),
        "nab_conditions": get_nab_business_conditions_monthly(),
        "consumer_sentiment": get_consumer_sentiment_monthly(),
    }


def _load_quarterly_indicators() -> dict[str, pd.Series]:
    """Load all quarterly indicator series (full history)."""
    indicators = {}
    loaders = {
        "cpi_quarterly": get_trimmed_mean_qrtly,
        "wpi": get_wpi_growth_qrtly,
        "business_profits": get_company_profits_growth_qrtly,
        "construction": get_total_construction_growth_qrtly,
        "capex": get_total_capex_growth_qrtly,
    }
    for name, loader in loaders.items():
        try:
            ds = loader()
            indicators[name] = ds.data.dropna()
        except (ValueError, KeyError, OSError):
            logger.warning("Failed to load quarterly indicator '%s'", name)
    return indicators


# --- Panel construction ---


def _monthly_to_growth(series: pd.Series) -> pd.Series:
    """Convert monthly level series to log-difference growth × 100."""
    s = series.dropna()
    if (s <= 0).any():
        # For series that can be negative (e.g. trade balance), use simple difference
        return s.diff(1)
    return np.log(s).diff(1) * 100


def _truncate_monthly(series: pd.Series, cutoff: pd.Period | None) -> pd.Series:
    """Truncate a monthly series to data available through cutoff period."""
    if cutoff is None:
        return pd.Series(dtype=float)
    return series.loc[series.index <= cutoff].copy()


def _build_monthly_panel(
    monthly_indicators: dict[str, DataSeries],
    availability: DataAvailability,
    target_quarter: pd.Period | None = None,
) -> pd.DataFrame:
    """Build a monthly-frequency DataFrame from indicator series.

    Converts levels to growth rates and truncates to available data.
    The ragged edge (different end dates) is left as NaN — DynamicFactorMQ
    handles missing data natively via the Kalman filter.
    """
    series_dict = {}

    sample_start_month = pd.Period(year=SAMPLE_START.year, month=1, freq="M")

    for name, ds in monthly_indicators.items():
        cutoff = getattr(availability, name, None)
        raw = ds.data.dropna()
        truncated = _truncate_monthly(raw, cutoff)

        if len(truncated) < MIN_OBS:
            logger.info("Monthly '%s': insufficient data (%d obs), skipping", name, len(truncated))
            continue

        growth = _monthly_to_growth(truncated)
        growth = growth.loc[growth.index >= sample_start_month]
        series_dict[name] = growth

    if not series_dict:
        msg = "No monthly indicators available"
        raise RuntimeError(msg)

    panel = pd.DataFrame(series_dict)
    if not isinstance(panel.index, pd.PeriodIndex):
        panel.index = panel.index.to_period("M")

    # Extend the panel index to include month 3 of the target quarter so the
    # Kalman filter forecasts the missing values forward.
    if target_quarter is not None:
        q_end_month = pd.Period(
            year=target_quarter.year,
            month=(target_quarter.quarter - 1) * 3 + 3,
            freq="M",
        )
        if panel.index[-1] < q_end_month:
            full_idx = pd.period_range(panel.index[0], q_end_month, freq="M")
            panel = panel.reindex(full_idx)

    # COVID masking (if enabled)
    if COVID_MASK_START is not None and COVID_MASK_END is not None:
        mask_start = pd.Period(year=COVID_MASK_START.year,
                               month=(COVID_MASK_START.quarter - 1) * 3 + 1, freq="M")
        mask_end = pd.Period(year=COVID_MASK_END.year,
                             month=(COVID_MASK_END.quarter - 1) * 3 + 3, freq="M")
        covid_mask = (panel.index >= mask_start) & (panel.index <= mask_end)
        panel.loc[covid_mask, :] = np.nan

    return panel


def _build_quarterly_panel(
    gdp_growth: pd.Series,
    quarterly_indicators: dict[str, pd.Series],
    target_quarter: pd.Period,
    availability: DataAvailability,
) -> pd.DataFrame:
    """Build a quarterly-frequency DataFrame from GDP + quarterly indicators.

    GDP is always included (truncated to exclude target quarter).
    Other quarterly indicators are included only if available for the target quarter.
    """
    series_dict = {"gdp_growth": gdp_growth.loc[gdp_growth.index < target_quarter]}

    for name, series in quarterly_indicators.items():
        available = getattr(availability, name, False)
        if available:
            series_dict[name] = series.loc[series.index <= target_quarter]
        else:
            series_dict[name] = series.loc[series.index < target_quarter]

    panel = pd.DataFrame(series_dict)
    if not isinstance(panel.index, pd.PeriodIndex):
        panel.index = panel.index.to_period("Q-DEC")

    panel = panel.loc[panel.index >= SAMPLE_START]

    # Extend the index to include the target quarter (with NaN for gdp_growth
    # so the Kalman filter produces the nowcast)
    if panel.index[-1] < target_quarter:
        full_idx = pd.period_range(panel.index[0], target_quarter, freq="Q-DEC")
        panel = panel.reindex(full_idx)

    # COVID masking (if enabled)
    if COVID_MASK_START is not None and COVID_MASK_END is not None:
        covid_mask = (panel.index >= COVID_MASK_START) & (panel.index <= COVID_MASK_END)
        panel.loc[covid_mask, :] = np.nan

    return panel


# --- DFM estimation ---


def _fit_dfm(
    monthly_panel: pd.DataFrame,
    quarterly_panel: pd.DataFrame,
    n_factors: int = N_FACTORS,
    factor_order: int = FACTOR_ORDER,
) -> DynamicFactorMQ:
    """Fit the Dynamic Factor Model on the mixed-frequency panel.

    Uses the EM algorithm (Bańbura & Modugno 2014) for parameter estimation,
    which handles arbitrary patterns of missing data.
    """
    model = DynamicFactorMQ(
        endog=monthly_panel,
        endog_quarterly=quarterly_panel,
        factors=n_factors,
        factor_orders=factor_order,
        idiosyncratic_ar1=True,
        standardize=True,
    )

    result = model.fit_em(maxiter=500, tolerance=1e-6, disp=False)

    return result


def _extract_nowcast(
    result,
    target_quarter: pd.Period,
    gdp: pd.Series,
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """Extract GDP growth nowcast and prediction intervals from the fitted DFM.

    The Kalman smoother produces filtered estimates for all variables including
    GDP growth for the target quarter. Prediction uncertainty comes from the
    state covariance matrix.
    """
    # Get the smoothed predictions for gdp_growth
    predicted = result.get_prediction()
    pred_mean = predicted.predicted_mean
    pred_se = predicted.se_mean

    gdp_col = "gdp_growth"
    if gdp_col not in pred_mean.columns:
        gdp_cols = [c for c in pred_mean.columns if "gdp" in c.lower()]
        if not gdp_cols:
            msg = f"GDP growth column not found. Columns: {list(pred_mean.columns)}"
            raise RuntimeError(msg)
        gdp_col = gdp_cols[0]

    # DynamicFactorMQ's prediction index labels are unreliable (statsmodels quirk
    # with PeriodIndex). Use positional indexing: the LAST observation in the
    # gdp_growth predictions corresponds to the last month of the model's data,
    # which is the third month of the target quarter.
    gdp_preds = pred_mean[gdp_col].dropna()
    gdp_ses = pred_se[gdp_col].dropna()

    if len(gdp_preds) == 0:
        msg = "No GDP growth predictions available"
        raise RuntimeError(msg)

    nowcast_qoq = float(gdp_preds.iloc[-1])
    se = float(gdp_ses.iloc[-1]) if len(gdp_ses) > 0 else np.nan

    # Compute prediction intervals
    if np.isfinite(se) and se > 0:
        ci_70 = (nowcast_qoq - 1.04 * se, nowcast_qoq + 1.04 * se)
        ci_90 = (nowcast_qoq - 1.645 * se, nowcast_qoq + 1.645 * se)
    else:
        # Fallback: use historical GDP growth volatility
        hist_std = gdp.pct_change().dropna().std() * 100
        ci_70 = (nowcast_qoq - 1.04 * hist_std, nowcast_qoq + 1.04 * hist_std)
        ci_90 = (nowcast_qoq - 1.645 * hist_std, nowcast_qoq + 1.645 * hist_std)

    return nowcast_qoq, ci_70, ci_90


# --- GDP helpers ---


def _detect_target_quarter(gdp: pd.Series) -> pd.Period:
    """Auto-detect the next unpublished GDP quarter."""
    last_published = gdp.dropna().index[-1]
    return last_published + 1


def _compute_gdp_growth(gdp: pd.Series) -> pd.Series:
    """Compute Q/Q GDP growth as log difference × 100."""
    return np.log(gdp).diff(1) * 100


def _compute_tty(gdp_growth_qoq: float, gdp: pd.Series, target_quarter: pd.Period) -> float:
    """Compute through-the-year growth from Q/Q nowcast."""
    last_gdp = gdp.iloc[-1]
    projected_gdp = last_gdp * np.exp(gdp_growth_qoq / 100)
    q_minus_4 = target_quarter - 4
    gdp_4q_ago = gdp.loc[q_minus_4] if q_minus_4 in gdp.index else gdp.iloc[-4]
    return (projected_gdp / gdp_4q_ago - 1) * 100


# --- Core nowcast ---


def nowcast(
    target_quarter: pd.Period | None = None,
    availability: DataAvailability | None = None,
    gdp: pd.Series | None = None,
    monthly_indicators: dict[str, DataSeries] | None = None,
    quarterly_indicators: dict[str, pd.Series] | None = None,
    n_factors: int = N_FACTORS,
    factor_order: int = FACTOR_ORDER,
    quiet: bool = False,
) -> NowcastResult:
    """Run the DFM GDP nowcast.

    Args:
        target_quarter: The quarter to nowcast. If None, auto-detected.
        availability: Data availability specification. If None, detected from data.
        gdp: GDP series (CVM, SA). If None, fetched from ABS.
        monthly_indicators: Pre-loaded monthly indicators. If None, fetched.
        quarterly_indicators: Pre-loaded quarterly indicators. If None, fetched.
        n_factors: Number of common factors to extract.
        factor_order: VAR order for factor dynamics.
        quiet: If True, suppress terminal output.

    Returns:
        NowcastResult with point estimates, intervals, and diagnostics.
    """
    # 1. Load data
    if gdp is None:
        gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()

    if target_quarter is None:
        target_quarter = _detect_target_quarter(gdp)

    gdp_truncated = gdp.loc[gdp.index < target_quarter]
    gdp_growth = _compute_gdp_growth(gdp_truncated)

    if monthly_indicators is None:
        monthly_indicators = _load_monthly_indicators()
    if quarterly_indicators is None:
        quarterly_indicators = _load_quarterly_indicators()

    # 2. Determine availability
    if availability is None:
        availability = DataAvailability.from_live_data(
            monthly_indicators, quarterly_indicators, target_quarter,
        )

    data_status = availability.monthly_status(target_quarter)

    if not quiet:
        print(f"Target quarter: {target_quarter}")
        print(f"Last published GDP: {gdp_truncated.index[-1]}")

    # 3. Build panels
    monthly_panel = _build_monthly_panel(monthly_indicators, availability, target_quarter)
    quarterly_panel = _build_quarterly_panel(
        gdp_growth, quarterly_indicators, target_quarter, availability,
    )

    if not quiet:
        print(f"Monthly panel: {len(monthly_panel.columns)} series, "
              f"{len(monthly_panel)} months")
        print(f"Quarterly panel: {len(quarterly_panel.columns)} series")
        print(f"Factors: {n_factors}, VAR order: {factor_order}")

    # 4. Fit DFM
    dfm_result = _fit_dfm(monthly_panel, quarterly_panel, n_factors, factor_order)

    # 5. Extract nowcast
    nowcast_qoq, ci_70_qoq, ci_90_qoq = _extract_nowcast(
        dfm_result, target_quarter, gdp_truncated,
    )

    # 6. Compute TTY
    tty = _compute_tty(nowcast_qoq, gdp_truncated, target_quarter)
    tty_70 = (
        _compute_tty(ci_70_qoq[0], gdp_truncated, target_quarter),
        _compute_tty(ci_70_qoq[1], gdp_truncated, target_quarter),
    )
    tty_90 = (
        _compute_tty(ci_90_qoq[0], gdp_truncated, target_quarter),
        _compute_tty(ci_90_qoq[1], gdp_truncated, target_quarter),
    )

    # 7. Extract factor diagnostics
    try:
        smoothed = dfm_result.factors.smoothed
        # `smoothed` is already a DataFrame with statsmodels' (unreliable) index.
        # Replace its index positionally with the panel's PeriodIndex and rename columns.
        factor_names = [f"Factor {i+1}" for i in range(smoothed.shape[1])]
        smoothed_factors = pd.DataFrame(
            smoothed.to_numpy(),
            index=monthly_panel.index[:len(smoothed)],
            columns=factor_names,
        )
    except (AttributeError, ValueError):
        smoothed_factors = None

    try:
        loadings = dfm_result.params
        loading_idx = [p for p in loadings.index if "loading" in p.lower()]
        factor_loadings = pd.DataFrame({"loading": loadings[loading_idx]}) if loading_idx else None
    except (AttributeError, ValueError):
        factor_loadings = None

    # 8. Assemble result
    result = NowcastResult(
        target_quarter=target_quarter,
        gdp_qoq=nowcast_qoq,
        gdp_tty=tty,
        gdp_qoq_70=ci_70_qoq,
        gdp_qoq_90=ci_90_qoq,
        gdp_tty_70=tty_70,
        gdp_tty_90=tty_90,
        n_factors=n_factors,
        factor_order=factor_order,
        n_monthly=len(monthly_panel.columns),
        n_quarterly=len(quarterly_panel.columns),
        data_status=data_status,
        availability=availability,
        factor_loadings=factor_loadings,
        smoothed_factors=smoothed_factors,
    )

    if not quiet:
        _print_summary(result)

    return result


# --- Output ---


def _print_summary(result: NowcastResult) -> None:
    """Print nowcast summary to terminal."""
    print("\n" + "=" * 70)
    print(f"  GDP NOWCAST (DFM): {result.target_quarter}")
    print("=" * 70)

    print(f"\n  Q/Q growth:   {result.gdp_qoq:+.2f}%")
    print(f"    70% CI:     [{result.gdp_qoq_70[0]:+.2f}%, {result.gdp_qoq_70[1]:+.2f}%]")
    print(f"    90% CI:     [{result.gdp_qoq_90[0]:+.2f}%, {result.gdp_qoq_90[1]:+.2f}%]")

    print(f"\n  TTY growth:   {result.gdp_tty:+.2f}%")
    print(f"    70% CI:     [{result.gdp_tty_70[0]:+.2f}%, {result.gdp_tty_70[1]:+.2f}%]")
    print(f"    90% CI:     [{result.gdp_tty_90[0]:+.2f}%, {result.gdp_tty_90[1]:+.2f}%]")

    print(f"\n  Model: {result.n_factors} factors, VAR({result.factor_order})")
    print(f"  Monthly indicators: {result.n_monthly}")
    print(f"  Quarterly indicators: {result.n_quarterly}")

    print("\n  Data availability (monthly indicators):")
    for name, status in result.data_status.items():
        print(f"    {name:<25} {status}")

    print("\n" + "=" * 70)


# --- Plotting ---


def _plot_fan_chart(
    hist: pd.Series,
    nowcast_value: float,
    ci_70: tuple[float, float],
    ci_90: tuple[float, float],
    target_quarter: pd.Period,
    title: str,
    lfooter_value: str,
) -> None:
    """Plot a nowcast fan chart using mgplot layering."""
    recent = hist.iloc[-20:]

    nowcast_idx = pd.PeriodIndex([recent.index[-1], target_quarter], freq="Q-DEC")
    nowcast_line = pd.Series([recent.iloc[-1], nowcast_value], index=nowcast_idx)

    band_90 = pd.DataFrame({
        "lower": [recent.iloc[-1], ci_90[0]],
        "upper": [recent.iloc[-1], ci_90[1]],
    }, index=nowcast_idx)

    band_70 = pd.DataFrame({
        "lower": [recent.iloc[-1], ci_70[0]],
        "upper": [recent.iloc[-1], ci_70[1]],
    }, index=nowcast_idx)

    recent = recent.rename("GDP growth")
    nowcast_line = nowcast_line.rename("Nowcast")

    ax = mg.fill_between_plot(band_90, color="red", alpha=0.1, label="90% CI")
    mg.fill_between_plot(band_70, ax=ax, color="red", alpha=0.2, label="70% CI")
    mg.line_plot(recent, ax=ax, color=["navy"], width=2)
    mg.line_plot(nowcast_line, ax=ax, color=["red"], width=2, style="--", annotate=True, rounding=2)

    mg.finalise_plot(
        ax,
        title=title,
        ylabel="Per cent",
        rfooter="Source: ABS 5206.0",
        lfooter=f"Australia. DFM nowcast: {lfooter_value}. {pd.Timestamp.now().strftime('%d %b %Y')}. ",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        show=SHOW,
    )


def _plot_nowcast(result: NowcastResult, gdp_growth_hist: pd.Series) -> None:
    """Plot Q/Q nowcast vs GDP history with fan chart."""
    _plot_fan_chart(
        hist=gdp_growth_hist.dropna(),
        nowcast_value=result.gdp_qoq,
        ci_70=result.gdp_qoq_70,
        ci_90=result.gdp_qoq_90,
        target_quarter=result.target_quarter,
        title=f"GDP Growth Nowcast DFM (Q/Q): {result.target_quarter}",
        lfooter_value=f"{result.gdp_qoq:+.2f}%",
    )


def _plot_nowcast_tty(result: NowcastResult, gdp: pd.Series) -> None:
    """Plot TTY nowcast vs GDP TTY history with fan chart."""
    tty_hist = ((gdp / gdp.shift(4)) - 1) * 100
    _plot_fan_chart(
        hist=tty_hist.dropna(),
        nowcast_value=result.gdp_tty,
        ci_70=result.gdp_tty_70,
        ci_90=result.gdp_tty_90,
        target_quarter=result.target_quarter,
        title=f"GDP Growth Nowcast DFM (TTY): {result.target_quarter}",
        lfooter_value=f"{result.gdp_tty:+.2f}%",
    )


def _plot_growth(result: NowcastResult, gdp: pd.Series) -> None:
    """Plot GDP growth in bar (Q/Q) + line (TTY) format with nowcast appended."""
    gdp_growth = _compute_gdp_growth(gdp).dropna()
    tty = ((gdp / gdp.shift(4)) - 1) * 100

    gdp_growth.loc[result.target_quarter] = result.gdp_qoq
    tty.loc[result.target_quarter] = result.gdp_tty

    growth_df = pd.DataFrame({
        "Annual Growth": tty,
        "Quarterly Growth": gdp_growth,
    }).iloc[-20:]

    mg.growth_plot_finalise(
        growth_df,
        title=f"GDP Growth DFM Nowcast: {result.target_quarter}",
        ylabel="Per cent",
        rfooter="Source: ABS 5206.0",
        lfooter=f"Australia. DFM Q/Q: {result.gdp_qoq:+.2f}%, TTY: {result.gdp_tty:+.2f}%. "
                f"{pd.Timestamp.now().strftime('%d %b %Y')}. ",
        legend={"loc": "best", "fontsize": "x-small"},
        show=SHOW,
    )


def _plot_factors(result: NowcastResult) -> None:
    """Plot the smoothed common factors."""
    if result.smoothed_factors is None:
        return

    factors = result.smoothed_factors.iloc[-120:]  # Last 10 years of monthly data

    mg.line_plot_finalise(
        factors,
        title="DFM Common Factors",
        ylabel="Standardised units",
        rfooter="Source: ABS, RBA",
        lfooter=f"Australia. {result.n_factors} factors, VAR({result.factor_order}). "
                f"{pd.Timestamp.now().strftime('%d %b %Y')}. ",
        y0=True,
        width=2,
        legend={"loc": "best", "fontsize": "small"},
        show=SHOW,
    )


# --- Live entry point ---


def run_nowcast() -> NowcastResult:
    """Run the live GDP nowcast with auto-detected target and availability."""
    result = nowcast()

    gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()
    gdp_growth = _compute_gdp_growth(gdp)

    mg.set_chart_dir(CHART_DIR)
    mg.clear_chart_dir()
    _plot_nowcast(result, gdp_growth)
    _plot_nowcast_tty(result, gdp)
    _plot_growth(result, gdp)
    _plot_factors(result)

    return result


if __name__ == "__main__":
    run_nowcast()
