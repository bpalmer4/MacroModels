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
        1. Retail turnover, trimmed-mean-deflated (ABS 5682.0) — real consumption proxy
        2. Building approvals (ABS 8731.0) — investment proxy (counts)
        3. Hours worked (ABS 6202.0) — labour input
        4. Employment (ABS 6202.0) — labour demand
        5. Goods trade balance, trimmed-mean-deflated (ABS 5368.0) — external sector
        6. NAB business conditions (RBA H3) — survey/soft data
        7. Westpac-MI consumer sentiment (RBA H3) — household soft data

    Quarterly (entered as quarterly series in the mixed-frequency model):
        7. GDP growth (ABS 5206.0) — target variable
        8. CPI trimmed mean (ABS 6401.0) — prices
        9. WPI growth (ABS 6345.0) — wages
       10. Company profits growth, trimmed-mean-deflated (ABS 5676.0) — real business activity
       11. Construction work done growth (ABS 8755.0) — investment (CVM)
       12. Private capex growth (ABS 5625.0) — investment (CVM)
       13. Household spending CVM growth (ABS 5682.0 table 5682015) — real consumption,
           HFCE-analogue, published ~5 weeks after quarter-end

Usage:
    # Live nowcast
    uv run python -m src.models.gdp_nowcast_dfm.model

    # Programmatic use
    from src.models.gdp_nowcast_dfm.model import nowcast, DataAvailability
    result = nowcast(target_quarter=pd.Period("2025Q4", "Q-DEC"))
"""

import logging
import warnings
from dataclasses import dataclass
from functools import cache

import mgplot as mg
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

from src.data import (
    get_building_approvals_monthly,
    get_goods_balance_real_monthly,
    get_hourly_coe_growth_qrtly,
    get_hours_worked_monthly,
    get_household_spending_cvm_growth_qrtly,
    get_retail_turnover_real_monthly,
    get_trimmed_mean_qrtly,
    get_ulc_growth_qrtly,
)
from src.data.abs_loader import load_series
from src.data.business_indicators import (
    get_company_profits_real_growth_qrtly,
)
from src.data.capex import get_total_capex_growth_qrtly
from src.data.construction import get_total_construction_growth_qrtly
from src.data.dataseries import DataSeries
from src.data.gdp import get_gdp
from src.data.henderson import hma as henderson_hma
from src.data.productivity import get_labour_productivity_growth
from src.data.series_specs import EMPLOYMENT_PERSONS
from src.data.surveys import (
    get_consumer_sentiment_monthly,
    get_nab_business_conditions_monthly,
)
from src.data.wpi import get_wpi_growth_qrtly
from src.models.common.nowcast_charts import NowcastChartSpec, plot_nowcast_charts
from src.models.common.nowcast_core import (
    compute_gdp_growth,
    compute_tty,
    detect_target_quarter,
    print_qoq_tty_header,
    truncate_monthly,
)

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

# Productivity-adjusted labour input. Net the labour-productivity trend into the monthly
# employment & hours-worked growth, so a productivity slump discounts an otherwise-strong
# labour signal (output growth = labour-input growth + productivity growth). Kept ON as a
# BIAS CORRECTION: on the 2022Q1-2025Q4 backtest it cuts the T-0 bias +0.33 -> ~+0.10 while
# preserving correlation (~0.69 -> 0.65). The bias reduction is genuine — it survives at +0.12
# when re-tested with a low-look-ahead trailing-4QMA signal. NOTE: the RMSE "beats naive"
# result is largely LOOK-AHEAD — the centered HMA(13) over the full sample (productivity not
# truncated per target) effectively knows the slump in advance (RMSE 0.41 -> 0.26); on an
# honest trailing signal RMSE only reaches ~0.34, ABOVE naive 0.285. So this is a de-biaser,
# not a benchmark-beating accuracy gain. The live nowcast is real-time-sound (centered HMA for
# the historical fit, 4Q-trailing-avg carry-forward at the edge — see _productivity_adjust_labour);
# the look-ahead is only in the backtest *evaluation*. A vintage (per-target) backtest is TODO.
# Set PRODUCTIVITY_ADJUST_LABOUR = False to revert to the vanilla DFM.
PRODUCTIVITY_ADJUST_LABOUR = True
PRODUCTIVITY_ADJUST_COLS = ("employment", "hours_worked")


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
    household_spending: bool = False

    @classmethod
    def from_live_data(
        cls,
        monthly_indicators: dict[str, DataSeries],
        quarterly_indicators: dict[str, pd.Series],
        target_quarter: pd.Period,
    ) -> DataAvailability:
        """Detect availability from actual data."""
        avail = cls()
        # Strict T-0: the nowcast uses no data beyond the target quarter, so cap each
        # monthly cutoff at the target quarter's third month even when later (next-quarter)
        # observations exist. This keeps the live information set identical to the
        # backtest's T-0 (at_t_minus_0) and prevents the panel extending past the target.
        q3_month = pd.Period(
            year=target_quarter.year,
            month=(target_quarter.quarter - 1) * 3 + 3,
            freq="M",
        )
        for name, ds in monthly_indicators.items():
            last = ds.data.dropna().index[-1]
            setattr(avail, name, min(last, q3_month))
        for q_name, series in quarterly_indicators.items():
            available = target_quarter in series.index and pd.notna(series.loc[target_quarter])
            setattr(avail, q_name, available)
        return avail

    @classmethod
    def at_t_minus_3m(cls, target_quarter: pd.Period) -> DataAvailability:
        """~0 months of target quarter data available."""
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        pre_q = pd.Period(year=q_year, month=q_start_month, freq="M") - 1
        avail = cls()
        for name, lag in PUBLICATION_LAGS.items():
            setattr(avail, name, pre_q - lag)
        return avail

    @classmethod
    def at_t_minus_2m(cls, target_quarter: pd.Period) -> DataAvailability:
        """~1 month of target quarter data available."""
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        month_1 = pd.Period(year=q_year, month=q_start_month, freq="M")
        avail = cls()
        for name, lag in PUBLICATION_LAGS.items():
            setattr(avail, name, month_1 - lag + 1)
        return avail

    @classmethod
    def at_t_minus_1m(cls, target_quarter: pd.Period) -> DataAvailability:
        """~2 months of target quarter data available."""
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        month_2 = pd.Period(year=q_year, month=q_start_month + 1, freq="M")
        avail = cls()
        for name, lag in PUBLICATION_LAGS.items():
            setattr(avail, name, month_2 - lag + 1)
        return avail

    @classmethod
    def at_t_minus_0(cls, target_quarter: pd.Period) -> DataAvailability:
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
            household_spending=True,
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
    """Load all monthly indicator series (full history).

    Retail turnover and goods balance are trimmed-mean-deflated (real) — see
    src/data/retail_trade.py and src/data/goods_trade.py for deflation logic.
    """
    return {
        "retail": get_retail_turnover_real_monthly(),
        "building_approvals": get_building_approvals_monthly(),
        "hours_worked": get_hours_worked_monthly(),
        "employment": load_series(EMPLOYMENT_PERSONS),
        "goods_balance": get_goods_balance_real_monthly(),
        "nab_conditions": get_nab_business_conditions_monthly(),
        "consumer_sentiment": get_consumer_sentiment_monthly(),
    }


def _load_quarterly_indicators(target_quarter: pd.Period) -> dict[str, pd.Series]:
    """Load all quarterly indicator series (full history).

    Profits is trimmed-mean-deflated (real). Household spending CVM (5682.0
    table 5682015) starts 2014Q3 — short history but the DFM handles
    ragged-edge panels via missing-data EM.

    ``target_quarter`` is required: the household spending CVM table is fetched by
    quarter-end-month snapshot (ABS needs a specific quarter), so we ground the
    request to the target quarter's end month (e.g. 2026Q1 -> "mar-2026").
    """
    indicators = {}
    hs_month = target_quarter.asfreq("M", how="end").strftime("%b-%Y").lower()
    loaders = {
        "cpi_quarterly": get_trimmed_mean_qrtly,
        "wpi": get_wpi_growth_qrtly,
        "business_profits": get_company_profits_real_growth_qrtly,
        "construction": get_total_construction_growth_qrtly,
        "capex": get_total_capex_growth_qrtly,
        "household_spending": lambda: get_household_spending_cvm_growth_qrtly(hs_month),
    }
    for name, loader in loaders.items():
        try:
            ds = loader()
            series = ds.data.dropna()
            if len(series) > 0:
                indicators[name] = series
            else:
                logger.warning("Quarterly indicator '%s' returned empty series", name)
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


@cache
def _labour_productivity_trend_monthly() -> pd.Series:
    """HMA(13) labour-productivity trend as a monthly-rate series (PeriodIndex[M]).

    Labour productivity growth (Δhcoe − Δulc, from ULC = HCOE / LP) is quarterly; we
    smooth it with a Henderson MA(13) — as the bridge model does — then spread each
    quarter's Q/Q trend evenly across its three months (value / 3) so it is expressed as a
    monthly-rate growth that can be added directly to the monthly labour-input growth series.
    """
    ulc = get_ulc_growth_qrtly().data
    hcoe = get_hourly_coe_growth_qrtly().data
    lp = get_labour_productivity_growth(ulc, hcoe).data.dropna()
    lp_trend_q = henderson_hma(lp, 13).reindex(lp.index).dropna()

    monthly = {}
    for quarter, value in lp_trend_q.items():
        for offset in range(3):
            month = pd.Period(year=quarter.year, month=(quarter.quarter - 1) * 3 + 1 + offset, freq="M")
            monthly[month] = value / 3.0
    return pd.Series(monthly).sort_index()


def _productivity_adjust_labour(panel: pd.DataFrame) -> pd.DataFrame:
    """Net the labour-productivity trend into the labour-input columns (in place).

    A weak productivity trend discounts otherwise-strong employment/hours growth
    (output growth = labour-input growth + productivity growth). No-op if disabled.

    Carry-forward into the target quarter (whose months sit beyond the last quarterly
    productivity reading) uses a two-stage smoother:
      1. HMA(13) removes spikes from the raw productivity series (interior trend), then
      2. a 4-quarter TRAILING average of that HMA trend sets the projected value.
    The trailing average is causal (defined at the edge, no future data) so it sidesteps
    the HMA endpoint's asymmetric end-weights, which extrapolate recent momentum and read
    high (~+0.37/q vs a ~+0.21/q trailing mean at 2025Q4). Only the forward-filled target
    months are affected; historical months — and the backtest, where the trend extends past
    the target — are unchanged.
    """
    if not PRODUCTIVITY_ADJUST_LABOUR:
        return panel
    trend = _labour_productivity_trend_monthly()
    pt = trend.reindex(panel.index)
    pt = pt.fillna(trend.iloc[-12:].mean())  # 4Q trailing avg of the HMA trend (12 monthly obs)
    for col in PRODUCTIVITY_ADJUST_COLS:
        if col in panel.columns:
            panel[col] = panel[col] + pt
    return panel


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
        truncated = truncate_monthly(raw, cutoff)

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

    # Productivity-adjust the labour-input columns when enabled (see PRODUCTIVITY_ADJUST_LABOUR).
    return _productivity_adjust_labour(panel)


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

    return model.fit_em(maxiter=500, tolerance=1e-6, disp=False)


def _extract_nowcast(
    result: DynamicFactorMQ,
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

    # The quarterly gdp_growth variable is carried on the model's MONTHLY index
    # (Mariano-Murasawa mapping), and predicted_mean is defined at EVERY month: the value
    # at a quarter's THIRD month is that quarter's growth estimate, while later months are
    # rolling estimates that drift toward the NEXT quarter. So select the target quarter's
    # third month by label. (The previous code used dropna().iloc[-1], which silently
    # returns the wrong quarter whenever the monthly panel extends past the target quarter —
    # the normal live T-0 case, where next-quarter monthly indicators have already arrived.)
    q3_month = pd.Period(
        year=target_quarter.year,
        month=(target_quarter.quarter - 1) * 3 + 3,
        freq="M",
    )
    gdp_pred = pred_mean[gdp_col]
    if q3_month not in gdp_pred.index or pd.isna(gdp_pred.loc[q3_month]):
        span = f"{gdp_pred.index[0]}..{gdp_pred.index[-1]}" if len(gdp_pred.index) else "empty"
        msg = (
            f"DFM nowcast: no gdp_growth prediction at the target quarter's third month "
            f"{q3_month} (target {target_quarter}); prediction index spans {span}. "
            f"statsmodels may have changed its prediction indexing."
        )
        raise RuntimeError(msg)

    nowcast_qoq = float(gdp_pred.loc[q3_month])
    gdp_se = pred_se[gdp_col]
    se = float(gdp_se.loc[q3_month]) if q3_month in gdp_se.index else np.nan

    # Compute prediction intervals
    if np.isfinite(se) and se > 0:
        ci_70 = (nowcast_qoq - 1.04 * se, nowcast_qoq + 1.04 * se)
        ci_90 = (nowcast_qoq - 1.645 * se, nowcast_qoq + 1.645 * se)
    else:
        # Fallback: use historical GDP growth volatility, in the same log-difference
        # units as nowcast_qoq (compute_gdp_growth already scales by 100).
        hist_std = compute_gdp_growth(gdp).dropna().std()
        ci_70 = (nowcast_qoq - 1.04 * hist_std, nowcast_qoq + 1.04 * hist_std)
        ci_90 = (nowcast_qoq - 1.645 * hist_std, nowcast_qoq + 1.645 * hist_std)

    return nowcast_qoq, ci_70, ci_90


# --- GDP helpers ---


# --- Core nowcast ---


def _resolve_inputs(
    target_quarter: pd.Period | None,
    availability: DataAvailability | None,
    gdp: pd.Series | None,
    monthly_indicators: dict[str, DataSeries] | None,
    quarterly_indicators: dict[str, pd.Series] | None,
) -> tuple[pd.Period, DataAvailability, pd.Series, dict[str, DataSeries], dict[str, pd.Series]]:
    """Fill in any inputs left as None: fetch data, detect the target quarter, derive availability."""
    if gdp is None:
        gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()
    if target_quarter is None:
        target_quarter = detect_target_quarter(gdp)
    if monthly_indicators is None:
        monthly_indicators = _load_monthly_indicators()
    if quarterly_indicators is None:
        quarterly_indicators = _load_quarterly_indicators(target_quarter)
    if availability is None:
        availability = DataAvailability.from_live_data(
            monthly_indicators, quarterly_indicators, target_quarter,
        )
    return target_quarter, availability, gdp, monthly_indicators, quarterly_indicators


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
    # 1. Resolve inputs (fetch data / detect target quarter / derive availability as needed)
    target_quarter, availability, gdp, monthly_indicators, quarterly_indicators = _resolve_inputs(
        target_quarter, availability, gdp, monthly_indicators, quarterly_indicators,
    )

    gdp_truncated = gdp.loc[gdp.index < target_quarter]
    gdp_growth = compute_gdp_growth(gdp_truncated)

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
    tty = compute_tty(nowcast_qoq, gdp_truncated, target_quarter)
    tty_70 = (
        compute_tty(ci_70_qoq[0], gdp_truncated, target_quarter),
        compute_tty(ci_70_qoq[1], gdp_truncated, target_quarter),
    )
    tty_90 = (
        compute_tty(ci_90_qoq[0], gdp_truncated, target_quarter),
        compute_tty(ci_90_qoq[1], gdp_truncated, target_quarter),
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
    print_qoq_tty_header(result, "DFM")

    print(f"\n  Model: {result.n_factors} factors, VAR({result.factor_order})")
    print(f"  Monthly indicators: {result.n_monthly}")
    print(f"  Quarterly indicators: {result.n_quarterly}")

    print("\n  Data availability (monthly indicators):")
    for name, status in result.data_status.items():
        print(f"    {name:<25} {status}")

    try:
        from src.models.common.nowcast_diagnostics import print_capex_imports_hotness  # noqa: PLC0415

        print_capex_imports_hotness(target_quarter=result.target_quarter)
    except (ValueError, KeyError, OSError) as exc:
        logger.warning("Capex-imports hotness diagnostic failed: %s", exc)

    print("\n" + "=" * 70)


# --- Plotting ---


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


def _plot_productivity_adjustment() -> None:
    """Plot the labour-productivity HMA(13) trend that is netted into the labour inputs.

    Documents the applied adjustment (see PRODUCTIVITY_ADJUST_LABOUR): the HMA(13) trend is
    added to monthly employment & hours growth, so a negative trend discounts the labour
    signal. Skipped when the adjustment is disabled.
    """
    if not PRODUCTIVITY_ADJUST_LABOUR:
        return

    ulc = get_ulc_growth_qrtly().data
    hcoe = get_hourly_coe_growth_qrtly().data
    lp = get_labour_productivity_growth(ulc, hcoe).data.dropna()
    lp_trend = henderson_hma(lp, 13).reindex(lp.index)
    lp_4qta = lp_trend.rolling(4).mean()  # 4Q trailing avg of HMA — the carry-forward line

    df = pd.DataFrame({
        "Productivity growth (raw)": lp,
        "HMA(13) trend": lp_trend,
        "4Q trailing avg of HMA (carry-forward)": lp_4qta,
    })
    mg.line_plot_finalise(
        df,
        title="DFM labour-productivity adjustment (HMA-13 + 4Q trailing avg)",
        ylabel="% per quarter",
        y0=True,
        plot_from=pd.Period("2010Q1", "Q-DEC"),
        color=["lightsteelblue", "navy", "darkorange"],
        width=[1.0, 2.2, 2.2],
        rfooter="Source: ABS 5206.0",
        lfooter="Australia. Productivity = Δ hourly COE − Δ ULC. Carry-forward = 4Q trailing avg of HMA. "
                f"{pd.Timestamp.now().strftime('%d %b %Y')}. ",
        show=SHOW,
    )


# --- Live entry point ---


def run_nowcast() -> NowcastResult:
    """Run the live GDP nowcast with auto-detected target and availability."""
    result = nowcast()

    gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()

    mg.set_chart_dir(CHART_DIR)
    mg.clear_chart_dir()
    plot_nowcast_charts(NowcastChartSpec(
        model_label="DFM",
        target_quarter=result.target_quarter,
        gdp=gdp,
        gdp_qoq=result.gdp_qoq,
        gdp_tty=result.gdp_tty,
        gdp_qoq_70=result.gdp_qoq_70,
        gdp_qoq_90=result.gdp_qoq_90,
        gdp_tty_70=result.gdp_tty_70,
        gdp_tty_90=result.gdp_tty_90,
        accent="red",
        show=SHOW,
    ))
    _plot_factors(result)
    _plot_productivity_adjustment()

    return result


if __name__ == "__main__":
    run_nowcast()
