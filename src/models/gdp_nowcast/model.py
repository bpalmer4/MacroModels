"""GDP nowcasting model using bridge equations.

Nowcasts the next unpublished quarterly GDP growth (Q/Q and TTY) by combining
bridge equations that link high-frequency indicators to GDP. Monthly indicators
are SARIMA-completed to fill partial quarters, and bridge forecasts are combined
using inverse-MSE weights from expanding-window out-of-sample evaluation.

Bridge groups:
    1. Consumption:  retail trade turnover (monthly, ~2 months lead)
    2. Labour:       employment growth, hours worked growth (monthly, ~2 months lead)
    3. Investment:   building approvals growth (monthly, ~2 months lead)
    4. Trade:        goods trade balance (monthly, ~2 months lead)
    5. Prices:       quarterly CPI trimmed mean (~5 weeks lead), WPI growth (~3 weeks lead)
    6. Business:     company profits growth, business sales growth (~3 days lead)
    7. Production:   Cobb-Douglas (time-varying α from factor income shares)

Uncertainty is estimated via bootstrap resampling of bridge equation residuals.

Usage:
    # Live nowcast
    uv run python -m src.models.gdp_nowcast.model

    # Programmatic use
    from src.models.gdp_nowcast.model import nowcast, DataAvailability
    result = nowcast(target_quarter=pd.Period("2025Q4", "Q-DEC"))
    result = nowcast(
        target_quarter=pd.Period("2025Q4", "Q-DEC"),
        availability=DataAvailability.at_t_minus_1m(pd.Period("2025Q4", "Q-DEC")),
    )
"""

import logging
import warnings
from dataclasses import dataclass

import mgplot as mg
import numpy as np
import pandas as pd
import readabs as ra
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data import (
    get_building_approvals_monthly,
    get_capital_share,
    get_goods_balance_monthly,
    get_hours_worked_monthly,
    get_retail_turnover_monthly,
    get_trimmed_mean_qrtly,
)
from src.data.abs_loader import load_series
from src.data.business_indicators import (
    get_business_sales_growth_qrtly,
    get_company_profits_growth_qrtly,
)
from src.data.dataseries import DataSeries
from src.data.gdp import get_gdp
from src.data.inflation import get_genuine_monthly_cpi_index, get_monthly_cpi_index
from src.data.series_specs import EMPLOYMENT_PERSONS
from src.data.wpi import get_wpi_growth_qrtly

logger = logging.getLogger(__name__)

# Suppress statsmodels convergence warnings during SARIMA grid search
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

# --- Constants ---

CHART_DIR = "./charts/GDP-Nowcast/"
N_BOOTSTRAP = 1000
COVID_START = "2020Q1"
COVID_END = "2021Q1"
SHOW = False

# SARIMA candidate orders (p, d, q) — kept small for 1-2 month forecasts
SARIMA_ORDERS = [
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 1),
]
SEASONAL_ORDER = (0, 0, 0, 0)
SEASONAL_ORDER_12 = (1, 0, 0, 12)
MIN_SARIMA_TRAINING = 24  # minimum monthly observations for SARIMA fitting
MIN_SEASONAL_TRAINING = 36  # minimum observations to try seasonal SARIMA
MIN_BRIDGE_OBS = 20  # minimum quarterly observations for bridge estimation

# Typical ABS publication lags (months after reference month)
# Used to approximate data availability at each backtest horizon
PUBLICATION_LAGS = {
    "employment": 1,        # ~3 weeks after reference month
    "hours_worked": 1,      # same release as employment
    "retail": 1,            # ~4 weeks after reference month
    "building_approvals": 2,  # ~5-6 weeks after reference month
    "goods_balance": 2,     # ~5 weeks after reference month
    "cpi_monthly": 1,       # ~4 weeks after reference month
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
    cpi_monthly: pd.Period | None = None
    cpi_quarterly: bool = False
    wpi: bool = False
    business_profits: bool = False
    business_sales: bool = False

    def monthly_status(self, target_quarter: pd.Period) -> dict[str, str]:
        """Report data availability for each monthly indicator.

        Returns dict of {name: "0/3 months", "1/3 months", etc.}

        """
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        months = [
            pd.Period(year=q_year, month=q_start_month + i, freq="M")
            for i in range(3)
        ]

        status = {}
        for name in ("employment", "hours_worked", "retail", "building_approvals", "goods_balance", "cpi_monthly"):
            cutoff = getattr(self, name)
            if cutoff is None:
                status[name] = "0/3 months"
            else:
                n = sum(1 for m in months if m <= cutoff)
                status[name] = f"{n}/3 months"
        return status

    @classmethod
    def from_live_data(
        cls,
        monthly_indicators: dict[str, DataSeries],
        quarterly_indicators: dict[str, pd.Series],
        target_quarter: pd.Period,
    ) -> DataAvailability:
        """Detect availability from actual data."""
        avail = cls()
        for name, ds in monthly_indicators.items():
            last = ds.data.dropna().index[-1]
            setattr(avail, name, last)

        # Check quarterly indicators
        for q_name, series in quarterly_indicators.items():
            available = target_quarter in series.index and pd.notna(series.loc[target_quarter])
            setattr(avail, q_name, available)

        return avail

    @classmethod
    def at_t_minus_3m(cls, target_quarter: pd.Period) -> DataAvailability:
        """Approximate data available when previous quarter's GDP is published.

        At this point, ~0 months of the target quarter's monthly data exist.
        Monthly indicators are available up to about 1 month before the target quarter starts.
        No quarterly indicators for the target quarter.

        """
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        # The month before the target quarter starts
        pre_q = pd.Period(year=q_year, month=q_start_month, freq="M") - 1

        avail = cls()
        for name, lag in PUBLICATION_LAGS.items():
            # Data published with `lag` months delay
            setattr(avail, name, pre_q - lag)
        return avail

    @classmethod
    def at_t_minus_2m(cls, target_quarter: pd.Period) -> DataAvailability:
        """Approximate data available ~1 month after previous GDP publication.

        First month of the target quarter may be available for fast indicators.

        """
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        month_1 = pd.Period(year=q_year, month=q_start_month, freq="M")

        avail = cls()
        for name, lag in PUBLICATION_LAGS.items():
            setattr(avail, name, month_1 - lag + 1)
        return avail

    @classmethod
    def at_t_minus_1m(cls, target_quarter: pd.Period) -> DataAvailability:
        """Approximate data available ~2 months after previous GDP publication.

        Two months of the target quarter available for fast indicators.

        """
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        month_2 = pd.Period(year=q_year, month=q_start_month + 1, freq="M")

        avail = cls()
        for name, lag in PUBLICATION_LAGS.items():
            setattr(avail, name, month_2 - lag + 1)
        return avail

    @classmethod
    def at_t_minus_0(cls, target_quarter: pd.Period) -> DataAvailability:
        """Complete data set, just before GDP publication.

        All 3 months of monthly data available.
        All quarterly indicators published.

        """
        q_start_month = (target_quarter.quarter - 1) * 3 + 1
        q_year = target_quarter.year
        month_3 = pd.Period(year=q_year, month=q_start_month + 2, freq="M")

        return cls(
            employment=month_3,
            hours_worked=month_3,
            retail=month_3,
            building_approvals=month_3,
            goods_balance=month_3,
            cpi_monthly=month_3,
            cpi_quarterly=True,
            wpi=True,
            business_profits=True,
            business_sales=True,
        )


# --- Data structures ---


@dataclass
class BridgeResult:
    """Result from a single bridge equation."""

    name: str
    nowcast_qoq: float
    coefficients: pd.Series
    residuals: pd.Series
    fitted: pd.Series
    r_squared: float
    mse_oos: float  # out-of-sample MSE for weighting
    available: bool  # whether indicator data covers the nowcast quarter


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
    bridge_results: list[BridgeResult]
    weights: dict[str, float]
    data_status: dict[str, str]
    availability: DataAvailability | None = None


# --- SARIMA completion ---


def _fit_best_sarima(series: pd.Series, try_seasonal: bool = True) -> SARIMAX:
    """Fit best SARIMA model from candidate set by AIC."""
    best_aic = np.inf
    best_fit = None

    orders_to_try = []
    for order in SARIMA_ORDERS:
        orders_to_try.append((order, SEASONAL_ORDER))
        if try_seasonal and len(series.dropna()) > MIN_SEASONAL_TRAINING:
            orders_to_try.append((order, SEASONAL_ORDER_12))

    for order, seasonal in orders_to_try:
        try:
            model = SARIMAX(
                series.dropna(),
                order=order,
                seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=200)
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
        except (ValueError, np.linalg.LinAlgError):
            continue

    if best_fit is None:
        msg = "No SARIMA model converged"
        raise RuntimeError(msg)

    return best_fit


def complete_quarter_sarima(
    monthly_series: pd.Series,
    target_quarter: pd.Period,
) -> tuple[pd.Series, list[str]]:
    """Complete a partial quarter using SARIMA forecasts."""
    q_start_month = (target_quarter.quarter - 1) * 3 + 1
    q_year = target_quarter.year
    months = [
        pd.Period(year=q_year, month=q_start_month + i, freq="M")
        for i in range(3)
    ]

    statuses = []
    values = []

    for month in months:
        if month in monthly_series.index and pd.notna(monthly_series.loc[month]):
            statuses.append("observed")
            values.append(monthly_series.loc[month])
        else:
            statuses.append("forecast")
            values.append(np.nan)

    if all(s == "observed" for s in statuses):
        result = pd.Series(values, index=pd.PeriodIndex(months, freq="M"))
        return result, statuses

    train_end = monthly_series.dropna().index[-1]
    train = monthly_series.loc[:train_end].dropna()

    if len(train) < MIN_SARIMA_TRAINING:
        logger.warning("Short training series (%d obs) for SARIMA", len(train))

    n_forecast = 0
    for i, month in enumerate(months):
        if month > train_end:
            n_forecast = 3 - i
            break

    if n_forecast > 0:
        sarima_fit = _fit_best_sarima(train)
        forecast = sarima_fit.forecast(steps=n_forecast)

        fc_idx = 0
        for i, _month in enumerate(months):
            if statuses[i] == "forecast" and fc_idx < len(forecast):
                values[i] = forecast.iloc[fc_idx]
                fc_idx += 1

    result = pd.Series(values, index=pd.PeriodIndex(months, freq="M"))
    return result, statuses


def monthly_to_quarterly_completed(
    monthly_series: pd.Series,
    target_quarter: pd.Period,
    agg: str = "mean",
) -> tuple[float, list[str]]:
    """Aggregate monthly data to a single quarterly value, SARIMA-completing if needed."""
    completed, statuses = complete_quarter_sarima(monthly_series, target_quarter)
    if agg == "sum":
        return completed.sum(), statuses
    return completed.mean(), statuses


# --- Data truncation ---


def _truncate_monthly(series: pd.Series, cutoff: pd.Period | None) -> pd.Series:
    """Truncate a monthly series to data available through cutoff period."""
    if cutoff is None:
        return pd.Series(dtype=float)
    return series.loc[series.index <= cutoff].copy()


def _truncate_quarterly(series: pd.Series, target_quarter: pd.Period) -> pd.Series:
    """Truncate a quarterly series to exclude the target quarter and beyond."""
    return series.loc[series.index < target_quarter].copy()


# --- Bridge equations ---


def _add_covid_dummy(df: pd.DataFrame) -> pd.DataFrame:
    """Add COVID dummy variable covering 2020Q1-2021Q1."""
    df = df.copy()
    df["covid"] = 0.0
    mask = (df.index >= COVID_START) & (df.index <= COVID_END)
    df.loc[mask, "covid"] = 1.0
    return df


def _estimate_bridge(
    gdp_growth: pd.Series,
    indicators: pd.DataFrame,
    bridge_name: str,
    n_lags: int = 1,
) -> BridgeResult | None:
    """Estimate a bridge equation via OLS with expanding-window OOS evaluation."""
    df = pd.DataFrame({"gdp_growth": gdp_growth})
    for col in indicators.columns:
        df[col] = indicators[col]
        for lag in range(1, n_lags + 1):
            df[f"{col}_L{lag}"] = indicators[col].shift(lag)

    df["gdp_growth_L1"] = gdp_growth.shift(1)
    df = _add_covid_dummy(df)
    df["const"] = 1.0

    df = df.dropna()

    if len(df) < MIN_BRIDGE_OBS:
        logger.warning("Bridge '%s': too few observations (%d)", bridge_name, len(df))
        return None

    y = df["gdp_growth"]
    X = df.drop(columns=["gdp_growth"])

    try:
        model = sm.OLS(y, X)
        fit = model.fit()
    except (ValueError, np.linalg.LinAlgError):
        logger.warning("Bridge '%s': OLS estimation failed", bridge_name)
        return None

    min_train = max(MIN_BRIDGE_OBS, len(df) // 3)
    oos_errors = []

    for t in range(min_train, len(df)):
        y_train = y.iloc[:t]
        X_train = X.iloc[:t]
        y_true = y.iloc[t]
        X_new = X.iloc[t : t + 1]

        try:
            oos_fit = sm.OLS(y_train, X_train).fit()
            y_pred = oos_fit.predict(X_new).iloc[0]
            oos_errors.append((y_true - y_pred) ** 2)
        except (ValueError, np.linalg.LinAlgError):
            continue

    mse_oos = np.mean(oos_errors) if oos_errors else np.inf

    return BridgeResult(
        name=bridge_name,
        nowcast_qoq=np.nan,
        coefficients=fit.params,
        residuals=fit.resid,
        fitted=fit.fittedvalues,
        r_squared=fit.rsquared,
        mse_oos=mse_oos,
        available=True,
    )


def _bridge_nowcast(
    bridge: BridgeResult,
    gdp_growth: pd.Series,
    indicators_nowcast: pd.DataFrame,
    target_quarter: pd.Period,
) -> float:
    """Generate a nowcast from a fitted bridge equation."""
    row = {}
    for col in indicators_nowcast.columns:
        if target_quarter in indicators_nowcast.index:
            row[col] = indicators_nowcast.loc[target_quarter, col]
        prev_q = target_quarter - 1
        if prev_q in indicators_nowcast.index:
            row[f"{col}_L1"] = indicators_nowcast.loc[prev_q, col]

    if target_quarter - 1 in gdp_growth.index:
        row["gdp_growth_L1"] = gdp_growth.loc[target_quarter - 1]
    else:
        row["gdp_growth_L1"] = gdp_growth.iloc[-1]

    row["covid"] = 1.0 if COVID_START <= str(target_quarter) <= COVID_END else 0.0
    row["const"] = 1.0

    X_new = pd.DataFrame([row])
    for col in bridge.coefficients.index:
        if col not in X_new.columns:
            X_new[col] = 0.0
    X_new = X_new[bridge.coefficients.index]

    return float(bridge.coefficients @ X_new.iloc[0])


# --- Production bridge (Cobb-Douglas) ---


def _build_production_bridge(
    gdp_growth: pd.Series,
    target_quarter: pd.Period,
    _availability: DataAvailability | None = None,  # retained for future vintage data truncation
) -> BridgeResult | None:
    """Build Cobb-Douglas production bridge with time-varying α and MFP trend.

    Uses independent inputs from the NAIRU model's production function:
        Y*_t = α × dK + (1-α) × dL + dMFP

    Where:
        dK = capital stock growth (Modellers Database 1364.0)
        dL = labour force growth (Modellers Database 1364.0)
        dMFP = Solow residual trend (derived from wage data, HMA-smoothed)
        α = GOS / (GOS + COE) from factor income shares

    These inputs are independent of the other bridges (which use employment,
    hours, retail, approvals, and trade data from different ABS catalogues).

    """
    from src.data import (  # noqa: PLC0415
        get_capital_growth_qrtly,
        get_hourly_coe_growth_qrtly,
        get_hours_growth_qrtly,
        get_labour_force_growth_qrtly,
        get_ulc_growth_qrtly,
    )

    alpha_ds = get_capital_share()
    alpha = alpha_ds.data

    capital_growth = get_capital_growth_qrtly().data
    lf_growth = get_labour_force_growth_qrtly().data

    # MFP trend from wage data (independent of GDP)
    ulc_growth = get_ulc_growth_qrtly().data
    hcoe_growth = get_hourly_coe_growth_qrtly().data
    hours_growth = get_hours_growth_qrtly().data

    # Use unfloored MFP: for nowcasting actual GDP (not potential output),
    # negative TFP is real and informative — it captures the productivity drag
    from src.data.productivity import get_mfp_growth  # noqa: PLC0415

    mfp_raw = get_mfp_growth(
        ulc_growth=ulc_growth,
        hcoe_growth=hcoe_growth,
        capital_growth=capital_growth,
        hours_growth=hours_growth,
        alpha=alpha,
    ).data

    # HMA smooth but do NOT floor at zero
    from src.data.henderson import hma as henderson_hma  # noqa: PLC0415

    mfp_clean = mfp_raw.dropna()
    mfp_trend = henderson_hma(mfp_clean, 51).reindex(mfp_raw.index)

    # Truncate all to before target quarter
    alpha = _truncate_quarterly(alpha, target_quarter)
    capital_growth = _truncate_quarterly(capital_growth, target_quarter)
    lf_growth = _truncate_quarterly(lf_growth, target_quarter)
    mfp_trend = _truncate_quarterly(mfp_trend, target_quarter)

    if len(alpha) == 0 or len(capital_growth) == 0 or len(lf_growth) == 0:
        return None

    # Compute potential growth: α × dK + (1-α) × dL + dMFP
    common_idx = (
        alpha.index
        .intersection(capital_growth.index)
        .intersection(lf_growth.index)
        .intersection(mfp_trend.index)
    )

    potential_growth = pd.Series(index=common_idx, dtype=float)
    for q in common_idx:
        a = alpha.loc[q]
        potential_growth.loc[q] = (
            a * capital_growth.loc[q]
            + (1 - a) * lf_growth.loc[q]
            + mfp_trend.loc[q]
        )

    # Bridge: GDP_growth ~ potential_growth + lags + covid
    indicators = pd.DataFrame({"potential_growth": potential_growth})
    bridge = _estimate_bridge(gdp_growth, indicators, "Production (Cobb-Douglas)")

    if bridge is None:
        return None

    # Nowcast: carry forward latest values
    latest_alpha = alpha.iloc[-1]
    latest_dk = capital_growth.iloc[-1]
    latest_dl = lf_growth.iloc[-1]
    latest_mfp = mfp_trend.iloc[-1]

    pg_nowcast = latest_alpha * latest_dk + (1 - latest_alpha) * latest_dl + latest_mfp

    indicators_ext = indicators.copy()
    indicators_ext.loc[target_quarter] = pg_nowcast

    bridge.nowcast_qoq = _bridge_nowcast(bridge, gdp_growth, indicators_ext, target_quarter)
    return bridge


# --- Core nowcast function ---


def _load_monthly_indicators() -> dict[str, DataSeries]:
    """Load all monthly indicator series (full history)."""
    indicators: dict[str, DataSeries] = {
        "retail": get_retail_turnover_monthly(),
        "building_approvals": get_building_approvals_monthly(),
        "hours_worked": get_hours_worked_monthly(),
        "employment": load_series(EMPLOYMENT_PERSONS),
        "goods_balance": get_goods_balance_monthly(),
    }

    # Monthly CPI: spliced series for bridge estimation (full history),
    # genuine monthly series stored separately for SARIMA completion
    cpi_spliced = get_monthly_cpi_index()
    indicators["cpi_monthly"] = cpi_spliced

    return indicators


def _load_quarterly_indicators() -> dict[str, pd.Series]:
    """Load all quarterly indicator series (full history)."""
    indicators = {}

    loaders = {
        "cpi_quarterly": get_trimmed_mean_qrtly,
        "wpi": get_wpi_growth_qrtly,
        "business_profits": get_company_profits_growth_qrtly,
        "business_sales": get_business_sales_growth_qrtly,
    }
    for name, loader in loaders.items():
        try:
            ds = loader()
            indicators[name] = ds.data.dropna()
        except (ValueError, KeyError, OSError):
            logger.warning("Failed to load quarterly indicator '%s'", name)

    return indicators


# Bridge display names for quarterly indicators
QUARTERLY_BRIDGE_NAMES = {
    "cpi_quarterly": "Prices: CPI",
    "wpi": "Prices: WPI",
    "business_profits": "Business: profits",
    "business_sales": "Business: sales",
}


def _get_labour_productivity_trend(target_quarter: pd.Period) -> pd.Series:
    """Compute HMA(13) labour productivity trend from wage data.

    Labour productivity = Δhcoe - Δulc (from ULC = HCOE / LP identity).
    Smoothed with Henderson MA(13) for a persistent, forecastable trend.
    Not floored: negative productivity is real and informative for nowcasting.

    """
    from src.data import get_hourly_coe_growth_qrtly, get_ulc_growth_qrtly  # noqa: PLC0415
    from src.data.henderson import hma as henderson_hma  # noqa: PLC0415
    from src.data.productivity import get_labour_productivity_growth  # noqa: PLC0415

    ulc = get_ulc_growth_qrtly().data
    hcoe = get_hourly_coe_growth_qrtly().data
    lp = get_labour_productivity_growth(ulc, hcoe).data

    lp_clean = lp.dropna()
    lp_trend = henderson_hma(lp_clean, 13).reindex(lp.index)

    # Truncate and carry forward latest value for target quarter
    lp_trend = _truncate_quarterly(lp_trend, target_quarter)
    if len(lp_trend) > 0:
        lp_trend.loc[target_quarter] = lp_trend.iloc[-1]

    return lp_trend


def _prepare_monthly_series(
    monthly: pd.Series,
    cutoff: pd.Period | None,
    sarima_override: pd.Series | None,
) -> tuple[pd.Series, pd.Series] | None:
    """Prepare monthly series for a bridge: truncate, resolve SARIMA series.

    Returns (bridge_series, sarima_series) or None if insufficient data.
    """
    bridge_series = _truncate_monthly(monthly.dropna(), cutoff)

    if len(bridge_series) < MIN_SARIMA_TRAINING:
        return None

    if sarima_override is not None:
        sarima_series = _truncate_monthly(sarima_override.dropna(), cutoff)
        if len(sarima_series) < MIN_SARIMA_TRAINING:
            return None
    else:
        sarima_series = bridge_series

    return bridge_series, sarima_series


def _build_monthly_bridges(
    gdp_growth: pd.Series,
    monthly_indicators: dict[str, DataSeries],
    target_quarter: pd.Period,
    availability: DataAvailability,
) -> list[BridgeResult]:
    """Build bridge equations for monthly indicators, respecting data availability.

    Labour bridges include an HMA(13) labour productivity trend as an additional
    regressor, so the model can discount strong employment when productivity is weak.

    """
    bridges = []

    # Compute labour productivity trend once for the labour bridges
    lp_trend = _get_labour_productivity_trend(target_quarter)

    # (name, indicator_key, aggregation, growth_calc, include_productivity, sarima_key)
    # sarima_key: if set, use a different series for SARIMA completion (e.g. genuine
    # monthly CPI for SARIMA, spliced series for bridge estimation)
    bridge_configs = [
        ("Consumption", "retail", "sum", True, False, None),
        ("Investment", "building_approvals", "sum", True, False, None),
        ("Labour: hours", "hours_worked", "sum", True, True, None),
        ("Labour: employment", "employment", "mean", True, True, None),
        ("Trade", "goods_balance", "sum", False, False, None),
        ("Prices: monthly CPI", "cpi_monthly", "mean", True, False, "cpi_monthly_genuine"),
    ]

    # Load genuine monthly CPI for SARIMA (6484.0 + 640106, not interpolated quarterly)
    genuine_monthly_cpi = get_genuine_monthly_cpi_index().data
    sarima_overrides = {"cpi_monthly_genuine": genuine_monthly_cpi}

    for bridge_name, ind_key, agg, calc_growth, include_prod, sarima_key in bridge_configs:
        ds = monthly_indicators[ind_key]
        cutoff = getattr(availability, ind_key)
        override = sarima_overrides.get(sarima_key) if sarima_key else None

        prepared = _prepare_monthly_series(ds.data, cutoff, override)
        if prepared is None:
            logger.info("Bridge '%s': insufficient data, skipping", bridge_name)
            continue
        monthly, sarima_series = prepared

        try:
            q_value, _month_statuses = monthly_to_quarterly_completed(
                sarima_series, target_quarter, agg=agg,
            )
        except RuntimeError:
            logger.warning("Bridge '%s': SARIMA completion failed", bridge_name)
            continue

        # Build full quarterly history
        if agg == "sum":
            quarterly_hist = ra.monthly_to_qtly(monthly, q_ending="DEC", f="sum")
        else:
            quarterly_hist = ra.monthly_to_qtly(monthly, q_ending="DEC", f="mean")

        # Append completed target quarter
        quarterly_full = quarterly_hist.copy()
        quarterly_full.loc[target_quarter] = q_value

        # Convert to growth if needed
        if calc_growth:
            log_q = np.log(quarterly_full) * 100
            growth_q = log_q.diff(1)
        else:
            growth_q = quarterly_full

        # Build indicator DataFrame, adding productivity for labour bridges
        indicators_df = pd.DataFrame({bridge_name: growth_q})
        if include_prod and len(lp_trend.dropna()) > 0:
            indicators_df["lp_trend"] = lp_trend

        # Estimate bridge on historical data (excluding target quarter)
        bridge = _estimate_bridge(
            gdp_growth,
            indicators_df.drop(index=target_quarter, errors="ignore"),
            bridge_name,
        )

        if bridge is None:
            continue

        bridge.nowcast_qoq = _bridge_nowcast(bridge, gdp_growth, indicators_df, target_quarter)
        bridge.available = True
        bridges.append(bridge)

    return bridges


def _build_quarterly_bridges(
    gdp_growth: pd.Series,
    quarterly_indicators: dict[str, pd.Series],
    target_quarter: pd.Period,
    availability: DataAvailability,
) -> list[BridgeResult]:
    """Build bridge equations for quarterly indicators, respecting availability."""
    bridges = []

    for q_name, series in quarterly_indicators.items():
        bridge_name = QUARTERLY_BRIDGE_NAMES.get(q_name, q_name)
        available = getattr(availability, q_name, False)

        # If available, include target quarter; otherwise exclude it
        if available:
            indicators_df = pd.DataFrame({bridge_name: series})
        else:
            indicators_df = pd.DataFrame({bridge_name: _truncate_quarterly(series, target_quarter)})

        hist_df = indicators_df.loc[indicators_df.index < target_quarter]
        bridge = _estimate_bridge(gdp_growth, hist_df, bridge_name)

        if bridge is None:
            continue

        if available and target_quarter in series.index:
            bridge.nowcast_qoq = _bridge_nowcast(bridge, gdp_growth, indicators_df, target_quarter)
            bridge.available = True
        else:
            bridge.available = False

        bridges.append(bridge)

    return bridges


# --- Combination and bootstrap ---


def _combine_bridges(
    bridges: list[BridgeResult],
) -> tuple[float, dict[str, float]]:
    """Combine bridge nowcasts using inverse-MSE weights."""
    active = [b for b in bridges if b.available and np.isfinite(b.mse_oos) and b.mse_oos > 0]

    if not active:
        msg = "No active bridges available for combination"
        raise RuntimeError(msg)

    inv_mse = {b.name: 1.0 / b.mse_oos for b in active}
    total_inv_mse = sum(inv_mse.values())
    weights = {name: w / total_inv_mse for name, w in inv_mse.items()}

    combined = sum(b.nowcast_qoq * weights[b.name] for b in active)

    return combined, weights


def _bootstrap_intervals(
    bridges: list[BridgeResult],
    weights: dict[str, float],
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bootstrap prediction intervals from bridge residuals."""
    active = [b for b in bridges if b.name in weights]

    pooled_residuals = []
    for bridge in active:
        w = weights[bridge.name]
        pooled_residuals.extend((bridge.residuals * w).tolist())

    pooled_residuals = np.array(pooled_residuals)
    pooled_residuals = pooled_residuals[np.isfinite(pooled_residuals)]

    combined_nowcast = sum(b.nowcast_qoq * weights[b.name] for b in active)

    rng = np.random.default_rng(42)
    bootstrap_draws = np.array([
        combined_nowcast + rng.choice(pooled_residuals, size=len(active)).sum()
        for _ in range(n_bootstrap)
    ])

    ci_70 = (float(np.percentile(bootstrap_draws, 15)), float(np.percentile(bootstrap_draws, 85)))
    ci_90 = (float(np.percentile(bootstrap_draws, 5)), float(np.percentile(bootstrap_draws, 95)))

    return ci_70, ci_90


# --- GDP helpers ---


def _detect_target_quarter(gdp: pd.Series) -> pd.Period:
    """Auto-detect the next unpublished GDP quarter."""
    last_published = gdp.dropna().index[-1]
    return last_published + 1


def _compute_gdp_growth(gdp: pd.Series) -> pd.Series:
    """Compute Q/Q GDP growth as log difference x 100."""
    return np.log(gdp).diff(1) * 100


def _compute_tty(gdp_growth_qoq: float, gdp: pd.Series, target_quarter: pd.Period) -> float:
    """Compute through-the-year growth from Q/Q nowcast."""
    last_gdp = gdp.iloc[-1]
    projected_gdp = last_gdp * np.exp(gdp_growth_qoq / 100)

    q_minus_4 = target_quarter - 4
    gdp_4q_ago = gdp.loc[q_minus_4] if q_minus_4 in gdp.index else gdp.iloc[-4]

    return (projected_gdp / gdp_4q_ago - 1) * 100


# --- Core nowcast function ---


def nowcast(
    target_quarter: pd.Period | None = None,
    availability: DataAvailability | None = None,
    gdp: pd.Series | None = None,
    monthly_indicators: dict[str, DataSeries] | None = None,
    quarterly_indicators: dict[str, pd.Series] | None = None,
    quiet: bool = False,
) -> NowcastResult:
    """Run the GDP nowcast model.

    Args:
        target_quarter: The quarter to nowcast. If None, auto-detected.
        availability: Data availability specification. If None, detected from data.
        gdp: GDP series (CVM, SA). If None, fetched from ABS.
        monthly_indicators: Pre-loaded monthly indicators. If None, fetched from ABS.
        quarterly_indicators: Pre-loaded quarterly indicators. If None, fetched from ABS.
        quiet: If True, suppress terminal output.

    Returns:
        NowcastResult with point estimates, intervals, and diagnostics.

    """
    # 1. Load data if not provided
    if gdp is None:
        gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()

    if target_quarter is None:
        target_quarter = _detect_target_quarter(gdp)

    # Truncate GDP to before target quarter (for backtesting)
    gdp = gdp.loc[gdp.index < target_quarter]
    gdp_growth = _compute_gdp_growth(gdp)

    if monthly_indicators is None:
        monthly_indicators = _load_monthly_indicators()

    if quarterly_indicators is None:
        quarterly_indicators = _load_quarterly_indicators()

    # 2. Determine data availability
    if availability is None:
        availability = DataAvailability.from_live_data(
            monthly_indicators, quarterly_indicators, target_quarter,
        )

    data_status = availability.monthly_status(target_quarter)

    if not quiet:
        print(f"Target quarter: {target_quarter}")
        print(f"Last published GDP: {gdp.index[-1]}")

    # 3. Build bridges
    monthly_bridges = _build_monthly_bridges(
        gdp_growth, monthly_indicators, target_quarter, availability,
    )
    quarterly_bridges = _build_quarterly_bridges(
        gdp_growth, quarterly_indicators, target_quarter, availability,
    )
    production_bridge = _build_production_bridge(gdp_growth, target_quarter, availability)

    all_bridges = monthly_bridges + quarterly_bridges
    if production_bridge is not None:
        all_bridges.append(production_bridge)

    # 4. Combine
    combined_qoq, weights = _combine_bridges(all_bridges)

    # 5. Bootstrap intervals
    ci_70_qoq, ci_90_qoq = _bootstrap_intervals(all_bridges, weights)

    # 6. Compute TTY
    tty = _compute_tty(combined_qoq, gdp, target_quarter)
    tty_70 = (_compute_tty(ci_70_qoq[0], gdp, target_quarter),
              _compute_tty(ci_70_qoq[1], gdp, target_quarter))
    tty_90 = (_compute_tty(ci_90_qoq[0], gdp, target_quarter),
              _compute_tty(ci_90_qoq[1], gdp, target_quarter))

    # 7. Assemble result
    result = NowcastResult(
        target_quarter=target_quarter,
        gdp_qoq=combined_qoq,
        gdp_tty=tty,
        gdp_qoq_70=ci_70_qoq,
        gdp_qoq_90=ci_90_qoq,
        gdp_tty_70=tty_70,
        gdp_tty_90=tty_90,
        bridge_results=all_bridges,
        weights=weights,
        data_status=data_status,
        availability=availability,
    )

    if not quiet:
        _print_summary(result)

    return result


# --- Output ---


def _print_summary(result: NowcastResult) -> None:
    """Print nowcast summary to terminal."""
    print("\n" + "=" * 70)
    print(f"  GDP NOWCAST: {result.target_quarter}")
    print("=" * 70)

    print(f"\n  Q/Q growth:   {result.gdp_qoq:+.2f}%")
    print(f"    70% CI:     [{result.gdp_qoq_70[0]:+.2f}%, {result.gdp_qoq_70[1]:+.2f}%]")
    print(f"    90% CI:     [{result.gdp_qoq_90[0]:+.2f}%, {result.gdp_qoq_90[1]:+.2f}%]")

    print(f"\n  TTY growth:   {result.gdp_tty:+.2f}%")
    print(f"    70% CI:     [{result.gdp_tty_70[0]:+.2f}%, {result.gdp_tty_70[1]:+.2f}%]")
    print(f"    90% CI:     [{result.gdp_tty_90[0]:+.2f}%, {result.gdp_tty_90[1]:+.2f}%]")

    print("\n  Bridge contributions:")
    print(f"  {'Bridge':<30} {'Weight':>8} {'Nowcast':>10} {'Status':>12}")
    print("  " + "-" * 62)
    for bridge in result.bridge_results:
        w = result.weights.get(bridge.name, 0.0)
        status = "active" if bridge.available else "excluded"
        nowcast_str = f"{bridge.nowcast_qoq:+.2f}%" if bridge.available else "n/a"
        print(f"  {bridge.name:<30} {w:>7.1%} {nowcast_str:>10} {status:>12}")

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
        lfooter=f"Australia. Nowcast: {lfooter_value}. {pd.Timestamp.now().strftime('%d %b %Y')}. ",
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
        title=f"GDP Growth Nowcast (Q/Q): {result.target_quarter}",
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
        title=f"GDP Growth Nowcast (TTY): {result.target_quarter}",
        lfooter_value=f"{result.gdp_tty:+.2f}%",
    )


def _plot_growth(result: NowcastResult, gdp: pd.Series) -> None:
    """Plot GDP growth in bar (Q/Q) + line (TTY) format with nowcast appended."""
    gdp_growth = _compute_gdp_growth(gdp).dropna()
    tty = ((gdp / gdp.shift(4)) - 1) * 100

    # Append nowcast quarter
    gdp_growth.loc[result.target_quarter] = result.gdp_qoq
    tty.loc[result.target_quarter] = result.gdp_tty

    # growth_plot_finalise: first column = line (TTY), second = bars (Q/Q)
    growth_df = pd.DataFrame({
        "Annual Growth": tty,
        "Quarterly Growth": gdp_growth,
    }).iloc[-20:]

    mg.growth_plot_finalise(
        growth_df,
        title=f"GDP Growth Nowcast: {result.target_quarter}",
        ylabel="Per cent",
        rfooter="Source: ABS 5206.0",
        lfooter=f"Australia. Nowcast Q/Q: {result.gdp_qoq:+.2f}%, TTY: {result.gdp_tty:+.2f}%. "
                f"{pd.Timestamp.now().strftime('%d %b %Y')}. ",
        legend={"loc": "best", "fontsize": "x-small"},
        show=SHOW,
    )


def _plot_bridge_weights(result: NowcastResult) -> None:
    """Plot bridge weights and nowcast values as two horizontal bar charts."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    active = [(b.name, b.nowcast_qoq, result.weights.get(b.name, 0))
              for b in result.bridge_results if b.available]

    if not active:
        return

    names, nowcasts, weights = zip(*active, strict=True)
    date_str = pd.Timestamp.now().strftime("%d %b %Y")
    q = result.target_quarter

    # Chart 1: Weights (inverse-MSE)
    _, ax1 = plt.subplots(figsize=(10, 5))
    ax1.barh(range(len(names)), [w * 100 for w in weights], color="steelblue", alpha=0.8)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    mg.finalise_plot(
        ax1,
        title=f"Bridge Weights: {q}",
        xlabel="Weight (%)",
        rfooter="Source: ABS",
        lfooter=f"Australia. Inverse-MSE weights. {date_str}. ",
        show=SHOW,
    )

    # Chart 2: Nowcast values (what each bridge thinks GDP growth is)
    colors = ["#2ca02c" if n >= 0 else "#d62728" for n in nowcasts]
    _, ax2 = plt.subplots(figsize=(10, 5))
    ax2.barh(range(len(names)), nowcasts, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    mg.finalise_plot(
        ax2,
        title=f"Bridge Nowcasts: {q}",
        xlabel="Nowcast Q/Q growth (%)",
        rfooter="Source: ABS",
        lfooter=f"Australia. Combined: {result.gdp_qoq:+.2f}%. {date_str}. ",
        show=SHOW,
    )


# --- Live entry point ---


def run_nowcast() -> NowcastResult:
    """Run the live GDP nowcast with auto-detected target and availability.

    Fetches latest data, generates charts, and prints summary.

    """
    result = nowcast()

    # Generate charts
    gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()
    gdp_growth = _compute_gdp_growth(gdp)

    mg.set_chart_dir(CHART_DIR)
    mg.clear_chart_dir()
    _plot_nowcast(result, gdp_growth)
    _plot_nowcast_tty(result, gdp)
    _plot_growth(result, gdp)
    _plot_bridge_weights(result)

    return result


if __name__ == "__main__":
    run_nowcast()
