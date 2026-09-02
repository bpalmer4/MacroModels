"""Observation assembly for the potential_uc model.

Three series, three sources, one aligned sample:

- log GDP x 100          (ABS 5206.0, chain volume, SA)
- log hours worked x 100 (ABS 6202.0, monthly hours in all jobs, SA, summed
                          to quarterly)
- log civilian population 15+ x 100 (ABS 6202.0, Original — no seasonality)
- log participation rate x 100      (ABS 6202.0, SA)
- trimmed mean inflation, annual % (ABS 6401.0), on either the quarterly
  annualised or the four-quarter basis (see `ModelConfig.pi_basis`)
- optionally, annual consumption-goods import price growth lagged one quarter
  (ABS 6457.0), demeaned, as the Phillips curve's supply control

Population and participation enter as the demographic decomposition of trend
hours (see `equations/trend_hours.py`). They play different roles: population
is used directly as data, because it is measured and acyclical; participation
is *not*, because it is cyclical, and instead gets its own observation equation
with a loading on the output gap.

Hours come from the Labour Force Survey rather than the National Accounts
hours index (also 5206.0) because the LFS series publishes ~6 weeks ahead of
GDP. The consequence is that GDP-per-LFS-hour is not the ABS published
productivity measure: the two hours concepts differ in scope (LFS is civilian
15+ from a rotating household survey; the National Accounts measure is
benchmarked and includes defence and non-civilian employment). The model
therefore *defines* labour productivity as GDP per LFS hour, which makes the
identity

    log_gdp = log_hours + log_productivity

hold exactly by construction. Hours is an index, so the identity holds only up
to an additive constant; that constant is absorbed by the initial-state prior
on lp*. The level of lp* is not interpretable — its growth and its gap are.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.data.gdp import get_log_gdp
from src.data.henderson import hma
from src.data.import_prices import get_import_price_growth_lagged_annual
from src.data.inflation import get_trimmed_mean_annual, get_trimmed_mean_qrtly
from src.data.labour_force import (
    get_civilian_population_qrtly,
    get_hours_worked_qrtly,
    get_participation_rate_qrtly,
)

_NAME_WIDTH = 27

# Bounds for the tail-extension ARIMA search. Small on purpose: the series being
# extended is a population level, which is close to deterministic over a few
# quarters, so a high-order fit would add variance without adding information.
_MAX_ARIMA_ORDER = 2


def _extend_tail_by_arima(series: pd.Series, periods: int) -> pd.Series:
    """Extend a series forward by `periods` using a low-order ARIMA fit.

    Returns the original series with the forecast appended. Falls back to the
    unextended series if no candidate model converges.
    """
    best_fit, best_aic = None, np.inf
    values = series.to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # non-invertible starts / convergence chatter
        for d in (1, 2):
            for p in range(_MAX_ARIMA_ORDER + 1):
                for q in range(_MAX_ARIMA_ORDER + 1):
                    try:
                        fit = ARIMA(values, order=(p, d, q)).fit()
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    if np.isfinite(fit.aic) and fit.aic < best_aic:
                        best_aic, best_fit = fit.aic, fit

        if best_fit is None:
            return series

        forecast = best_fit.forecast(periods)

    index = pd.period_range(series.index[-1] + 1, periods=periods, freq="Q")
    return pd.concat([series, pd.Series(forecast, index=index)])


def smooth_log_level(log_level: pd.Series, terms: int) -> pd.Series:
    """Henderson-smooth a log level series, ARIMA-extending the tail first.

    Henderson's end weights are asymmetric, which drags the trend's last point
    toward the interior of the series — on population it pulled 2026Q2 growth
    to 1.62 against a recent run-rate near 1.75. That endpoint feeds current
    trend hours growth, hence current potential growth and the current output
    gap, so the drag is not cosmetic: it biases the estimated gap upward.

    Extending the series forward by half the filter width means the last real
    observation is smoothed with symmetric weights. The start needs no such
    treatment: the labour force series run from 1978 while the model sample
    starts in 1993Q1, so smoothing the full history and trimming afterwards
    already gives the first in-sample quarter a full symmetric window.

    This mirrors the `arima_extend=True` handling used in the 6202 labour force
    notebook's breakeven-employment chart, for the same reason.
    """
    log_level = log_level.dropna()
    extended = _extend_tail_by_arima(log_level, periods=(terms - 1) // 2)
    return hma(extended, terms).reindex(log_level.index)


def smooth_population(log_pop: pd.Series, terms: int) -> pd.Series:
    """Henderson-smooth log population. See `smooth_log_level`."""
    return smooth_log_level(log_pop, terms)


def _inflation(pi_basis: str) -> tuple[pd.Series, str]:
    """Return the Phillips curve's left-hand side, in annual per cent.

    "quarterly" is the quarterly trimmed mean rate multiplied by four, so it is
    on the same scale as the anchor but is *non-overlapping*: consecutive
    observations share no CPI quarters, and iid errors are therefore
    defensible. "annual" is the four-quarter rate, which overlaps three
    quarters out of four; see `ModelConfig.pi_basis`.
    """
    if pi_basis == "quarterly":
        return get_trimmed_mean_qrtly().data * 4.0, "trimmed mean q/q ann. (6401.0)"
    return get_trimmed_mean_annual().data, "trimmed mean y/y (6401.0)"


def _load_series(
    spec: str,
    smooth_pop: int,
    pi_basis: str = "quarterly",
    supply_control: str | None = None,
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Load the input series this specification needs, with display labels."""
    pi_series, pi_label = _inflation(pi_basis)
    columns: dict[str, pd.Series] = {
        "log_gdp": get_log_gdp().data,
        "pi": pi_series,
    }
    labels = {
        "log_gdp": "log GDP (5206.0)",
        "pi": pi_label,
    }

    if supply_control == "import_prices":
        columns["supply"] = get_import_price_growth_lagged_annual().data
        labels["supply"] = "import price growth (6457.0)"

    if spec != "labour":
        return columns, labels

    log_pop = np.log(get_civilian_population_qrtly().data) * 100
    # Population is a trend input to h*, so its estimation noise would show up
    # as jitter in trend hours growth. Smoothed on the full history (from
    # 1978Q2) and trimmed to the sample later, so the sample start still gets a
    # full symmetric window.
    if smooth_pop:
        log_pop = smooth_population(log_pop, smooth_pop)

    columns["log_hours"] = np.log(get_hours_worked_qrtly().data) * 100
    columns["log_pop"] = log_pop
    columns["log_pr"] = np.log(get_participation_rate_qrtly().data) * 100
    labels["log_hours"] = "log hours (6202.0)"
    labels["log_pop"] = "log pop 15+ (6202.0)"
    labels["log_pr"] = "log participation (6202.0)"

    return columns, labels


def _align(columns: dict[str, pd.Series], start: str | None, end: str | None) -> pd.DataFrame:
    """Put the series on one quarterly index, drop partial rows, trim the sample."""
    df = pd.DataFrame(columns)

    # Narrow to a PeriodIndex so the quarterly comparisons below are well defined.
    period_index = df.index
    if not isinstance(period_index, pd.PeriodIndex):
        period_index = pd.PeriodIndex(period_index, freq="Q")
    df.index = period_index.asfreq("Q")
    df = df.dropna()

    if start:
        df = df.loc[df.index >= pd.Period(start, "Q")]
    if end:
        df = df.loc[df.index <= pd.Period(end, "Q")]

    if df.empty:
        raise ValueError(f"No observations remain for start={start!r}, end={end!r}")

    return df


def build_observations(
    start: str | None = "1993Q1",
    end: str | None = None,
    *,
    verbose: bool = False,
    smooth_pop: int = 7,
    spec: str = "core",
    pi_basis: str = "quarterly",
    supply_control: str | None = None,
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex, pd.DataFrame]:
    """Build observation arrays for potential_uc estimation.

    The core specification loads only log GDP and inflation. The labour
    specification additionally loads hours, population and participation, and
    the aligned sample is then the intersection of all five.

    Args:
        start: First quarter of the sample (default 1993Q1).
        end: Last quarter (default: latest available).
        verbose: Print per-series coverage and the aligned sample.
        smooth_pop: Henderson MA terms applied to log population (0 = off).
            Labour spec only. See ModelConfig.smooth_pop for why this is here.
        spec: "core" (Y and pi) or "labour" (adds the hours block).
        pi_basis: "quarterly" (annualised, non-overlapping) or "annual"
            (four-quarter, overlapping). See ModelConfig.pi_basis.
        supply_control: None, or "import_prices" to add a cost-push regressor
            to the Phillips curve. See ModelConfig.supply_control.

    Returns:
        Tuple of:
          - obs: dict of numpy arrays keyed by variable name
          - obs_index: the aligned PeriodIndex
          - chart_obs: DataFrame of the same series, for charting

    """
    columns, labels = _load_series(spec, smooth_pop, pi_basis, supply_control)

    if verbose:
        print(f"Input series coverage ({spec} specification):")
        for key, series in columns.items():
            clean = series.dropna()
            label = labels[key]
            print(f"  {label:<{_NAME_WIDTH}} {clean.index.min()} -> {clean.index.max()}  n={len(clean)}")

    df = _align(columns, start, end)

    # The supply control must enter the Phillips curve with mean zero. The
    # anchor is fixed at 2.5, so an undemeaned regressor with mean m would add
    # gamma·m to average predicted inflation and be absorbed by a permanent
    # shift in the output gap — i.e. it would move the anchor rather than
    # explain deviations from it. Demeaned over the estimation sample, after
    # trimming, so the mean removed is the one the likelihood actually sees.
    if "supply" in df.columns:
        df["supply"] = df["supply"] - df["supply"].mean()

    # log productivity, by the identity. Retained for charting and diagnostics;
    # the model derives it from the latents rather than reading it here.
    if spec == "labour":
        df["log_prod"] = df["log_gdp"] - df["log_hours"]

    if verbose:
        print(f"\nAligned sample: {df.index.min()} -> {df.index.max()}  n={len(df)}")

    obs = {col: df[col].to_numpy(dtype=float) for col in df.columns}

    obs_index = df.index
    if not isinstance(obs_index, pd.PeriodIndex):
        raise TypeError(f"Expected a PeriodIndex after alignment, got {type(obs_index).__name__}")

    return obs, obs_index, df
