"""Observation matrix building for the HLW Bayesian r-star model.

Loads and aligns the series needed by the IS curve and Phillips curve:

- log GDP (level, log x 100) for the output gap
- Cash rate (annualised %) for the IS curve real rate
- Model-derived unanchored inflation expectations (annualised %)
- Annual trimmed mean inflation (annualised %) — Phillips curve LHS
- Quarterly trimmed mean inflation (%) — kept for diagnostics

The unanchored model expectations series is used both for the IS-curve real
rate and for the Phillips-curve anchor: it is continuous across the 1980s
disinflation and represents what agents actually expected (rather than a
target-counterfactual).
"""

import numpy as np
import pandas as pd

from src.data.bonds import get_indexed_yield
from src.data.cash_rate import get_cash_rate_qrtly
from src.data.expectations_model import get_model_expectations_unanchored
from src.data.gdp import get_log_gdp
from src.data.gov_spending import get_fiscal_impulse_lagged_qrtly
from src.data.inflation import get_trimmed_mean_annual, get_trimmed_mean_qrtly

_NAME_WIDTH = 20


def build_observations(
    start: str | None = "1980Q1",
    end: str | None = None,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex, pd.DataFrame]:
    """Build observation arrays for HLW estimation.

    Args:
        start: Start period (default 1980Q1, matching NAIRU model)
        end: End period (default: latest available)
        verbose: Print per-series ranges and the aligned sample

    Returns:
        Tuple of:
          - obs: dict of numpy arrays keyed by variable name
          - obs_index: aligned PeriodIndex
          - chart_obs: DataFrame containing the same series for charting

    """
    log_gdp = get_log_gdp().data
    cash_rate = get_cash_rate_qrtly().data
    pi_exp = get_model_expectations_unanchored().data
    pi_q = get_trimmed_mean_qrtly().data
    pi_4 = get_trimmed_mean_annual().data
    fiscal_impulse_1 = get_fiscal_impulse_lagged_qrtly().data
    indexed_10y = get_indexed_yield().data  # 10y inflation-linked bond yield (real)

    df = pd.DataFrame({
        "log_gdp": log_gdp,
        "cash_rate": cash_rate,
        "pi_exp": pi_exp,
        "pi_q": pi_q,
        "pi_4": pi_4,
        "fiscal_impulse_1": fiscal_impulse_1,
        "indexed_10y": indexed_10y,
    })
    df.index = df.index.asfreq("Q")
    df = df.dropna()

    if start:
        df = df.loc[df.index >= pd.Period(start, "Q")]
    if end:
        df = df.loc[df.index <= pd.Period(end, "Q")]

    # Linear regression of year-on-year GDP growth over the filtered sample,
    # used as a soft anchor for trend growth g. Computed *after* filtering so
    # the slope/intercept reflect the model period only.
    yoy_growth_full = get_log_gdp().data.diff(4)
    yoy_in_sample = yoy_growth_full.reindex(df.index).dropna()
    t = np.arange(len(yoy_in_sample))
    slope, intercept = np.polyfit(t, yoy_in_sample.values, 1)
    linear_trend = pd.Series(intercept + slope * t, index=yoy_in_sample.index)
    df["trend_growth_obs"] = linear_trend.reindex(df.index)
    df = df.dropna()

    if verbose:
        print(f"\nObservation sample: {df.index[0]} to {df.index[-1]} ({len(df)} periods)")
        for col in df.columns:
            print(f"  {col:<{_NAME_WIDTH}}: [{df[col].min():.2f}, {df[col].max():.2f}]")
        print(f"  linear g-anchor:     slope {slope * 4:+.3f} pp/year, "
              f"{linear_trend.iloc[0]:.2f}% -> {linear_trend.iloc[-1]:.2f}%")

    obs = {col: df[col].to_numpy() for col in df.columns}

    return obs, df.index, df
