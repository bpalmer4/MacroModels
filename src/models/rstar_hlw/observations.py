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

from typing import Literal

import numpy as np
import pandas as pd

from src.data.bonds import get_indexed_yield_filled
from src.data.cash_rate import get_cash_rate_qrtly
from src.data.commodity_prices import get_icp_aud_change_lagged_qrtly
from src.data.expectations_model import get_model_expectations_unanchored
from src.data.gdp import get_log_gdp
from src.data.gov_spending import get_fiscal_impulse_lagged_qrtly
from src.data.inflation import get_trimmed_mean_annual, get_trimmed_mean_qrtly
from src.data.tot import get_tot_change_qrtly
from src.data.twi import get_twi_change_lagged_qrtly

_NAME_WIDTH = 20

# Sample start. 1993Q1 is the inflation-targeting era, and the change from the
# old 1980Q1 default (which the indexed bond yield pinned to an effective
# 1986Q3) is not cosmetic:
#
#   output gap at 2026Q2   +4.39 on 1986Q3   ->   +2.56 on 1993Q1
#   gap change 2022Q4-2026Q2  -0.56          ->   -1.87   (Okun implies -1.71)
#   b_y                       0.169          ->    0.273
#
# On the longer sample the gap barely moved through the 2022-26 disinflation,
# which unemployment flatly contradicts. On this one it closes at about the
# rate Okun implies and lands just above the plausible range. The Phillips
# slope nearly doubling is the clue to why: the pre-1993 quarters have no
# inflation target, and asking one b_y to span that regime change flattens it.
#
# Pass `--start 1980Q1` to reproduce the older runs, including the sweeps.
DEFAULT_START = "1993Q1"

# Quarters in the trailing window behind the `cagr40` g-anchor. Ten years:
# long enough that one recession cannot set the level, and the cost is that a
# window this long dates a change in trend about five years late, which is
# what being one-sided buys and what it charges.
_CAGR_WINDOW_QTRS = 40

GAnchor = Literal["linear", "cagr40"]
DEFAULT_G_ANCHOR: GAnchor = "linear"

# How each anchor is named on a chart. Built from `_CAGR_WINDOW_QTRS` so the
# label cannot claim a window the series does not use.
G_ANCHOR_LABELS: dict[str, str] = {
    "linear": "linear trend of YoY growth",
    "cagr40": f"{_CAGR_WINDOW_QTRS}q trailing CAGR",
}


def _g_anchor(
    kind: GAnchor,
    log_gdp: pd.Series,
    index: pd.PeriodIndex,
) -> tuple[pd.Series, str]:
    """Build the soft anchor on trend growth, annualised %, plus a description.

    `linear` regresses year-on-year growth on time across the estimation
    sample. A single shock barely moves a regression line, which is why it
    replaced an HMA that had the 2020 dip baked into it. The costs are that it
    is monotone by construction, so it cannot show the growth slowdown
    pausing, and that it is fitted on quarters that come after each date it
    describes.

    `cagr40` is the trailing compound annual growth rate over
    `_CAGR_WINDOW_QTRS`. It uses only data up to each date and is free to
    flatten out. It is computed on the whole GDP series rather than the
    filtered sample, so the window is already full in the sample's first
    quarter and no warm-up is lost.
    """
    if kind == "linear":
        yoy = log_gdp.diff(4).reindex(index).dropna()
        t = np.arange(len(yoy))
        slope, intercept = np.polyfit(t, yoy.to_numpy(dtype=float), 1)
        anchor = pd.Series(intercept + slope * t, index=yoy.index).reindex(index)
        desc = (
            f"linear trend: slope {slope * 4:+.3f} pp/year, "
            f"{anchor.iloc[0]:.2f}% -> {anchor.iloc[-1]:.2f}%"
        )
        return anchor, desc

    k = _CAGR_WINDOW_QTRS
    # log_gdp is log x 100, so a k-quarter log difference scaled by 4/k is
    # already an annualised percentage growth rate.
    anchor = ((log_gdp - log_gdp.shift(k)) * 4.0 / k).reindex(index)
    if anchor.isna().any():
        missing = index[anchor.isna()]
        raise ValueError(
            f"a {k}-quarter trailing window needs GDP back to {k} quarters before "
            f"{index[0]}, and {len(missing)} quarters have none "
            f"({missing[0]} to {missing[-1]}). Start the sample later or shorten "
            f"the window",
        )
    desc = (
        f"{k}q trailing CAGR: {anchor.iloc[0]:.2f}% -> {anchor.iloc[-1]:.2f}%, "
        f"range {anchor.min():.2f} to {anchor.max():.2f}"
    )
    return anchor, desc


def build_observations(
    start: str | None = DEFAULT_START,
    end: str | None = None,
    verbose: bool = False,
    g_anchor: GAnchor = DEFAULT_G_ANCHOR,
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex, pd.DataFrame]:
    """Build observation arrays for HLW estimation.

    Args:
        start: Start period (default 1980Q1, matching NAIRU model)
        end: End period (default: latest available)
        verbose: Print per-series ranges and the aligned sample
        g_anchor: Which soft anchor on trend growth to build, `linear` or
            `cagr40`. Only the resolutions that wire the anchor in consume it;
            A and B drop the series. See `_g_anchor`.

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
    # 10y inflation-linked bond yield (real); the 2013Q3-2014Q3 gap is filled
    # with nominal 10y less interpolated breakeven.
    indexed_10y = get_indexed_yield_filled().data
    # SOE-block IS-curve regressors (used by Resolution D)
    tot_change_1 = get_tot_change_qrtly().data.shift(1)  # quarterly ToT % change, lag 1
    twi_change_1 = get_twi_change_lagged_qrtly().data    # quarterly TWI change (lag 1)
    icp_change_1 = get_icp_aud_change_lagged_qrtly().data  # RBA ICP A$ change (lag 1)

    df = pd.DataFrame({
        "log_gdp": log_gdp,
        "cash_rate": cash_rate,
        "pi_exp": pi_exp,
        "pi_q": pi_q,
        "pi_4": pi_4,
        "fiscal_impulse_1": fiscal_impulse_1,
        "indexed_10y": indexed_10y,
        "tot_change_1": tot_change_1,
        "twi_change_1": twi_change_1,
        "icp_change_1": icp_change_1,
    })
    # Every input series is quarterly, so this is a re-labelling rather than a
    # conversion. Checked rather than assumed: a non-period index here would
    # otherwise fail much later, inside the model.
    if not isinstance(df.index, pd.PeriodIndex):
        raise TypeError(f"expected a PeriodIndex after joining the series, got {type(df.index)}")
    df.index = df.index.asfreq("Q")
    df = df.dropna()

    if start:
        df = df.loc[df.index >= pd.Period(start, "Q")]
    if end:
        df = df.loc[df.index <= pd.Period(end, "Q")]

    # Soft anchor for trend growth g, used by the resolutions that wire it in.
    anchor, anchor_desc = _g_anchor(g_anchor, get_log_gdp().data, df.index)
    df["trend_growth_obs"] = anchor
    df = df.dropna()

    if verbose:
        print(f"\nObservation sample: {df.index[0]} to {df.index[-1]} ({len(df)} periods)")
        for col in df.columns:
            print(f"  {col:<{_NAME_WIDTH}}: [{df[col].min():.2f}, {df[col].max():.2f}]")
        print(f"  g-anchor ({g_anchor}): {anchor_desc}")

    obs = {col: df[col].to_numpy() for col in df.columns}

    obs_index = df.index
    if not isinstance(obs_index, pd.PeriodIndex):
        raise TypeError(f"sample index is no longer a PeriodIndex, got {type(obs_index)}")

    return obs, obs_index, df
