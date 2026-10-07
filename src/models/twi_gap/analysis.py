"""Fit the real TWI on US$ commodity prices and measure the gap."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from src.data.commodity_prices import get_icp_usd_qrtly
from src.data.twi import get_real_twi_qrtly
from src.models.twi_gap import DARK_SD, NEUTRAL_SD, ROBUSTNESS_STARTS, START

PERCENT = 100


@dataclass(frozen=True)
class GapFit:
    """The fitted equation and its residual.

    gap is the residual in per cent (log points x 100): the real TWI above (+)
    or below (-) the level commodity prices explain. eg_p is the Engle-Granger
    cointegration p-value: small means the two share a long-run path, so the
    gap returns towards zero.
    """

    gap: pd.Series
    alpha: float
    beta: float
    beta_se: float
    r2: float
    sd: float
    eg_p: float


def build_frame() -> pd.DataFrame:
    """Return the quarterly frame: rtwi and icp, both index levels."""
    frame = pd.DataFrame(
        {
            "rtwi": get_real_twi_qrtly().data,
            "icp": get_icp_usd_qrtly().data,
        }
    ).sort_index()
    if not isinstance(frame.index, pd.PeriodIndex):
        raise TypeError("frame must have a PeriodIndex")
    return frame


def fit_gap(frame: pd.DataFrame, start: pd.Period = START) -> GapFit:
    """Regress log real TWI on log commodity prices from start, by OLS."""
    logs = np.log(frame[["rtwi", "icp"]])
    if not isinstance(logs, pd.DataFrame):
        raise TypeError("log of the frame is not a DataFrame")
    logs = logs.dropna()
    logs = logs[logs.index >= start]
    if logs.empty:
        raise ValueError(f"no overlapping real TWI and commodity price data from {start}")
    fit = sm.OLS(logs["rtwi"], sm.add_constant(logs["icp"])).fit()
    gap = fit.resid * PERCENT
    _, eg_p, _ = coint(logs["rtwi"], logs["icp"])
    return GapFit(
        gap=gap,
        alpha=float(fit.params["const"]),
        beta=float(fit.params["icp"]),
        beta_se=float(fit.bse["icp"]),
        r2=float(fit.rsquared),
        sd=float(gap.std()),
        eg_p=float(eg_p),
    )


def robustness(frame: pd.DataFrame) -> pd.DataFrame:
    """Refit from each of ROBUSTNESS_STARTS, to show whether the relationship holds whatever the start."""
    rows = {}
    for year in ROBUSTNESS_STARTS:
        fit = fit_gap(frame, pd.Period(f"{year}Q1", "Q"))
        rows[year] = {
            "first": str(fit.gap.index[0]),
            "beta": fit.beta,
            "r2": fit.r2,
            "sd": fit.sd,
            "eg_p": fit.eg_p,
            "gap_now": fit.gap.iloc[-1],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def band_shares(fit: GapFit) -> dict[float, float]:
    """Return the share of quarters within NEUTRAL_SD and within DARK_SD of zero."""
    z = (fit.gap / fit.sd).abs()
    return {band: float((z < band).mean()) for band in (NEUTRAL_SD, DARK_SD)}
