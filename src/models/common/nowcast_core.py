"""Shared core helpers for the GDP nowcast suite.

Small pieces every GDP nowcast model needs, previously copy-pasted across the
bridge/DFM/BVAR/components modules:

  * :func:`compute_tty`          — Q/Q nowcast → through-the-year (annual) growth
  * :func:`compute_gdp_growth`   — CVM level → Q/Q log-difference growth (%)
  * :func:`detect_target_quarter`— next unpublished quarter from a GDP series
  * :func:`truncate_monthly`     — clip a monthly series to an as-of cutoff
  * :func:`print_qoq_tty_header` — the shared Q/Q + TTY summary block

Chart code lives in ``nowcast_charts``; cross-model diagnostics in
``nowcast_diagnostics``.
"""

from typing import Protocol

import numpy as np
import pandas as pd


def compute_tty(qoq: float, gdp: pd.Series, target_quarter: pd.Period) -> float:
    """Through-the-year (annual) growth implied by a single Q/Q nowcast.

    Rolls the last published GDP level forward by the Q/Q nowcast, then compares
    it to the level four quarters earlier: ``(GDP_T / GDP_{T-4} - 1) × 100``.
    """
    last_gdp_level = gdp.iloc[-1]
    projected = last_gdp_level * np.exp(qoq / 100)
    q_minus_4 = target_quarter - 4
    gdp_4q_ago = gdp.loc[q_minus_4] if q_minus_4 in gdp.index else gdp.iloc[-4]
    return float((projected / gdp_4q_ago - 1) * 100)


def compute_gdp_growth(gdp: pd.Series) -> pd.Series:
    """Q/Q GDP growth as the log difference of a CVM level × 100."""
    return np.log(gdp).diff(1) * 100


def detect_target_quarter(gdp: pd.Series) -> pd.Period:
    """Next unpublished GDP quarter (the quarter after the last published one)."""
    last_published = gdp.dropna().index[-1]
    return last_published + 1


def truncate_monthly(series: pd.Series, cutoff: pd.Period | None) -> pd.Series:
    """Truncate a monthly series to the data available through ``cutoff``."""
    if cutoff is None:
        return pd.Series(dtype=float)
    return series.loc[series.index <= cutoff].copy()


class QoqTtyResult(Protocol):
    """Structural type for a nowcast result carrying Q/Q + TTY point + intervals."""

    target_quarter: pd.Period
    gdp_qoq: float
    gdp_tty: float
    gdp_qoq_70: tuple[float, float]
    gdp_qoq_90: tuple[float, float]
    gdp_tty_70: tuple[float, float]
    gdp_tty_90: tuple[float, float]


def print_qoq_tty_header(result: QoqTtyResult, model_tag: str = "") -> None:
    """Print the shared summary header: title rule + Q/Q and TTY point + 70/90 CIs.

    ``model_tag`` is the bracketed model name in the title (e.g. ``"BVAR"``); pass
    ``""`` for no tag. Callers print their model-specific detail and footer after.
    """
    tag = f" ({model_tag})" if model_tag else ""
    print("\n" + "=" * 70)
    print(f"  GDP NOWCAST{tag}: {result.target_quarter}")
    print("=" * 70)

    print(f"\n  Q/Q growth:   {result.gdp_qoq:+.2f}%")
    print(f"    70% CI:     [{result.gdp_qoq_70[0]:+.2f}%, {result.gdp_qoq_70[1]:+.2f}%]")
    print(f"    90% CI:     [{result.gdp_qoq_90[0]:+.2f}%, {result.gdp_qoq_90[1]:+.2f}%]")

    print(f"\n  TTY growth:   {result.gdp_tty:+.2f}%")
    print(f"    70% CI:     [{result.gdp_tty_70[0]:+.2f}%, {result.gdp_tty_70[1]:+.2f}%]")
    print(f"    90% CI:     [{result.gdp_tty_90[0]:+.2f}%, {result.gdp_tty_90[1]:+.2f}%]")
