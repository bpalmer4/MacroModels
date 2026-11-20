"""Shared core helpers for the GDP nowcast suite.

Small pieces every GDP nowcast model needs, previously copy-pasted across the
bridge/DFM/BVAR/components modules:

  * :func:`compute_tty`          — Q/Q nowcast → through-the-year (annual) growth
  * :func:`nowcast_intervals`    — Q/Q and TTY intervals around a normal Q/Q nowcast
  * :func:`value_at`             — a series' value at a Period, typed as float
  * :func:`compute_gdp_growth`   — CVM level → Q/Q log-difference growth (%)
  * :func:`detect_target_quarter`— next unpublished quarter from a GDP series
  * :func:`truncate_monthly`     — clip a monthly series to an as-of cutoff
  * :func:`print_qoq_tty_header` — the shared Q/Q + TTY summary block

Chart code lives in ``nowcast_charts``; cross-model diagnostics in
``nowcast_diagnostics``.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

# Two-sided normal quantiles for the 70%, 80% and 90% nowcast intervals (mean ± Z·sigma).
Z_70 = 1.04
Z_80 = 1.282
Z_90 = 1.645


def compute_tty(qoq: float, gdp: pd.Series, target_quarter: pd.Period) -> float:
    """Through-the-year (annual) growth implied by a single Q/Q nowcast.

    Rolls the last published GDP level forward by the Q/Q nowcast, then compares
    it to the level four quarters earlier: ``(GDP_T / GDP_{T-4} - 1) × 100``.

    Both reference levels are selected by LABEL (``target_quarter - 1`` and
    ``target_quarter - 4``), not by position. Callers pass a GDP series ending at
    ``target_quarter - 1`` (the next-unpublished-quarter invariant), so positional
    ``iloc[-1]`` / ``iloc[-4]`` coincide today — but they would silently return the
    wrong quarter if the series ever extended past the target. Fail loudly instead if
    the required quarters are absent.
    """
    prev_q = target_quarter - 1
    q_minus_4 = target_quarter - 4
    if prev_q not in gdp.index or q_minus_4 not in gdp.index:
        msg = (
            f"compute_tty: GDP series missing required quarters for target {target_quarter} "
            f"(need {prev_q} and {q_minus_4}; index spans {gdp.index[0]}..{gdp.index[-1]})."
        )
        raise KeyError(msg)
    projected = value_at(gdp, prev_q) * np.exp(qoq / 100)
    return float((projected / value_at(gdp, q_minus_4) - 1) * 100)


def value_at(series: pd.Series, period: pd.Period) -> float:
    """Value of ``series`` at ``period``, as a float.

    Goes via the integer position rather than ``series.loc[period]``: pandas-stubs
    types a ``Series`` generically, so a ``Period`` key is not an accepted
    overload. ``get_loc`` returns an int for a unique index; anything else means
    the caller handed us a duplicated or unsorted index, which is a bug.
    """
    loc = series.index.get_loc(period)
    if not isinstance(loc, int):
        msg = f"value_at: {period} is not a unique entry in the series index"
        raise KeyError(msg)
    return float(series.to_numpy()[loc])


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


@dataclass(frozen=True)
class NowcastIntervals:
    """Q/Q intervals around a normal Q/Q nowcast, and the same bounds as annual (TTY) growth."""

    qoq_70: tuple[float, float]
    qoq_80: tuple[float, float]
    qoq_90: tuple[float, float]
    tty_70: tuple[float, float]
    tty_80: tuple[float, float]
    tty_90: tuple[float, float]


def nowcast_intervals(
    qoq: float, sigma: float, gdp: pd.Series, target_quarter: pd.Period,
) -> NowcastIntervals:
    """Build ``qoq ± Z·sigma`` intervals and roll each bound through :func:`compute_tty`."""
    qoq_70, qoq_80, qoq_90 = ((qoq - z * sigma, qoq + z * sigma) for z in (Z_70, Z_80, Z_90))
    tty_70, tty_80, tty_90 = (
        (compute_tty(lo, gdp, target_quarter), compute_tty(hi, gdp, target_quarter))
        for lo, hi in (qoq_70, qoq_80, qoq_90)
    )
    return NowcastIntervals(
        qoq_70=qoq_70, qoq_80=qoq_80, qoq_90=qoq_90,
        tty_70=tty_70, tty_80=tty_80, tty_90=tty_90,
    )


class QoqTtyResult(Protocol):
    """Structural type for a nowcast result carrying Q/Q + TTY point + intervals."""

    target_quarter: pd.Period
    gdp_qoq: float
    gdp_tty: float
    intervals: NowcastIntervals


def print_qoq_tty_header(result: QoqTtyResult, model_tag: str = "") -> None:
    """Print the shared summary header: title rule + Q/Q and TTY point + 70/90 CIs.

    ``model_tag`` is the bracketed model name in the title (e.g. ``"BVAR"``); pass
    ``""`` for no tag. Callers print their model-specific detail and footer after.
    """
    tag = f" ({model_tag})" if model_tag else ""
    print("\n" + "=" * 70)
    print(f"  GDP NOWCAST{tag}: {result.target_quarter}")
    print("=" * 70)

    ci = result.intervals
    print(f"\n  Q/Q growth:   {result.gdp_qoq:+.2f}%")
    print(f"    70% CI:     [{ci.qoq_70[0]:+.2f}%, {ci.qoq_70[1]:+.2f}%]")
    print(f"    90% CI:     [{ci.qoq_90[0]:+.2f}%, {ci.qoq_90[1]:+.2f}%]")

    print(f"\n  TTY growth:   {result.gdp_tty:+.2f}%")
    print(f"    70% CI:     [{ci.tty_70[0]:+.2f}%, {ci.tty_70[1]:+.2f}%]")
    print(f"    90% CI:     [{ci.tty_90[0]:+.2f}%, {ci.tty_90[1]:+.2f}%]")
