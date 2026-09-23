"""Build the rstar_qpm observation frame.

Every series is published data. Nothing here is another model's output,
except that measured inflation expectations are themselves estimated upstream
in `src.data` and enter as data.
"""

import numpy as np
import pandas as pd

from src.data.aofm_loader import get_aofm_5y5y_forward
from src.data.cash_rate import get_cash_rate_qrtly
from src.data.debt_servicing import get_dsr_qrtly
from src.data.expectations_model import get_model_expectations_unanchored
from src.data.fred_loader import get_fred_quarterly
from src.data.gdp import get_log_gdp
from src.data.import_prices import get_import_price_index_qrtly
from src.data.inflation import get_trimmed_mean_annual, get_trimmed_mean_qrtly
from src.data.rba_loader import get_lending_rate
from src.data.twi import get_real_twi_qrtly
from src.models.common.sources import SourceSet
from src.models.rstar_qpm.config import (
    FORWARD_METHOD,
    KIM_WRIGHT_SERIES,
    LENDING_RATE,
    WORLD_REAL_SERIES,
    ModelConfig,
)

# Annualising a quarterly rate.
QUARTERS_PER_YEAR = 4

# Series that must be present for a quarter to enter the sample. The forward
# may be missing: its equation drops out of those quarters on its own.
REQUIRED = ("y", "q", "pi", "pi_lag", "pi4", "pie", "pie_lag", "i", "i_lag",
            "r", "r_lag", "rw", "rw_lag", "m4_lag", "d_dsr", "d_dsr_lag", "d_i", "exposure")


def _quarterly(series: pd.Series) -> pd.Series:
    """Return `series` on a quarterly PeriodIndex, averaging anything finer."""
    index = series.index
    if isinstance(index, pd.PeriodIndex) and index.freqstr.startswith("Q"):
        return series.astype(float)
    periods = pd.PeriodIndex(index, freq="Q")
    return series.groupby(periods).mean().astype(float)


def build_observations(config: ModelConfig, *, verbose: bool = False) -> tuple[pd.DataFrame, SourceSet]:
    """Return the observation frame and the sources behind it.

    Columns, all in per cent or 100 x log:
      y     log real GDP x 100
      q     log real TWI x 100, up = appreciation
      pi    quarterly trimmed mean, annualised
      pi4   year-ended trimmed mean
      pie   measured inflation expectations
      i     cash rate, quarterly average
      r     real cash rate, i - pie
      rw    world real rate: Cleveland Fed 10y expected real less Kim-Wright premium
      f     AOFM 5y5y risk-neutral forward, less pie (may be missing)
      m4    year-ended import price inflation, consumption goods
      d_dsr quarterly change in household interest payments / disposable income
      exposure  last quarter's DSR over the standard variable mortgage rate
      d_i   quarterly change in the cash rate
    plus `_lag` columns for the one-quarter lags the equations use.
    """
    sources = SourceSet()
    frame = pd.DataFrame({
        "y": _quarterly(sources.take(get_log_gdp())),
        "q": 100.0 * np.log(_quarterly(sources.take(get_real_twi_qrtly()))),
        "pi": QUARTERS_PER_YEAR * _quarterly(sources.take(get_trimmed_mean_qrtly())),
        "pi4": _quarterly(sources.take(get_trimmed_mean_annual())),
        "pie": _quarterly(sources.take(get_model_expectations_unanchored())),
        "i": _quarterly(sources.take(get_cash_rate_qrtly())),
    })

    world = get_fred_quarterly(WORLD_REAL_SERIES)
    sources.add("Federal Reserve Bank of Cleveland via FRED", WORLD_REAL_SERIES)
    premium = get_fred_quarterly(KIM_WRIGHT_SERIES)
    sources.add("Federal Reserve Board via FRED", KIM_WRIGHT_SERIES)
    frame["rw"] = _quarterly(world) - _quarterly(premium)

    forward = _quarterly(sources.take(get_aofm_5y5y_forward(FORWARD_METHOD)))
    frame["f"] = forward - frame["pie"]

    imports = _quarterly(sources.take(get_import_price_index_qrtly()))
    frame["m4"] = 100.0 * np.log(imports).diff(QUARTERS_PER_YEAR)

    # Cash flow. Debt servicing over the mortgage rate is debt over income, so
    # last quarter's ratio is the stock a change in the cash rate works on.
    dsr = _quarterly(sources.take(get_dsr_qrtly()))
    lending = _quarterly(sources.take(get_lending_rate(LENDING_RATE)))
    frame["d_dsr"] = dsr.diff()
    frame["exposure"] = (dsr / lending).shift(1)

    frame["r"] = frame["i"] - frame["pie"]
    for name in ("pi", "pie", "i", "r", "rw", "m4", "d_dsr"):
        frame[f"{name}_lag"] = frame[name].shift(1)
    frame["d_i"] = frame["i"] - frame["i_lag"]

    frame = frame.loc[config.start:]
    if config.end:
        frame = frame.loc[: config.end]
    # Trim the ragged end: the last quarter every required series reaches.
    complete = frame[list(REQUIRED)].notna().all(axis=1)
    last = complete[complete].index.max()
    frame = frame.loc[:last]
    if not frame[list(REQUIRED)].notna().all().all():
        gaps = frame[list(REQUIRED)].isna().sum()
        raise ValueError(f"missing values inside the sample: {gaps[gaps > 0].to_dict()}")

    if verbose:
        print(f"Sample: {frame.index.min()} to {frame.index.max()}  ({len(frame)} quarters)")
        print(f"  forward present in {int(frame['f'].notna().sum())} quarters")
        for name in ("y", "q", "pi", "pie", "i", "r", "rw", "f", "m4"):
            col = frame[name].dropna()
            print(f"  {name:4s} mean {col.mean():9.2f}  sd {col.std():6.2f}  last {col.iloc[-1]:9.2f}")

    return frame, sources
