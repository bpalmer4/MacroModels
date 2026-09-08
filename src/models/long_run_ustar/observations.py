"""The two series the long-run reading needs, and where they came from."""

import pandas as pd

from src.data.labour_force import get_unemployment_rate_qrtly
from src.data.long_cpi import get_long_headline_annual
from src.models.common.sources import SourceSet
from src.models.long_run_ustar.config import ModelConfig


def build_observations(config: ModelConfig) -> tuple[pd.DataFrame, SourceSet]:
    """Return a quarterly frame of `pi` and `u`, aligned, and its sources.

    Headline rather than trimmed mean, because the trimmed mean begins in 1983
    and the question is about the decades before that. The cost is that supply
    shocks are in the measure: see MODEL_NOTES on what that does to the 1970s.
    """
    sources = SourceSet()
    columns = {
        "pi": sources.take(get_long_headline_annual(), "headline CPI, year-ended", key="pi"),
        "u": sources.take(get_unemployment_rate_qrtly(), "unemployment rate", key="u"),
    }

    frame = pd.DataFrame(columns).dropna()
    if not isinstance(frame.index, pd.PeriodIndex):
        raise TypeError("both series must carry a quarterly PeriodIndex")

    if config.start is not None:
        frame = frame.loc[frame.index >= pd.Period(config.start, freq="Q")]
    if config.end is not None:
        frame = frame.loc[frame.index <= pd.Period(config.end, freq="Q")]

    if frame.empty:
        raise ValueError("no overlapping quarters of inflation and unemployment")

    return frame, sources
