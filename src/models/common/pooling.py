"""Pool posterior draws from several sources into one set of quantiles, each source weighted equally."""

from collections.abc import Mapping
from functools import reduce

import numpy as np
import pandas as pd

# The combined estimate's median and central bands, as quantiles of the pool.
COMBINED_MEDIAN = 0.50
BAND_50 = (0.25, 0.75)
BAND_90 = (0.05, 0.95)


def combine(draws: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return quantiles of the pooled draws, on the quarters every source covers.

    Each source carries equal weight however many draws it holds, so a draw
    weighs 1 / (sources x that source's draws). A quantile is the first pooled
    value whose cumulative weight reaches it.

    Args:
        draws: one frame per source, one row per quarter and one column per draw

    Returns:
        One row per quarter, one column per quantile, named by that quantile.

    """
    common = reduce(lambda left, right: left.intersection(right), (frame.index for frame in draws.values()))
    values = np.hstack([frame.loc[common].to_numpy() for frame in draws.values()])
    weights = np.concatenate([np.full(frame.shape[1], 1 / (len(draws) * frame.shape[1])) for frame in draws.values()])

    order = np.argsort(values, axis=1)
    ordered = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    rows = np.arange(len(common))
    quantiles = (BAND_90[0], BAND_50[0], COMBINED_MEDIAN, BAND_50[1], BAND_90[1])
    return pd.DataFrame(
        {quantile: ordered[rows, (cumulative >= quantile).argmax(axis=1)] for quantile in quantiles},
        index=common,
    )
