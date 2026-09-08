"""The headline CPI reaching back to 1948, for models that need the long run.

`inflation.py` serves the models that start in the inflation-targeting era, and
its headline series comes from the seasonally adjusted analytical table
(64010Appendix1a), which begins in 1987. Anything asking about the 1950s and
1960s needs the Original quarterly table instead.

The index is rebuilt rather than read. The published index is rounded to few
significant figures in the early years, which puts visible steps into any growth
computed off it, while the published quarterly percentage change is finer. So
the change is chained into a relative index and rebased onto the published level
at the latest quarter the two share. The method is `abs_prices._get_cpi_headline`
from the ABS notebooks, ported here so this package does not reach outside
itself for data.

Original rather than seasonally adjusted: no SA version of the All groups index
runs back this far, and year-ended growth is what the long-run models use, which
is largely insensitive to seasonality anyway.
"""

from functools import cache

import pandas as pd
import readabs as ra
from readabs import metacol as mc

from src.data.dataseries import DataSeries

# Table 6401017: All groups CPI, Australia, Original, quarterly. Selected by
# data-item description rather than series ID, which the ABS changes.
_TABLE = "6401017"
_QOQ_DID = "Percentage Change from Previous Period ;  All groups CPI ;  Australia ;"
_INDEX_DID = "Index Numbers ;  All groups CPI ;  Australia ;"

_PER_CENT = 100.0
_QUARTERS_PER_YEAR = 4


@cache
def get_long_cpi_index() -> DataSeries:
    """Return the reconstructed headline CPI index, quarterly, back to 1948Q4."""
    data, meta = ra.read_abs_cat("6401.0", single_excel_only=_TABLE, verbose=False)
    base = {_TABLE: mc.table, "Original": mc.stype, "Quarter": mc.freq}
    _table, qoq_id, _units = ra.find_abs_id(
        meta, base | {_QOQ_DID: mc.did, "Percent": mc.unit}, verbose=False,
    )
    _table, index_id, _units = ra.find_abs_id(
        meta, base | {_INDEX_DID: mc.did, "Index Numbers": mc.unit}, verbose=False,
    )

    qoq = data[_TABLE][qoq_id].dropna() / _PER_CENT
    published = data[_TABLE][index_id].dropna()
    relative = (1.0 + qoq).cumprod()
    anchor = relative.index.intersection(published.index)[-1]
    index = relative / relative.loc[anchor] * published.loc[anchor]

    return DataSeries(
        data=index.rename("Headline CPI (reconstructed)"),
        source="ABS",
        units="Index Numbers",
        description="All groups CPI, Australia, Original, chained from the quarterly change",
        cat="6401.0",
        table=_TABLE,
        series_id=qoq_id,
        stype="Original",
    )


@cache
def get_long_headline_annual() -> DataSeries:
    """Return year-ended headline CPI inflation, per cent, back to 1949Q4."""
    index = get_long_cpi_index()
    growth = (index.data / index.data.shift(_QUARTERS_PER_YEAR) - 1.0) * _PER_CENT
    return DataSeries(
        data=growth.dropna().rename("Headline CPI, year-ended"),
        source="ABS",
        units="Per cent",
        description="Year-ended growth in the reconstructed All groups CPI, Original",
        cat="6401.0",
        table=_TABLE,
        series_id=index.series_id,
        stype="Original",
    )


def _as_quarterly(series: pd.Series) -> pd.Series:
    """Return `series` on a quarterly PeriodIndex, for callers holding raw frames."""
    if isinstance(series.index, pd.PeriodIndex) and series.index.freqstr.startswith("Q"):
        return series
    return series.set_axis(pd.PeriodIndex(series.index, freq="Q"))
