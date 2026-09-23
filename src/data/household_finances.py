"""Household balance sheet ratios from RBA table E2.

E2 publishes household debt and assets as ratios to annualised household
disposable income, quarterly from 1977. It does NOT publish interest paid as
a ratio to income, which is what a debt serviceability ratio needs, so that
has to be built: interest over income is debt over income multiplied by the
interest rate, and E2 supplies the first half.

Sourced from ABS national accounts except where noted; the RBA republishes
them on a common basis, which is why they are taken from here rather than
assembled from the ABS series directly.
"""

from functools import cache

import pandas as pd

from src.data.dataseries import DataSeries

E2_URL = "https://www.rba.gov.au/statistics/tables/xls/e02hist.xlsx"

# E2's header block: eleven rows of metadata, with the series identifiers on
# the last of them and the observations beginning after.
_HEADER_ROWS = 11
_SERIES_ID_ROW = 10

# The two debt measures that can anchor a serviceability ratio. Housing debt
# is the narrower and more literal reading of "mortgage serviceability";
# total household debt also carries personal and credit card borrowing, and
# is what reproduces the figures CBA publish.
DEBT_SERIES = {
    "total": "BHFDDIT",
    "housing": "BHFDDIH",
    "owner_occupier": "BHFDDIO",
}


@cache
def _load_e2() -> pd.DataFrame:
    """Return E2 as a quarterly frame indexed by period, columns by series ID."""
    header = pd.read_excel(E2_URL, sheet_name=0, header=None, nrows=_HEADER_ROWS)
    series_ids = [str(x) for x in header.iloc[_SERIES_ID_ROW].tolist()]

    frame = pd.read_excel(
        E2_URL, sheet_name=0, header=None, skiprows=_HEADER_ROWS, index_col=0,
    )
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[frame.index.notna()]
    frame.columns = series_ids[1:]
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    return frame


def get_debt_to_income(measure: str = "total") -> DataSeries:
    """Return household debt as a per cent of annualised disposable income.

    `measure` picks which debt: `total`, `housing` or `owner_occupier`. The
    choice moves a serviceability ratio by two to three percentage points, so
    it is a switch rather than a default buried in the code.
    """
    if measure not in DEBT_SERIES:
        raise ValueError(
            f"measure must be one of {sorted(DEBT_SERIES)}, got {measure!r}",
        )

    series_id = DEBT_SERIES[measure]
    frame = _load_e2()
    if series_id not in frame.columns:
        raise KeyError(
            f"{series_id} is not in RBA E2; columns are {list(frame.columns)}",
        )

    data = pd.to_numeric(frame[series_id], errors="coerce").dropna().astype(float)
    return DataSeries(
        data=data,
        source="RBA",
        units="Per cent of annualised household disposable income",
        description=f"{measure.replace('_', ' ')} debt to income",
        table="E2",
        series_id=series_id,
    )
