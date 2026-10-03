"""Labour force status by age and sex, monthly, from the 6202.0 LMS2 datacube.

LMS2 is a long-format datacube (Original terms only), not an ABS time-series
spreadsheet, so readabs' `grab_abs_url` returns its sheets raw and the table
is parsed here. It replaced 6291.0.55.001 LM1 in April 2026.
"""

from functools import cache

import pandas as pd
import readabs as ra

CAT = "6202.0"
TABLE = "LMS2"
DATA_SHEET_SUFFIX = "Data 1"
HEADER_MARKER = "Month"  # first cell of the header row, below the ABS banner

# Status columns, found by the start of their header text ('000 persons).
STATUS_COLUMNS = {
    "eft": "Employed full-time",
    "ept": "Employed part-time",
    "uft": "Unemployed looked for full-time",
    "upt": "Unemployed looked for only part-time",
    "nilf": "Not in the labour force",
}


@cache
def get_labour_force_status_by_age() -> pd.DataFrame:
    """Return labour force status ('000) indexed by (month, sex, age).

    Columns: eft, ept (employed full/part-time), uft, upt (unemployed looking
    for full/part-time work), nilf (not in the labour force). Summed over the
    marital status and region dimensions of the cube.
    """
    sheets = ra.grab_abs_url(cat=CAT, single_excel_only=TABLE)
    data = next((v for k, v in sheets.items() if k.endswith(DATA_SHEET_SUFFIX)), None)
    if data is None:
        msg = f"{TABLE}: no '{DATA_SHEET_SUFFIX}' sheet found"
        raise ValueError(msg)

    first_col = data.iloc[:, 0].astype(str)
    header_rows = data.index[first_col == HEADER_MARKER]
    if len(header_rows) == 0:
        msg = f"{TABLE}: header row starting '{HEADER_MARKER}' not found"
        raise ValueError(msg)
    header = data.loc[header_rows[0]]
    body = data.loc[header_rows[0] + 1 :]

    picked: dict[str, pd.Series] = {}
    for name, label in [("month", "Month"), ("sex", "Sex"), ("age", "Age")]:
        picked[name] = body.iloc[:, _column(header, label)]
    for name, label in STATUS_COLUMNS.items():
        picked[name] = pd.to_numeric(body.iloc[:, _column(header, label)], errors="coerce")
    frame = pd.DataFrame(picked).dropna(subset=["sex"])
    frame["month"] = pd.PeriodIndex(pd.to_datetime(frame["month"]), freq="M")
    return frame.groupby(["month", "sex", "age"])[list(STATUS_COLUMNS)].sum()


def _column(header: pd.Series, label: str) -> int:
    """Position of the single header cell starting with label."""
    # Blank header cells come through as NaN (astype(str) keeps NaN under pandas 3).
    hits = [i for i, text in enumerate(header) if isinstance(text, str) and text.startswith(label)]
    if len(hits) != 1:
        msg = f"{TABLE}: expected one column starting '{label}', found {len(hits)}"
        raise ValueError(msg)
    return hits[0]
