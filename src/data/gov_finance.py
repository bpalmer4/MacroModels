"""Government finance statistics data loading.

Provides quarterly government spending from ABS Government Finance Statistics,
Australia (formerly 5519.0). Published ~2 weeks before GDP, covering government
final consumption expenditure and gross fixed capital formation.

The ABS publishes GFS as a data cube (not standard time series format), so this
module parses the Excel workbook directly rather than using readabs metadata search.
"""

import logging
import re
from functools import cache
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import requests

from src.data.dataseries import DataSeries

logger = logging.getLogger(__name__)

GFS_URL = (
    "https://www.abs.gov.au/statistics/economy/government/"
    "government-finance-statistics-australia/latest-release"
)
_CACHE_DIR = Path(".readabs_cache")
_CACHE_FILE = _CACHE_DIR / "gfs_quarterly.xlsx"

# Quarter name → pd.Period mapping
_QTR_MAP = {"Mar": 3, "Jun": 6, "Sep": 9, "Dec": 12}


def _parse_quarter(label: str) -> pd.Period:
    """Parse 'Dec Qtr 2025' → pd.Period('2025Q4', 'Q-DEC')."""
    match = re.match(r"(Mar|Jun|Sep|Dec)\s+Qtr\s+(\d{4})", label)
    if not match:
        msg = f"Cannot parse quarter label: {label!r}"
        raise ValueError(msg)
    month, year = match.group(1), int(match.group(2))
    quarter = {3: 1, 6: 2, 9: 3, 12: 4}[_QTR_MAP[month]]
    return pd.Period(year=year, quarter=quarter, freq="Q-DEC")


def _download_gfs_workbook() -> Path:
    """Download the GFS workbook, caching locally."""
    _CACHE_DIR.mkdir(exist_ok=True)

    # Fetch landing page to find the Excel link
    resp = requests.get(GFS_URL, timeout=30)
    resp.raise_for_status()

    # Find xlsx link in the page
    xlsx_match = re.search(r'href="([^"]+\.xlsx)"', resp.text)
    if xlsx_match is None:
        msg = "Could not find GFS Excel link on landing page"
        raise RuntimeError(msg)

    xlsx_url = xlsx_match.group(1)
    if not xlsx_url.startswith("http"):
        # Relative URL — construct absolute
        from urllib.parse import urljoin  # noqa: PLC0415

        xlsx_url = urljoin(GFS_URL, xlsx_url)

    # Download workbook
    wb_resp = requests.get(xlsx_url, timeout=60)
    wb_resp.raise_for_status()
    _CACHE_FILE.write_bytes(wb_resp.content)
    return _CACHE_FILE


@cache
def _load_gfs_table15() -> dict[str, pd.Series]:
    """Parse Table 15 (Key Measures and Fiscal Aggregates) from GFS workbook.

    Returns dict of {series_name: pd.Series} with quarterly PeriodIndex.
    """
    if not _CACHE_FILE.exists():
        _download_gfs_workbook()

    wb = openpyxl.load_workbook(_CACHE_FILE, read_only=True)
    ws = wb["Table 15"]

    rows = list(ws.iter_rows(values_only=True))

    # Row 6 has quarter labels (1-indexed in Excel, 0-indexed here → index 5)
    quarter_labels = [v for v in rows[5] if v is not None and v != ""]
    periods = [_parse_quarter(q) for q in quarter_labels]
    idx = pd.PeriodIndex(periods, freq="Q-DEC")

    # Parse data rows (rows 10-12 in Excel → indices 9-11)
    series_rows = {
        "gfce": 9,       # General government final consumption expenditure
        "gg_gfcf": 10,   # General government gross fixed capital formation
        "pub_gfcf": 11,  # Total public gross fixed capital formation
    }

    result = {}
    for name, row_idx in series_rows.items():
        values = rows[row_idx][1 : len(periods) + 1]
        s = pd.Series([float(v) if v is not None else np.nan for v in values], index=idx, name=name)
        result[name] = s

    wb.close()
    return result


def get_gov_consumption_gfs_qrtly() -> DataSeries:
    """Get general government final consumption expenditure (quarterly, SA, CVM).

    From GFS Table 15. This is a direct GDP expenditure component.

    Returns:
        DataSeries with quarterly GFCE ($m)

    """
    data = _load_gfs_table15()
    return DataSeries(
        data=data["gfce"],
        source="ABS",
        units="$m",
        description="General government final consumption expenditure (SA, CVM)",
        cat="5519.0",
        table="Table 15",
    )


def get_gov_consumption_gfs_growth_qrtly() -> DataSeries:
    """Get quarterly GFCE growth (log difference).

    Returns:
        DataSeries with GFCE growth (% per quarter)

    """
    gfce = get_gov_consumption_gfs_qrtly()
    log_g = np.log(gfce.data) * 100
    growth = log_g.diff(1)

    return DataSeries(
        data=growth,
        source=gfce.source,
        units="% per quarter",
        description="GFCE growth (quarterly, log difference, from GFS)",
        cat=gfce.cat,
        table=gfce.table,
    )


def get_gov_consumption_spliced_growth_qrtly() -> DataSeries:
    """Get GFCE growth spliced from 5206.0 history + GFS early release.

    Uses 5206.0 (national accounts) for long bridge estimation history, extended
    with GFS Table 15 for quarters not yet in 5206.0. This provides the full
    historical series for OLS estimation plus the early GFS read for the target
    quarter (~2 weeks before GDP publication).

    The two sources produce identical growth rates on revised data, so no
    scaling is needed — GFS quarters beyond 5206.0 are simply appended.

    Returns:
        DataSeries with quarterly GFCE growth (% per quarter)

    """
    from src.data.gov_spending import get_gov_growth_qrtly  # noqa: PLC0415

    na_growth = get_gov_growth_qrtly().data  # 5206.0 — long history
    gfs_growth = get_gov_consumption_gfs_growth_qrtly().data  # GFS — early release

    # Append GFS quarters that extend beyond the national accounts
    na_last = na_growth.dropna().index[-1]
    gfs_extension = gfs_growth[gfs_growth.index > na_last].dropna()

    spliced = pd.concat([na_growth, gfs_extension]) if len(gfs_extension) > 0 else na_growth

    return DataSeries(
        data=spliced,
        source="ABS",
        units="% per quarter",
        description="GFCE growth (5206.0 history + GFS early release)",
        cat="5206.0 + GFS",
        table="spliced",
    )


def get_public_investment_gfs_qrtly() -> DataSeries:
    """Get total public gross fixed capital formation (quarterly, SA, CVM).

    From GFS Table 15. Includes general government + public corporations.

    Returns:
        DataSeries with quarterly public GFCF ($m)

    """
    data = _load_gfs_table15()
    return DataSeries(
        data=data["pub_gfcf"],
        source="ABS",
        units="$m",
        description="Total public gross fixed capital formation (SA, CVM)",
        cat="5519.0",
        table="Table 15",
    )


def get_public_investment_gfs_growth_qrtly() -> DataSeries:
    """Get quarterly public GFCF growth (log difference).

    Returns:
        DataSeries with public GFCF growth (% per quarter)

    """
    gfcf = get_public_investment_gfs_qrtly()
    log_g = np.log(gfcf.data) * 100
    growth = log_g.diff(1)

    return DataSeries(
        data=growth,
        source=gfcf.source,
        units="% per quarter",
        description="Public GFCF growth (quarterly, log difference, from GFS)",
        cat=gfcf.cat,
        table=gfcf.table,
    )
