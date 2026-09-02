"""Survey data loading.

Provides monthly business and consumer survey indicators from RBA Statistical
Table H3 (Monthly Activity Indicators). These are soft data that lead hard
activity indicators at turning points.
"""

from io import BytesIO

import pandas as pd
import readabs as ra
import requests

from src.data.dataseries import DataSeries

H3_URL = "https://www.rba.gov.au/statistics/tables/xls/h03hist.xlsx"


def _load_h3_series(col: str) -> pd.Series:
    """Load a series from RBA H3 table.

    Args:
        col: Column name (series ID, e.g. "GICNBC")

    Returns:
        Series with monthly PeriodIndex

    """
    # Fetch via requests (certifi trust store) rather than letting pandas use
    # urllib: the macOS system CA bundle lacks the RBA's current root.
    response = requests.get(H3_URL, timeout=60)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), sheet_name="Data", skiprows=10, index_col=0)
    df.index = pd.to_datetime(df.index)
    s = df[col].dropna()
    dates = s.index
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError(f"H3 sheet index did not parse as dates: {type(dates).__name__}")
    s.index = dates.to_period("M")
    return s


def get_nab_business_conditions_monthly() -> DataSeries:
    """Get NAB business conditions index (monthly, SA).

    Deviation from long-run average (percentage points).
    Published monthly by NAB, republished in RBA Table H3.

    Returns:
        DataSeries with monthly NAB business conditions

    """
    s = _load_h3_series("GICNBC")
    return DataSeries(
        data=s,
        source="NAB via RBA",
        units="pp from average",
        description="NAB business conditions index (SA)",
        table="H3",
        series_id="GICNBC",
    )


def get_nab_business_conditions_qrtly() -> DataSeries:
    """Get NAB business conditions (quarterly mean).

    Returns:
        DataSeries with quarterly NAB business conditions

    """
    monthly = get_nab_business_conditions_monthly()
    quarterly = ra.monthly_to_qtly(monthly.data, q_ending="DEC", f="mean")
    return DataSeries(
        data=quarterly,
        source=monthly.source,
        units=monthly.units,
        description=f"{monthly.description} (quarterly mean)",
        table=monthly.table,
        series_id=monthly.series_id,
    )


def get_consumer_sentiment_monthly() -> DataSeries:
    """Get Westpac-MI consumer sentiment index (monthly, SA).

    Published monthly by Westpac/Melbourne Institute, republished in RBA Table H3.

    Returns:
        DataSeries with monthly consumer sentiment index

    """
    s = _load_h3_series("GICWMICS")
    return DataSeries(
        data=s,
        source="Westpac-MI via RBA",
        units="Index",
        description="Westpac-MI consumer sentiment index (SA)",
        table="H3",
        series_id="GICWMICS",
    )


def get_consumer_sentiment_qrtly() -> DataSeries:
    """Get consumer sentiment (quarterly mean).

    Returns:
        DataSeries with quarterly consumer sentiment index

    """
    monthly = get_consumer_sentiment_monthly()
    quarterly = ra.monthly_to_qtly(monthly.data, q_ending="DEC", f="mean")
    return DataSeries(
        data=quarterly,
        source=monthly.source,
        units=monthly.units,
        description=f"{monthly.description} (quarterly mean)",
        table=monthly.table,
        series_id=monthly.series_id,
    )
