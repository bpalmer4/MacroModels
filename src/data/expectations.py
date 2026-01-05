"""Inflation expectations survey data from RBA G3 table.

Provides survey-based inflation expectations measures:
- Business expectations (1989+)
- Union expectations (1996-2023)
- Market/professional forecasters (1993+)
"""

import pandas as pd

from src.data.dataseries import DataSeries

# --- Constants ---

G3_URL = "https://www.rba.gov.au/statistics/tables/xls/g03hist.xlsx"

# Series codes in RBA G3 table
G3_SERIES = {
    "business": "GBUSEXP",      # Business expectations (1989+)
    "union_1y": "GUNIEXPY",     # Union expectations 1yr (1996-2023)
    "union_yoy": "GUNIEXPYY",   # Union expectations YoY (1997-2023)
    "market_1y": "GMAREXPY",    # Market/professional 1yr (1993+)
    "market_yoy": "GMAREXPYY",  # Market/professional YoY (1994+)
}


# --- Public API ---


def get_expectations_surveys() -> dict[str, DataSeries]:
    """Get inflation expectations survey measures from RBA G3.

    Returns:
        Dict of DataSeries keyed by series name:
        - business: Business inflation expectations
        - union_1y: Union expectations 1 year ahead
        - union_yoy: Union expectations year-on-year
        - market_1y: Market/professional forecasters 1 year ahead
        - market_yoy: Market/professional forecasters year-on-year

    """
    df = pd.read_excel(G3_URL, sheet_name="Data", skiprows=10, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Convert to quarterly (end of quarter)
    df.index = df.index.to_period("Q")
    df = df.groupby(df.index).last()

    result = {}
    for name, code in G3_SERIES.items():
        if code in df.columns:
            series = df[code].dropna()
            result[name] = DataSeries(
                data=series,
                source="RBA",
                units="%",
                description=f"Inflation Expectations ({name})",
                table="G3",
                series_id=code,
            )

    return result


# --- Testing ---

if __name__ == "__main__":
    print("Loading G3 expectations surveys...")
    surveys = get_expectations_surveys()

    for name, ds in surveys.items():
        series = ds.data
        print(f"\n{name}:")
        print(f"  Range: {series.index[0]} to {series.index[-1]}")
        print(f"  N obs: {len(series)}")
        print(f"  Latest: {series.iloc[-1]:.2f}%")
