"""Oil price data loading.

Fetches crude oil prices from World Bank Pink Sheet (USD) and converts
to AUD using RBA exchange rates.

World Bank Commodity Price Data ("Pink Sheet")
----------------------------------------------
The World Bank publishes monthly commodity price data including crude oil
(average of Brent, Dubai, and WTI). Data is in nominal USD terms.

For Australian import price modeling, we convert to AUD since import prices
are paid in foreign currency. A weaker AUD amplifies the AUD cost of oil
even if USD oil prices are stable.

Data sources:
- World Bank: https://www.worldbank.org/en/research/commodity-markets
- RBA: Exchange rates (F11 table)

Note: Uses same parsing logic as ~/ABS/notebooks/World Bank Commodity Prices.ipynb
"""

from functools import cache

import numpy as np
import pandas as pd

from src.data.dataseries import DataSeries


# World Bank Pink Sheet URL (monthly data)
# WARNING: This URL may change as new data is added
PINK_SHEET_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "18675f1d1639c7a34d463f59263ba0a2-0050012025/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)


@cache
def _get_commodity_data() -> pd.DataFrame:
    """Download and process World Bank commodity price data.

    Returns:
        DataFrame with all commodity prices, PeriodIndex

    """
    commodities: pd.DataFrame = pd.read_excel(
        PINK_SHEET_URL,
        sheet_name="Monthly Prices",
        header=None,
        na_values=["N/A", "missing", "-", "…", "..."],
        index_col=0,
    )
    labels = commodities.iloc[4]
    commodities.columns = labels
    commodities = commodities.iloc[7:].dropna(axis=0, how="all")
    commodities.index = pd.PeriodIndex(
        commodities.index.str.replace("M", "-"), freq="M"
    )
    return commodities


@cache
def _get_aud_usd_rate() -> pd.Series:
    """Get AUD/USD exchange rate from RBA.

    Returns:
        Series with AUD/USD rate, PeriodIndex

    """
    hist_url = "https://www.rba.gov.au/statistics/tables/xls-hist/f11hist-1969-2009.xls"
    now_url = "https://www.rba.gov.au/statistics/tables/xls-hist/f11hist.xls"

    container = []
    for url in [hist_url, now_url]:
        table = pd.read_excel(url, sheet_name="Data", header=10, index_col=0)
        series: pd.Series = table["FXRUSD"].dropna()
        series.index = pd.PeriodIndex(series.index, freq="M")
        container.append(series)

    return pd.concat(container)


def get_oil_price_usd_monthly() -> DataSeries:
    """Get crude oil price in USD from World Bank Pink Sheet.

    Returns average of Brent, Dubai, and WTI (the World Bank's crude oil index).

    Returns:
        DataSeries with monthly oil price (USD per barrel)

    """
    commodities = _get_commodity_data()
    oil = commodities["Crude oil, average"].dropna().astype(float)

    return DataSeries(
        data=oil,
        source="World Bank",
        units="USD/barrel",
        description="Crude Oil Price (Brent/Dubai/WTI average)",
        metadata={"url": PINK_SHEET_URL},
    )


def get_usd_aud_monthly() -> DataSeries:
    """Get USD/AUD exchange rate (monthly).

    Note: RBA publishes AUD/USD (how many USD per AUD).
    We invert to get USD/AUD (how many AUD per USD) for oil conversion.

    Returns:
        DataSeries with monthly USD/AUD rate

    """
    aud_usd = _get_aud_usd_rate()
    usd_aud = 1 / aud_usd

    return DataSeries(
        data=usd_aud,
        source="RBA",
        units="AUD per USD",
        description="USD/AUD Exchange Rate (inverted from AUD/USD)",
        table="F11",
    )


def get_oil_price_aud_monthly() -> DataSeries:
    """Get crude oil price in AUD (monthly).

    Converts USD oil price to AUD using RBA exchange rate.
    Higher AUD oil price can result from either:
    - Higher USD oil price, or
    - Weaker AUD (higher USD/AUD rate)

    Returns:
        DataSeries with monthly oil price (AUD per barrel)

    """
    oil_usd = get_oil_price_usd_monthly()
    usd_aud = get_usd_aud_monthly()

    # Align indices
    common_idx = oil_usd.data.index.intersection(usd_aud.data.index)
    oil_aligned = oil_usd.data.reindex(common_idx)
    rate_aligned = usd_aud.data.reindex(common_idx)

    # Convert: USD/barrel × AUD/USD = AUD/barrel
    oil_aud = oil_aligned * rate_aligned

    return DataSeries(
        data=oil_aud,
        source="World Bank / RBA",
        units="AUD/barrel",
        description="Crude Oil Price in AUD",
    )


def get_oil_price_aud_qrtly() -> DataSeries:
    """Get crude oil price in AUD (quarterly average).

    Returns:
        DataSeries with quarterly oil price (AUD per barrel)

    """
    monthly = get_oil_price_aud_monthly()
    quarterly = monthly.data.groupby(monthly.data.index.asfreq("Q")).mean()

    return DataSeries(
        data=quarterly,
        source="World Bank / RBA",
        units="AUD/barrel",
        description="Crude Oil Price in AUD (quarterly average)",
    )


def get_oil_change_annual() -> DataSeries:
    """Get annual percentage change in AUD oil price (4-quarter log difference).

    Returns:
        DataSeries with annual oil price change (%)

    """
    oil_q = get_oil_price_aud_qrtly()
    log_oil = np.log(oil_q.data) * 100
    delta_4_oil = log_oil.diff(periods=4)

    return DataSeries(
        data=delta_4_oil,
        source="World Bank / RBA",
        units="% per year",
        description="Oil price change in AUD (annual, 4-quarter log diff)",
    )
