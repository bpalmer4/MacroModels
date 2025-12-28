"""Global Supply Chain Pressure Index data loading.

Provides GSCPI from NY Fed for supply shock analysis.
Source: https://www.newyorkfed.org/research/policy/gscpi

Note: The raw GSCPI has values for all periods. Models typically
only use GSCPI during the COVID period (2020Q1-2023Q2) and set
it to zero otherwise - that masking is a model-level decision.
"""

from pathlib import Path

import pandas as pd
import readabs as ra

from src.data.dataseries import DataSeries


def get_gscpi_monthly() -> DataSeries:
    """Get Global Supply Chain Pressure Index (monthly).

    Returns full GSCPI series - models may mask to COVID period only.

    Returns:
        DataSeries with monthly GSCPI

    """
    gscpi_path = Path(__file__).parent.parent.parent / "input_data" / "gscpi_data.xls"
    gscpi = pd.read_excel(
        gscpi_path,
        sheet_name="GSCPI Monthly Data",
        index_col=0,
        parse_dates=True,
    )["GSCPI"]

    return DataSeries(
        data=gscpi,
        source="NY Fed",
        units="Index (std devs from mean)",
        description="Global Supply Chain Pressure Index (monthly)",
    )


def get_gscpi_qrtly() -> DataSeries:
    """Get Global Supply Chain Pressure Index (quarterly mean).

    Returns full GSCPI series - models may mask to COVID period only.

    Returns:
        DataSeries with quarterly GSCPI

    """
    monthly = get_gscpi_monthly()
    quarterly = ra.monthly_to_qtly(monthly.data, q_ending="DEC", f="mean")
    quarterly.index = pd.PeriodIndex(quarterly.index, freq="Q")

    return DataSeries(
        data=quarterly,
        source="NY Fed",
        units="Index (std devs from mean)",
        description="Global Supply Chain Pressure Index (quarterly mean)",
    )


def get_gscpi_covid_lagged_qrtly() -> DataSeries:
    """Get GSCPI masked to COVID period only, lagged 2 quarters.

    The GSCPI is only used during the COVID period (2020Q1-2023Q2) to capture
    supply chain disruptions. Outside this period, it is set to zero.
    Lagged by 2 quarters as supply chain effects take time to impact inflation.

    Returns:
        DataSeries with COVID-masked, lagged GSCPI

    """
    gscpi = get_gscpi_qrtly().data

    # Mask to COVID period only (zero otherwise)
    covid_period = pd.period_range("2020Q1", "2023Q2", freq="Q")
    gscpi_masked = gscpi.where(gscpi.index.isin(covid_period), other=0.0)

    # Reindex to full range (1960Q1 to current quarter) and fill missing with 0
    current_q = pd.Period.now("Q")
    full_range = pd.period_range("1960Q1", current_q, freq="Q")
    gscpi_full = gscpi_masked.reindex(full_range, fill_value=0.0)

    # Lag by 2 quarters (supply chain effects take time)
    gscpi_lagged = gscpi_full.shift(2).fillna(0.0)

    return DataSeries(
        data=gscpi_lagged,
        source="NY Fed",
        units="Index (std devs from mean)",
        description="GSCPI (COVID period only, lagged 2 quarters)",
    )
