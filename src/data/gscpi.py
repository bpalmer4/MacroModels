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
    gscpi_path = Path(__file__).parent.parent.parent / "data" / "gscpi_data.xls"
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
