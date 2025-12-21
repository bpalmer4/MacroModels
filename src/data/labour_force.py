"""Labour force data loading.

Provides unemployment rate, labour force, and hours worked from ABS data.
Uses Modellers Database (1364.0.15.003) for total population coverage
including defence/non-civilian employment.
"""

import numpy as np
import readabs as ra

from src.data.abs_loader import get_abs_data, load_series
from src.data.dataseries import DataSeries
from src.data.series_specs import (
    HOURS_WORKED,
    LABOUR_FORCE_TOTAL,
    PARTICIPATION_RATE,
    UNEMPLOYED_TOTAL,
)


def get_labour_force_qrtly() -> DataSeries:
    """Get total labour force from Modellers Database (quarterly).

    Returns:
        DataSeries with quarterly labour force

    """
    return load_series(LABOUR_FORCE_TOTAL)


def get_unemployed_qrtly() -> DataSeries:
    """Get total unemployed from Modellers Database (quarterly).

    Returns:
        DataSeries with quarterly unemployed count

    """
    return load_series(UNEMPLOYED_TOTAL)


def get_unemployment_rate_qrtly() -> DataSeries:
    """Calculate unemployment rate from labour force and unemployed (quarterly).

    Unemployment rate = (unemployed / labour_force) * 100

    Uses Modellers Database (1364.0.15.003) which provides quarterly data
    covering total population including defence/non-civilian employment.

    Returns:
        DataSeries with unemployment rate (%)

    """
    data = get_abs_data({
        "LF": LABOUR_FORCE_TOTAL,
        "Unemp": UNEMPLOYED_TOTAL,
    })

    lf = data["LF"].data
    unemp = data["Unemp"].data
    U = (unemp / lf) * 100

    return DataSeries(
        data=U,
        source="ABS",
        units="%",
        description="Unemployment rate (quarterly, from Modellers Database)",
        cat="1364.0.15.003",
        table="1364015003",
    )


def get_unemployment_change_qrtly() -> DataSeries:
    """Get quarterly change in unemployment rate.

    Returns:
        DataSeries with ΔU (percentage points)

    """
    U = get_unemployment_rate_qrtly()
    delta_U = U.data.diff(1)

    return DataSeries(
        data=delta_U,
        source=U.source,
        units="pp",
        description="Change in unemployment rate (quarterly)",
        cat=U.cat,
        table=U.table,
    )


def get_hours_worked_monthly() -> DataSeries:
    """Get hours worked from ABS Labour Force survey (monthly).

    Returns:
        DataSeries with monthly hours worked

    """
    return load_series(HOURS_WORKED)


def get_hours_worked_qrtly() -> DataSeries:
    """Get hours worked from ABS Labour Force survey (quarterly sum).

    Converts monthly hours to quarterly by summing 3 months.

    Returns:
        DataSeries with quarterly hours worked

    """
    hours = load_series(HOURS_WORKED)
    hours_q = ra.monthly_to_qtly(hours.data, q_ending="DEC", f="sum")

    return DataSeries(
        data=hours_q,
        source=hours.source,
        units=hours.units,
        description=f"{hours.description} (quarterly sum)",
        series_id=hours.series_id,
        table=hours.table,
        cat=hours.cat,
        stype=hours.stype,
    )


def get_labour_force_growth_qrtly() -> DataSeries:
    """Get quarterly labour force growth (log difference).

    Returns:
        DataSeries with labour force growth (% per quarter)

    """
    lf = get_labour_force_qrtly()
    log_lf = np.log(lf.data) * 100
    lf_growth = log_lf.diff(1)

    return DataSeries(
        data=lf_growth,
        source=lf.source,
        units="% per quarter",
        description="Labour force growth (quarterly, log difference)",
        cat=lf.cat,
        table=lf.table,
    )


def get_participation_rate_monthly() -> DataSeries:
    """Get participation rate from ABS Labour Force survey (monthly).

    Participation rate = Labour Force / Civilian Population 15+ (%)

    Returns:
        DataSeries with monthly participation rate (%)

    """
    return load_series(PARTICIPATION_RATE)


def get_participation_rate_qrtly() -> DataSeries:
    """Get participation rate (quarterly average).

    Returns:
        DataSeries with quarterly participation rate (%)

    """
    monthly = get_participation_rate_monthly()
    quarterly = ra.monthly_to_qtly(monthly.data, q_ending="DEC", f="mean")

    return DataSeries(
        data=quarterly,
        source=monthly.source,
        units="%",
        description="Participation rate (quarterly average)",
        cat=monthly.cat,
        table=monthly.table,
    )


def get_participation_rate_change_qrtly() -> DataSeries:
    """Get quarterly change in participation rate.

    Returns:
        DataSeries with Δpr (percentage points)

    """
    pr = get_participation_rate_qrtly()
    delta_pr = pr.data.diff(1)

    return DataSeries(
        data=delta_pr,
        source=pr.source,
        units="pp",
        description="Change in participation rate (quarterly)",
        cat=pr.cat,
        table=pr.table,
    )
