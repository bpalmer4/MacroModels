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


def get_unemployment_rate_lagged_qrtly() -> DataSeries:
    """Get lagged unemployment rate (U_{t-1}).

    Returns:
        DataSeries with lagged unemployment rate (%)

    """
    U = get_unemployment_rate_qrtly()
    U_1 = U.data.shift(1)

    return DataSeries(
        data=U_1,
        source=U.source,
        units="%",
        description="Unemployment rate lagged one quarter",
        cat=U.cat,
        table=U.table,
    )


def get_unemployment_speed_limit_qrtly() -> DataSeries:
    """Get unemployment speed limit term for wage equations.

    Speed limit = ΔU_{t-1} / U_t

    This captures the rate of change in unemployment relative to its level,
    used in wage Phillips curves to capture labour market momentum effects.

    Returns:
        DataSeries with speed limit term (ratio)

    """
    U = get_unemployment_rate_qrtly()
    delta_U = U.data.diff(1)
    speed_limit = delta_U.shift(1) / U.data

    return DataSeries(
        data=speed_limit,
        source=U.source,
        units="ratio",
        description="Unemployment speed limit (ΔU_{t-1}/U_t)",
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


def get_employed_qrtly() -> DataSeries:
    """Get total employed from Modellers Database (quarterly).

    Calculated as Labour Force - Unemployed.

    Returns:
        DataSeries with quarterly employment count

    """
    lf = get_labour_force_qrtly()
    unemp = get_unemployed_qrtly()

    employed = lf.data - unemp.data

    return DataSeries(
        data=employed,
        source="ABS",
        units="000",
        description="Total employed (quarterly, from Modellers Database)",
        cat="1364.0.15.003",
        table="1364015003",
    )


def get_employment_growth_qrtly() -> DataSeries:
    """Get quarterly employment growth (log difference).

    Returns:
        DataSeries with employment growth (% per quarter)

    """
    emp = get_employed_qrtly()
    log_emp = np.log(emp.data) * 100
    emp_growth = log_emp.diff(1)

    return DataSeries(
        data=emp_growth,
        source=emp.source,
        units="% per quarter",
        description="Employment growth (quarterly, log difference)",
        cat=emp.cat,
        table=emp.table,
    )


def get_employment_growth_lagged_qrtly() -> DataSeries:
    """Get lagged quarterly employment growth.

    Returns:
        DataSeries with employment growth lagged one quarter (% per quarter)

    """
    emp_growth = get_employment_growth_qrtly()
    emp_growth_1 = emp_growth.data.shift(1)

    return DataSeries(
        data=emp_growth_1,
        source=emp_growth.source,
        units="% per quarter",
        description="Employment growth lagged one quarter",
        cat=emp_growth.cat,
        table=emp_growth.table,
    )


def get_hours_growth_qrtly() -> DataSeries:
    """Get quarterly hours worked growth (log difference).

    Uses hours worked index from Labour Force survey.

    Returns:
        DataSeries with hours growth (% per quarter)

    """
    from src.data.series_specs import HOURS_WORKED_INDEX

    hours_index = load_series(HOURS_WORKED_INDEX)
    log_hours = np.log(hours_index.data) * 100
    hours_growth = log_hours.diff(1)

    return DataSeries(
        data=hours_growth,
        source=hours_index.source,
        units="% per quarter",
        description="Hours worked growth (quarterly, log difference)",
        cat=hours_index.cat,
        table=hours_index.table,
    )
