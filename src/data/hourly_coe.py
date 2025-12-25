"""Hourly Compensation of Employees data loading.

Calculates hourly compensation from National Accounts components:
Hourly COE = Compensation of Employees / Hours Worked Index

This provides a measure of labour cost per hour, distinct from:
- ULC (cost per unit of output) which includes productivity
- AWE (survey earnings) which has compositional effects
"""

import numpy as np

from src.data.abs_loader import get_abs_data
from src.data.dataseries import DataSeries
from src.data.series_specs import COMPENSATION_OF_EMPLOYEES, HOURS_WORKED_INDEX


def get_hourly_coe() -> DataSeries:
    """Get hourly compensation of employees (level).

    Returns:
        DataSeries with hourly COE index

    """
    data = get_abs_data({
        "CoE": COMPENSATION_OF_EMPLOYEES,
        "Hours": HOURS_WORKED_INDEX,
    })

    coe = data["CoE"].data
    hours = data["Hours"].data

    # Align series
    common_idx = coe.index.intersection(hours.index)
    coe = coe.loc[common_idx]
    hours = hours.loc[common_idx]

    hourly_coe = coe / hours

    return DataSeries(
        data=hourly_coe,
        source="ABS",
        units="Index",
        description="Hourly Compensation of Employees (COE / Hours Worked)",
        cat="5206.0",
    )


def get_hourly_coe_growth_qrtly() -> DataSeries:
    """Get quarterly hourly COE growth.

    Returns:
        DataSeries with hourly COE growth (% per quarter)

    """
    hourly_coe = get_hourly_coe().data

    log_hourly_coe = np.log(hourly_coe)
    delta_hourly_coe = log_hourly_coe.diff(1) * 100

    return DataSeries(
        data=delta_hourly_coe,
        source="ABS",
        units="% per quarter",
        description="Hourly Compensation of Employees growth (quarterly, log difference)",
        cat="5206.0",
    )


def get_hourly_coe_growth_annual() -> DataSeries:
    """Get annual hourly COE growth (year-on-year).

    Returns:
        DataSeries with hourly COE growth (% per year)

    """
    hourly_coe = get_hourly_coe().data

    log_hourly_coe = np.log(hourly_coe)
    delta4_hourly_coe = log_hourly_coe.diff(4) * 100

    return DataSeries(
        data=delta4_hourly_coe,
        source="ABS",
        units="% per year",
        description="Hourly Compensation of Employees growth (annual, year-on-year)",
        cat="5206.0",
    )


def get_hourly_coe_growth_lagged_qrtly() -> DataSeries:
    """Get lagged quarterly hourly COE growth.

    Returns:
        DataSeries with hourly COE growth lagged one quarter (% per quarter)

    """
    hcoe_growth = get_hourly_coe_growth_qrtly()
    hcoe_growth_1 = hcoe_growth.data.shift(1)

    return DataSeries(
        data=hcoe_growth_1,
        source=hcoe_growth.source,
        units="% per quarter",
        description="Hourly COE growth lagged one quarter",
        cat=hcoe_growth.cat,
    )


if __name__ == "__main__":
    print("Testing Hourly COE data loading...\n")

    print("Hourly COE (level):")
    hcoe = get_hourly_coe()
    print(f"  {hcoe}")
    print(f"  Range: {hcoe.data.index.min()} to {hcoe.data.index.max()}")
    print(hcoe.data.tail())

    print("\nHourly COE Growth (quarterly):")
    hcoe_q = get_hourly_coe_growth_qrtly()
    print(f"  {hcoe_q}")
    print(hcoe_q.data.tail())

    print("\nHourly COE Growth (annual):")
    hcoe_a = get_hourly_coe_growth_annual()
    print(f"  {hcoe_a}")
    print(hcoe_a.data.tail())
