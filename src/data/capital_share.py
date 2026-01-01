"""Capital share calculation from National Accounts data.

Computes α = GOS / (GOS + COE) from ABS 5206.0 income components.

This gives a time-varying capital share for the Cobb-Douglas production
function, based on observed factor income shares rather than a fixed assumption.
"""

import pandas as pd

from src.data.abs_loader import get_abs_data
from src.data.dataseries import DataSeries
from src.data.series_specs import COMPENSATION_OF_EMPLOYEES, GROSS_OPERATING_SURPLUS


def get_capital_share() -> DataSeries:
    """Compute capital share α = GOS / (GOS + COE).

    Uses Gross Operating Surplus (corporations) and Compensation of Employees
    from ABS National Accounts to derive the time-varying capital share.

    Returns:
        DataSeries with quarterly capital share (0 < α < 1)

    """
    data = get_abs_data({
        "GOS": GROSS_OPERATING_SURPLUS,
        "COE": COMPENSATION_OF_EMPLOYEES,
    })

    gos = data["GOS"].data
    coe = data["COE"].data

    # Align series to common index
    common_idx = gos.index.intersection(coe.index)
    gos = gos.loc[common_idx]
    coe = coe.loc[common_idx]

    # Capital share: GOS / (GOS + COE)
    alpha = gos / (gos + coe)

    return DataSeries(
        data=alpha,
        source="ABS",
        units="ratio",
        description="Capital share α = GOS / (GOS + COE)",
        cat="5206.0",
    )


if __name__ == "__main__":
    print("Testing capital share calculation...\n")

    alpha = get_capital_share()
    print(f"Capital share: {alpha}")
    print(f"Range: {alpha.data.index.min()} to {alpha.data.index.max()}")
    print(f"\nMean: {alpha.data.mean():.4f}")
    print(f"Std:  {alpha.data.std():.4f}")
    print(f"Min:  {alpha.data.min():.4f}")
    print(f"Max:  {alpha.data.max():.4f}")
    print("\nRecent values:")
    print(alpha.data.tail(8).round(4))
