"""Average Weekly Earnings data loading.

Loads AWE from ABS 6302.0 (Average Weekly Earnings, Australia).
Note: ABS 6302 is published semi-annually (May and November quarters).
"""

import numpy as np
import pandas as pd

from src.data.abs_loader import load_series
from src.data.dataseries import DataSeries
from src.data.series_specs import AWE_FULL_TIME_ADULTS


def get_awe_index() -> DataSeries:
    """Get Average Weekly Earnings index (full-time adult ordinary time).

    Returns:
        DataSeries with AWE levels ($)

    """
    return load_series(AWE_FULL_TIME_ADULTS)


def get_awe_growth_qrtly() -> DataSeries:
    """Get quarterly AWE growth (interpolated from semi-annual).

    ABS 6302 is published semi-annually. This function:
    1. Loads the semi-annual AWE data
    2. Interpolates to quarterly frequency
    3. Computes log growth rate

    Returns:
        DataSeries with AWE growth (% per quarter)

    """
    awe = load_series(AWE_FULL_TIME_ADULTS).data

    # Interpolate to quarterly (ABS 6302 is semi-annual: May & November)
    # Convert to timestamp, resample to quarterly, interpolate
    awe_ts = awe.to_timestamp(how="end")
    awe_quarterly = awe_ts.resample("QE").interpolate(method="linear")
    awe_quarterly = awe_quarterly.to_period("Q")

    # Compute log growth
    log_awe = np.log(awe_quarterly)
    delta_awe = log_awe.diff(1) * 100

    return DataSeries(
        data=delta_awe,
        source="ABS",
        units="% per quarter",
        description="Average Weekly Earnings growth (quarterly, interpolated from semi-annual)",
        cat="6302.0",
    )


def get_awe_growth_annual() -> DataSeries:
    """Get annual AWE growth (year-on-year).

    Returns:
        DataSeries with AWE growth (% per year)

    """
    awe = load_series(AWE_FULL_TIME_ADULTS).data

    # Interpolate to quarterly first
    awe_ts = awe.to_timestamp(how="end")
    awe_quarterly = awe_ts.resample("QE").interpolate(method="linear")
    awe_quarterly = awe_quarterly.to_period("Q")

    # Compute year-on-year log growth
    log_awe = np.log(awe_quarterly)
    delta4_awe = log_awe.diff(4) * 100

    return DataSeries(
        data=delta4_awe,
        source="ABS",
        units="% per year",
        description="Average Weekly Earnings growth (annual, year-on-year)",
        cat="6302.0",
    )


if __name__ == "__main__":
    print("Testing AWE data loading...\n")

    print("AWE Index:")
    awe = get_awe_index()
    print(f"  {awe}")
    print(f"  Range: {awe.data.index.min()} to {awe.data.index.max()}")
    print(awe.data.tail())

    print("\nAWE Growth (quarterly):")
    awe_q = get_awe_growth_qrtly()
    print(f"  {awe_q}")
    print(awe_q.data.tail())

    print("\nAWE Growth (annual):")
    awe_a = get_awe_growth_annual()
    print(f"  {awe_a}")
    print(awe_a.data.tail())
