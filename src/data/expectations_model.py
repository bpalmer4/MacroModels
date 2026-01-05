"""Load inflation expectations from signal extraction model output.

Provides model-derived expectations estimates for use in NAIRU Phillips curves.
"""

from pathlib import Path

import pandas as pd

from src.data.dataseries import DataSeries

# --- Output Location ---

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "expectations"


# --- Public API ---


def get_model_expectations() -> DataSeries:
    """Load inflation expectations from signal extraction model.

    Loads the median estimate from the model output. For full posterior
    uncertainty, use get_model_expectations_hdi() instead.

    Returns:
        DataSeries with quarterly expectations median (%)

    Raises:
        FileNotFoundError: If model output not found (run expectations model first)

    """
    hdi = _load_hdi()
    return DataSeries(
        data=hdi["median"],
        source="Model",
        units="%",
        description="Inflation Expectations (signal extraction model)",
        table="expectations_hdi",
        series_id="median",
    )


def get_model_expectations_hdi() -> pd.DataFrame:
    """Load full HDI bounds from signal extraction model.

    Returns:
        DataFrame with columns: lower, median, upper
        Index is PeriodIndex (quarterly)

    Raises:
        FileNotFoundError: If model output not found (run expectations model first)

    """
    return _load_hdi()


# --- Internal ---


def _load_hdi() -> pd.DataFrame:
    """Load HDI parquet file."""
    path = OUTPUT_DIR / "expectations_hdi.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Expectations model output not found at {path}. "
            "Run the expectations model first: "
            "uv run python -m src.models.expectations.model"
        )
    hdi = pd.read_parquet(path)
    # Ensure index is PeriodIndex
    if not isinstance(hdi.index, pd.PeriodIndex):
        hdi.index = pd.PeriodIndex(hdi.index, freq="Q")
    return hdi


# --- Testing ---

if __name__ == "__main__":
    print("Loading model expectations...")

    try:
        exp = get_model_expectations()
        print(f"\nExpectations: {exp.data.index[0]} to {exp.data.index[-1]}")
        print(f"Latest: {exp.data.iloc[-1]:.2f}%")
        print(f"\nRecent values:")
        print(exp.data.tail(8))

        hdi = get_model_expectations_hdi()
        print(f"\nHDI bounds (last 4 quarters):")
        print(hdi.tail(4))
    except FileNotFoundError as e:
        print(f"Error: {e}")
