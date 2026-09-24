"""Load inflation expectations from signal extraction model output.

Provides the expectations model's unanchored median, quarterly.
"""

import pandas as pd

from src.data.dataseries import DataSeries
from src.paths import OUTPUT

# --- Output Location ---

OUTPUT_DIR = OUTPUT / "expectations"


# --- Public API ---


def get_model_expectations_unanchored() -> DataSeries:
    """Load Unanchored inflation expectations from signal extraction model.

    Returns the median of the unanchored estimate (no 2.5% target prior).

    Returns:
        DataSeries with quarterly expectations median (%)

    Raises:
        FileNotFoundError: If model output not found (run expectations model first)

    """
    median = _load_model_output("unanchored")["median"]
    return DataSeries(
        data=median,
        source="Model",
        units="%",
        description="Inflation Expectations (Unanchored)",
        table="expectations_unanchored_hdi",
        series_id="median",
    )


# --- Internal ---


def _load_model_output(model_type: str) -> pd.DataFrame:
    """Load quarterly parquet file for a specific model type.

    Reads the quarterly file every run writes, falling back to the main HDI
    file only for output saved before runs wrote one.
    """
    quarterly_path = OUTPUT_DIR / f"expectations_{model_type}_hdi_quarterly.parquet"
    main_path = OUTPUT_DIR / f"expectations_{model_type}_hdi.parquet"
    path = quarterly_path if quarterly_path.exists() else main_path

    if not path.exists():
        raise FileNotFoundError(
            f"Expectations model output not found at {main_path}. "
            f"Run the expectations model first: "
            f"uv run python -m src.models.expectations.stage1 --model {model_type}"
        )
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.PeriodIndex):
        df.index = pd.PeriodIndex(df.index, freq="Q")
    return df
