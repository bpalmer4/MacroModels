"""Load inflation expectations from signal extraction model output.

Provides model-derived expectations estimates for use in NAIRU Phillips curves.

The series splices two models:
- Long Run (10yr bond) through 1991Q2: Smooth decline through disinflation
- Target Anchored from 1991Q3 onwards: Survey-based with target anchor

This gives a more plausible gradual decline in the 1980s (Long Run) while
using the richer survey information once available (Target Anchored).
"""

from pathlib import Path

import pandas as pd

from src.data.dataseries import DataSeries

# --- Output Location ---

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "expectations"

# --- Splice Configuration ---

# Splice point: use Long Run before this, Target Anchored from this point onwards
# 1992Q1 chosen for smallest jump (0.54pp): Long Run 4.22% (1991Q4) → Target 3.68%
SPLICE_POINT = pd.Period("1992Q1", freq="Q")


# --- Public API ---


def get_model_expectations() -> DataSeries:
    """Load inflation expectations from signal extraction model.

    Returns the full Target Anchored series (survey-based with target anchor).

    For full posterior uncertainty, use get_model_expectations_hdi() instead.

    Returns:
        DataSeries with quarterly expectations median (%)

    Raises:
        FileNotFoundError: If model output not found (run expectations model first)

    """
    median = _load_model_output("target")["median"]
    return DataSeries(
        data=median,
        source="Model",
        units="%",
        description="Inflation Expectations (Target Anchored)",
        table="expectations_target_hdi",
        series_id="median",
    )


def get_model_expectations_hdi() -> pd.DataFrame:
    """Load full HDI bounds from Target Anchored model only.

    Note: HDI bounds are not spliced (statistically meaningless to concatenate
    uncertainty intervals from different models). Returns Target Anchored HDI
    for the full sample.

    Returns:
        DataFrame with columns: lower, median, upper
        Index is PeriodIndex (quarterly)

    Raises:
        FileNotFoundError: If model output not found (run expectations model first)

    """
    return _load_model_output("target")


# --- Internal ---


def _load_model_output(model_type: str) -> pd.DataFrame:
    """Load parquet file for a specific model type."""
    path = OUTPUT_DIR / f"expectations_{model_type}_hdi.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Expectations model output not found at {path}. "
            f"Run the expectations model first: "
            f"uv run python -m src.models.expectations.stage1 --model {model_type}"
        )
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.PeriodIndex):
        df.index = pd.PeriodIndex(df.index, freq="Q")
    return df


def _load_spliced_median() -> pd.Series:
    """Load and splice Long Run + Target Anchored median series.

    Uses Long Run (10yr bond) through 1991Q2 for smooth disinflation decline,
    then Target Anchored from 1991Q3 onwards for survey-based estimates.
    """
    long_run = _load_model_output("market")["median"]
    target = _load_model_output("target")["median"]

    # Splice: Long Run before SPLICE_POINT, Target from SPLICE_POINT onwards
    early = long_run[long_run.index < SPLICE_POINT]
    late = target[target.index >= SPLICE_POINT]

    return pd.concat([early, late]).sort_index()


# --- Testing ---

if __name__ == "__main__":
    print("Loading spliced model expectations...")
    print(f"Splice point: {SPLICE_POINT} (Long Run before, Target Anchored from)")

    try:
        exp = get_model_expectations()
        print(f"\nExpectations: {exp.data.index[0]} to {exp.data.index[-1]}")
        print(f"Latest: {exp.data.iloc[-1]:.2f}%")

        # Show splice region
        splice_region = exp.data.loc["1991Q1":"1991Q4"]
        print(f"\nSplice region (1991):")
        print(splice_region.to_string())

        print(f"\nRecent values:")
        print(exp.data.tail(8))
    except FileNotFoundError as e:
        print(f"Error: {e}")
