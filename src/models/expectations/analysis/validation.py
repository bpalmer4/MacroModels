"""Validate expectations model against RBA PIE_RBAQ series."""

from typing import TYPE_CHECKING, Any

import mgplot as mg
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from src.data.rba_loader import get_inflation_expectations

if TYPE_CHECKING:
    from src.models.expectations.model import ExpectationsResults


def get_pie_rbaq() -> pd.Series:
    """Load RBA PIE_RBAQ series for comparison."""
    ds = get_inflation_expectations()
    return ds.data


def compare_to_rba(
    results: "ExpectationsResults",
) -> pd.DataFrame:
    """Compare model output to RBA PIE_RBAQ series.

    Returns:
        DataFrame with model median, RBA series, and difference
    """
    pie_rbaq = get_pie_rbaq()
    model_median = results.expectations_median()

    # Align to common index
    common_idx = model_median.index.intersection(pie_rbaq.index)

    comparison = pd.DataFrame(
        {
            "model_median": model_median.reindex(common_idx),
            "rba_series": pie_rbaq.reindex(common_idx),
        }
    )
    comparison["difference"] = comparison["model_median"] - comparison["rba_series"]
    comparison["abs_diff"] = comparison["difference"].abs()

    return comparison


def validation_statistics(results: "ExpectationsResults") -> dict[str, float]:
    """Compute validation statistics comparing model to RBA series.

    Returns:
        Dict with RMSE, MAE, correlation, and mean bias
    """
    comparison = compare_to_rba(results)

    diff = comparison["difference"].dropna()

    return {
        "rmse": np.sqrt((diff**2).mean()),
        "mae": diff.abs().mean(),
        "correlation": comparison["model_median"].corr(comparison["rba_series"]),
        "mean_bias": diff.mean(),
        "n_obs": len(diff),
    }


def print_validation_summary(results: "ExpectationsResults") -> None:
    """Print validation summary comparing model to RBA series."""
    stats = validation_statistics(results)

    print("\n=== Validation against RBA PIE_RBAQ ===")
    print(f"Observations: {stats['n_obs']}")
    print(f"Correlation:  {stats['correlation']:.3f}")
    print(f"RMSE:         {stats['rmse']:.3f} pp")
    print(f"MAE:          {stats['mae']:.3f} pp")
    print(f"Mean bias:    {stats['mean_bias']:+.3f} pp")

    # Interpret results
    if stats["correlation"] > 0.9:
        print("\nStrong alignment with RBA series.")
    elif stats["correlation"] > 0.7:
        print("\nModerate alignment with RBA series.")
    else:
        print("\nWeak alignment - review model specification.")


def plot_validation(
    results: "ExpectationsResults",
    start: pd.Period | None = None,
    title: str = "Inflation Expectations: Model vs RBA",
    lfooter: str = "Cusbert (2017) signal extraction",
    rfooter: str | None = None,
    **kwargs: Any,
) -> Axes | None:
    """Plot model estimates against RBA PIE_RBAQ series.

    Args:
        results: ExpectationsResults from model estimation
        start: Start period for plot
        title: Chart title
        lfooter: Left footer text
        rfooter: Right footer text
        **kwargs: Additional arguments passed to mg.finalise_plot

    Returns:
        Axes if finalise=False, None otherwise
    """
    from src.models.common.timeseries import plot_posterior_timeseries

    pie_rbaq = get_pie_rbaq()
    data = results.expectations_posterior()

    if rfooter is None:
        stats = validation_statistics(results)
        rfooter = f"Correlation: {stats['correlation']:.2f}, RMSE: {stats['rmse']:.2f}pp"

    # Plot model posterior (don't finalise)
    ax = plot_posterior_timeseries(
        data=data,
        legend_stem="Model",
        color="blue",
        start=start,
        finalise=False,
    )

    if ax is None:
        return None

    # Filter RBA series
    if start is not None:
        pie_rbaq = pie_rbaq[pie_rbaq.index >= start]

    # Plot RBA series
    ax = mg.line_plot(
        pie_rbaq,
        ax=ax,
        color="red",
        width=1.5,
        label="RBA PIE_RBAQ",
        annotate=False,
        zorder=5,
    )

    kwargs.setdefault("axisbelow", True)
    mg.finalise_plot(ax, title=title, lfooter=lfooter, rfooter=rfooter, **kwargs)
    return None
