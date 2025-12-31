"""Shared utilities for DSGE model estimation.

This module provides common functionality used across different DSGE models:
- Regime definitions for Australian macroeconomic history
- Generic MLE estimation wrapper
- Result printing utilities
- Common data loading patterns
"""

from dataclasses import dataclass, fields
from typing import Any, Callable, TypeVar
import numpy as np
import pandas as pd
from scipy import optimize


# =============================================================================
# Regime Definitions
# =============================================================================

# Standard Australian macroeconomic regimes
REGIMES = [
    ("Pre-GFC", "1984Q1", "2008Q3"),
    ("GFC-COVID", "2008Q4", "2020Q4"),
    ("Post-COVID", "2021Q1", None),
]


def get_regime_dates(
    regime_name: str,
    start_override: str | None = None,
) -> tuple[str, str | None]:
    """Get start/end dates for a regime.

    Args:
        regime_name: Name of regime ("Pre-GFC", "GFC-COVID", "Post-COVID")
        start_override: Optional earlier start to use (takes max with regime start)

    Returns:
        (start, end) tuple of period strings
    """
    for name, reg_start, reg_end in REGIMES:
        if name == regime_name:
            if start_override:
                actual_start = max(start_override, reg_start)
            else:
                actual_start = reg_start
            return actual_start, reg_end
    raise ValueError(f"Unknown regime: {regime_name}")


# =============================================================================
# Generic MLE Estimation
# =============================================================================

@dataclass
class EstimationResult:
    """Generic estimation result container."""
    params: Any  # Parameter dataclass
    log_likelihood: float
    convergence: dict
    n_obs: int


ParamClass = TypeVar("ParamClass")


def estimate_mle(
    neg_log_likelihood_fn: Callable[[np.ndarray], float],
    param_class: type[ParamClass],
    initial_params: ParamClass | None,
    estimate_params: list[str],
    param_bounds: dict[str, tuple[float, float]],
    fixed_params: dict[str, float] | None = None,
    n_obs: int = 0,
    maxiter: int = 500,
) -> EstimationResult:
    """Generic MLE estimation wrapper.

    Args:
        neg_log_likelihood_fn: Function taking parameter array, returns -LL
        param_class: Dataclass type for parameters
        initial_params: Initial parameter values (or None for defaults)
        estimate_params: List of parameter names to estimate
        param_bounds: Dict of parameter name -> (lower, upper) bounds
        fixed_params: Dict of parameter name -> fixed value (not estimated)
        n_obs: Number of observations (for result)
        maxiter: Maximum optimization iterations

    Returns:
        EstimationResult with estimated parameters
    """
    if initial_params is None:
        initial_params = param_class()

    # Build parameter dict from initial values
    param_dict = initial_params.to_dict()

    # Apply any fixed parameters
    if fixed_params:
        for name, value in fixed_params.items():
            param_dict[name] = value

    # Initial values and bounds for estimated parameters
    x0 = np.array([param_dict[name] for name in estimate_params])
    bounds = [param_bounds[name] for name in estimate_params]

    def neg_ll(x: np.ndarray) -> float:
        for i, name in enumerate(estimate_params):
            param_dict[name] = x[i]
        return neg_log_likelihood_fn(param_dict)

    result = optimize.minimize(
        neg_ll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )

    # Extract final parameters
    for i, name in enumerate(estimate_params):
        param_dict[name] = result.x[i]
    final_params = param_class(**param_dict)

    return EstimationResult(
        params=final_params,
        log_likelihood=-result.fun,
        convergence={"success": result.success, "nit": result.nit, "nfev": result.nfev},
        n_obs=n_obs,
    )


# =============================================================================
# Regime Estimation Loop
# =============================================================================

def estimate_by_regime(
    load_data_fn: Callable[..., tuple],
    estimate_fn: Callable[..., EstimationResult],
    start: str = "1984Q1",
    min_obs: int = 10,
    verbose: bool = True,
    **load_kwargs,
) -> dict[str, EstimationResult | float]:
    """Generic regime-switching estimation loop.

    Args:
        load_data_fn: Function(start, end, **kwargs) -> (data_tuple)
            Must return tuple where last element is dates (PeriodIndex)
        estimate_fn: Function(data_tuple, **kwargs) -> EstimationResult
        start: Earliest start date
        min_obs: Minimum observations required per regime
        verbose: Print progress
        **load_kwargs: Additional kwargs passed to load_data_fn

    Returns:
        Dict with regime names as keys, EstimationResult as values,
        plus "total_ll" key with summed log-likelihood
    """
    results = {}
    total_ll = 0.0

    for name, reg_start, reg_end in REGIMES:
        actual_start = max(start, reg_start)

        # Load data for this regime
        data = load_data_fn(start=actual_start, end=reg_end, **load_kwargs)
        dates = data[-1]  # Assume last element is dates
        n_obs = len(dates)

        if n_obs < min_obs:
            if verbose:
                print(f"  {name}: insufficient data ({n_obs} obs), skipping")
            continue

        if verbose:
            print(f"  Estimating {name} ({dates[0]} to {dates[-1]}, n={n_obs})...")

        # Estimate
        result = estimate_fn(*data[:-1], dates=dates)  # Pass all but dates, then dates
        results[name] = result
        total_ll += result.log_likelihood

        if verbose:
            print(f"    LL: {result.log_likelihood:.2f}")

    results["total_ll"] = total_ll
    return results


# =============================================================================
# Result Printing
# =============================================================================

def print_regime_results(
    results: dict,
    model_name: str,
    model_desc: str,
    params_to_show: list[str],
    col_width: int = 12,
) -> None:
    """Print estimation results across regimes.

    Args:
        results: Dict from estimate_by_regime
        model_name: Short model name for header
        model_desc: Description line for header
        params_to_show: List of parameter names to display
        col_width: Column width for values
    """
    line_width = 70

    print("=" * line_width)
    print(f"{model_name} ESTIMATION RESULTS")
    print(f"({model_desc})")
    print("=" * line_width)

    # Header row
    print(f"\n{'Param':<14} ", end="")
    for name in results:
        if name != "total_ll":
            print(f"{name:>{col_width}} ", end="")
    print()
    print("-" * line_width)

    # Parameter rows
    for param in params_to_show:
        print(f"{param:<14} ", end="")
        for name, result in results.items():
            if name != "total_ll" and hasattr(result, "params"):
                val = getattr(result.params, param)
                print(f"{val:>{col_width}.4f} ", end="")
        print()

    print("=" * line_width)
    print(f"Total log-likelihood: {results.get('total_ll', 0):.2f}")


def check_bounds(
    result: EstimationResult,
    param_bounds: dict[str, tuple[float, float]],
    tolerance: float = 0.01,
) -> list[str]:
    """Check which parameters are at or near their bounds.

    Args:
        result: Estimation result
        param_bounds: Dict of parameter bounds
        tolerance: Fraction of range to consider "at bound"

    Returns:
        List of parameter names at bounds
    """
    at_bounds = []
    params = result.params

    for name, (lower, upper) in param_bounds.items():
        if hasattr(params, name):
            val = getattr(params, name)
            range_size = upper - lower
            if val <= lower + tolerance * range_size:
                at_bounds.append(f"{name} (at lower)")
            elif val >= upper - tolerance * range_size:
                at_bounds.append(f"{name} (at upper)")

    return at_bounds


# =============================================================================
# Data Loading Utilities
# =============================================================================

def ensure_period_index(series: pd.Series, freq: str = "Q") -> pd.Series:
    """Ensure series has PeriodIndex.

    Args:
        series: Input series
        freq: Frequency for PeriodIndex

    Returns:
        Series with PeriodIndex
    """
    if not isinstance(series.index, pd.PeriodIndex):
        series = series.copy()
        series.index = pd.PeriodIndex(series.index, freq=freq)
    return series


def filter_date_range(
    df: pd.DataFrame,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Filter DataFrame to date range.

    Args:
        df: DataFrame with PeriodIndex
        start: Start period string
        end: End period string (optional)

    Returns:
        Filtered DataFrame
    """
    start_period = pd.Period(start, freq="Q")
    if end is not None:
        end_period = pd.Period(end, freq="Q")
        return df[(df.index >= start_period) & (df.index <= end_period)]
    return df[df.index >= start_period]
