"""Generic maximum likelihood estimation for DSGE models.

Provides ModelSpec dataclass and estimate_model() function that work with
any model that defines its specification. Each model file defines its own
SPEC constant and the generic estimator handles the rest.

Usage:
    from estimation import estimate_model
    from nairu_phillips_model import NAIRU_PHILLIPS_SPEC, load_nairu_phillips_data

    data = load_nairu_phillips_data(start="1984Q1")
    result = estimate_model(NAIRU_PHILLIPS_SPEC, data)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

# =============================================================================
# Model Specification
# =============================================================================


@dataclass
class ModelSpec:
    """Specification for a DSGE model.

    Encapsulates everything needed to estimate a model:
    - Parameter class and bounds
    - Which parameters to estimate vs fix
    - Likelihood function
    - State extractor for two-stage estimation

    Attributes:
        name: Model name for display
        param_class: Dataclass type for parameters (must have to_dict method)
        param_bounds: Dict mapping parameter names to (lower, upper) bounds
        estimate_params: List of parameter names to estimate
        likelihood_fn: Function(params, data) -> log_likelihood
        fixed_params: Dict of parameter name -> fixed value (not estimated)
        description: Optional description for output
        state_extractor_fn: Function(params, data) -> dict with 'states' DataFrame

    """

    name: str
    param_class: type
    param_bounds: dict[str, tuple[float, float]]
    estimate_params: list[str]
    likelihood_fn: Callable[[Any, dict], float]
    fixed_params: dict[str, float] = field(default_factory=dict)
    description: str = ""
    state_extractor_fn: Callable[[Any, dict], dict] | None = None


# =============================================================================
# Estimation Result
# =============================================================================


@dataclass
class EstimationResult:
    """Results from model estimation.

    Attributes:
        params: Estimated parameter object
        log_likelihood: Log-likelihood at optimum
        convergence: Dict with success, nit, nfev, message
        n_obs: Number of observations
        params_at_bounds: List of parameter names at bounds (if any)

    """

    params: Any
    log_likelihood: float
    convergence: dict
    n_obs: int
    params_at_bounds: list[str] = field(default_factory=list)


# =============================================================================
# Generic Estimator
# =============================================================================


def estimate_model(
    spec: ModelSpec,
    data: dict,
    initial_params: Any | None = None,
    maxiter: int = 500,
    verbose: bool = False,
) -> EstimationResult:
    """Estimate a model via maximum likelihood.

    Args:
        spec: ModelSpec defining the model
        data: Dict of data arrays (passed to likelihood_fn)
        initial_params: Initial parameter values (or None for defaults)
        maxiter: Maximum optimizer iterations
        verbose: Print progress

    Returns:
        EstimationResult with estimated parameters and diagnostics

    """
    if initial_params is None:
        initial_params = spec.param_class()

    # Build parameter dict from initial values
    param_dict = initial_params.to_dict()

    # Apply fixed parameters
    for name, value in spec.fixed_params.items():
        param_dict[name] = value

    # Initial values and bounds for estimated parameters
    x0 = np.array([param_dict[name] for name in spec.estimate_params])
    bounds = [spec.param_bounds[name] for name in spec.estimate_params]

    # Get n_obs from data
    n_obs = _get_n_obs(data)

    def neg_ll(x: np.ndarray) -> float:
        for i, name in enumerate(spec.estimate_params):
            param_dict[name] = x[i]
        params = spec.param_class(**param_dict)
        ll = spec.likelihood_fn(params, data)
        if verbose and np.isfinite(ll):
            print(f"  LL: {ll:.2f}")
        return -ll if np.isfinite(ll) else 1e10

    result = optimize.minimize(
        neg_ll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )

    # Extract final parameters
    for i, name in enumerate(spec.estimate_params):
        param_dict[name] = result.x[i]
    final_params = spec.param_class(**param_dict)

    # Check for parameters at bounds
    at_bounds = _check_bounds(result.x, spec.estimate_params, bounds)

    return EstimationResult(
        params=final_params,
        log_likelihood=-result.fun,
        convergence={
            "success": result.success,
            "nit": result.nit,
            "nfev": result.nfev,
            "message": getattr(result, "message", ""),
        },
        n_obs=n_obs,
        params_at_bounds=at_bounds,
    )


def _get_n_obs(data: dict) -> int:
    """Extract number of observations from data dict."""
    # Look for common keys
    for key in ["y", "y_obs", "dates"]:
        if key in data:
            return len(data[key])
    # Try first array value
    for v in data.values():
        if isinstance(v, np.ndarray):
            return len(v)
    return 0


def _check_bounds(
    x: np.ndarray,
    param_names: list[str],
    bounds: list[tuple[float, float]],
    tol: float = 0.01,
) -> list[str]:
    """Check which parameters are at or near their bounds."""
    at_bounds = []
    for i, (name, (lo, hi)) in enumerate(zip(param_names, bounds)):
        range_size = hi - lo
        if x[i] <= lo + tol * range_size:
            at_bounds.append(f"{name} (lower)")
        elif x[i] >= hi - tol * range_size:
            at_bounds.append(f"{name} (upper)")
    return at_bounds


# =============================================================================
# Two-Stage Estimation (exclude crisis, extract full states)
# =============================================================================


@dataclass
class TwoStageResult:
    """Results from two-stage estimation.

    Attributes:
        params: Estimated parameters (from filtered data)
        estimation_result: Full EstimationResult from stage 1
        states: DataFrame of smoothed states (from full data)
        estimation_dates: Dates used for parameter estimation
        full_dates: All dates (for state extraction)

    """

    params: Any
    estimation_result: EstimationResult
    states: pd.DataFrame
    estimation_dates: pd.PeriodIndex
    full_dates: pd.PeriodIndex


def exclude_period(data: dict, exclude_start: str, exclude_end: str) -> dict:
    """Exclude a date range from data dict.

    Args:
        data: Dict with 'dates' (PeriodIndex) and numpy arrays
        exclude_start: Start of exclusion period (e.g., "2008Q4")
        exclude_end: End of exclusion period (e.g., "2020Q4")

    Returns:
        New data dict with excluded period removed

    """
    dates = data.get("dates")
    if dates is None:
        raise ValueError("Data dict must have 'dates' key")

    exclude_start_period = pd.Period(exclude_start, freq="Q")
    exclude_end_period = pd.Period(exclude_end, freq="Q")

    # Mask for dates to KEEP
    mask = (dates < exclude_start_period) | (dates > exclude_end_period)

    filtered = {}
    for key, value in data.items():
        if key == "dates":
            filtered[key] = dates[mask]
        elif isinstance(value, np.ndarray):
            filtered[key] = value[mask]
        else:
            filtered[key] = value

    return filtered


def estimate_two_stage(
    spec: ModelSpec,
    load_data_fn: Callable[..., dict],
    exclude_start: str = "2008Q4",
    exclude_end: str = "2020Q4",
    start: str = "1984Q1",
    end: str | None = None,
    verbose: bool = True,
    **load_kwargs,
) -> TwoStageResult:
    """Two-stage estimation: parameters on clean data, states on full data.

    Stage 1: Estimate structural parameters excluding crisis years
    Stage 2: Extract smoothed states for ALL periods using those parameters

    Args:
        spec: ModelSpec (must have state_extractor_fn defined)
        load_data_fn: Function(start, end, **kwargs) -> data dict
        exclude_start: Start of exclusion period (default: "2008Q4")
        exclude_end: End of exclusion period (default: "2020Q4")
        start: Data start date
        end: Data end date (None = latest)
        verbose: Print progress
        **load_kwargs: Additional kwargs passed to load_data_fn

    Returns:
        TwoStageResult with estimated params and full states

    """
    if spec.state_extractor_fn is None:
        raise ValueError(f"Model {spec.name} has no state_extractor_fn defined")

    # Load full data
    full_data = load_data_fn(start=start, end=end, **load_kwargs)
    full_dates = full_data["dates"]

    # Filter out crisis years for estimation
    filtered_data = exclude_period(full_data, exclude_start, exclude_end)
    estimation_dates = filtered_data["dates"]

    if verbose:
        print(f"\n{spec.name} - Two-Stage Estimation")
        if spec.description:
            print(f"  ({spec.description})")
        print(f"  Stage 1: Estimate parameters excluding {exclude_start} to {exclude_end}")
        print(f"    Using: {estimation_dates[0]} to {estimation_dates[-1]} (n={len(estimation_dates)})")

    # Stage 1: Estimate on filtered data
    est_result = estimate_model(spec, filtered_data)

    if verbose:
        print(f"    Log-likelihood: {est_result.log_likelihood:.2f}")
        if est_result.params_at_bounds:
            print(f"    Warning: at bounds: {', '.join(est_result.params_at_bounds)}")

    # Stage 2: Extract states on full data
    if verbose:
        print("  Stage 2: Extract states for full period")
        print(f"    Using: {full_dates[0]} to {full_dates[-1]} (n={len(full_dates)})")

    state_result = spec.state_extractor_fn(est_result.params, full_data)
    states_df = state_result["states"]

    if verbose:
        print(f"    States extracted: {list(states_df.columns)}")

    return TwoStageResult(
        params=est_result.params,
        estimation_result=est_result,
        states=states_df,
        estimation_dates=estimation_dates,
        full_dates=full_dates,
    )


# =============================================================================
# Result Printing
# =============================================================================


def print_single_result(
    result: EstimationResult,
    spec: ModelSpec,
    params_to_show: list[str] | None = None,
) -> None:
    """Print a single estimation result.

    Args:
        result: EstimationResult to print
        spec: ModelSpec for the model
        params_to_show: List of parameter names to display

    """
    if params_to_show is None:
        params_to_show = spec.estimate_params

    print("=" * 50)
    print(f"{spec.name} ESTIMATION RESULT")
    print("=" * 50)

    print(f"\nObservations: {result.n_obs}")
    print(f"Log-likelihood: {result.log_likelihood:.2f}")
    print(f"Converged: {'Yes' if result.convergence.get('success') else 'No'}")
    print(f"Iterations: {result.convergence.get('nit', 'N/A')}")

    print("\nParameters:")
    for param in params_to_show:
        if hasattr(result.params, param):
            val = getattr(result.params, param)
            at_bound = param in " ".join(result.params_at_bounds)
            marker = " *" if at_bound else ""
            print(f"  {param:<20} = {val:>10.4f}{marker}")

    if result.params_at_bounds:
        print(f"\n* At bounds: {', '.join(result.params_at_bounds)}")

    print("=" * 50)
