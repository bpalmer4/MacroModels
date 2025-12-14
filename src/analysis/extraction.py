"""Extract variables from PyMC traces."""

import arviz as az
import numpy as np
import pandas as pd


def get_vector_var(var_name: str, trace: az.InferenceData) -> pd.DataFrame:
    """Extract chains/draws for a vector variable.

    Returns DataFrame with rows=time periods, columns=samples.
    """
    return (
        az.extract(trace, var_names=var_name)
        .transpose("sample", ...)
        .to_dataframe()[var_name]
        .unstack(level=2)
        .T
    )


def get_scalar_var(var_name: str, trace: az.InferenceData) -> pd.Series:
    """Extract chains/draws for a scalar variable.

    Returns Series of posterior samples.
    """
    return az.extract(trace, var_names=var_name).to_dataframe()[var_name]


def is_scalar_var(var_name: str, trace: az.InferenceData) -> bool:
    """Check if a variable in the trace is scalar (not a vector/time series).

    Returns True if the variable has only (chain, draw) dimensions,
    False if it has additional dimensions (e.g., time steps).
    """
    var_data = trace.posterior[var_name]
    return set(var_data.dims) == {"chain", "draw"}


def get_scalar_var_names(trace: az.InferenceData) -> list[str]:
    """Get list of all scalar variable names in the trace.

    Returns list of variable names that are scalars (not vectors).
    """
    return [
        var_name
        for var_name in trace.posterior.data_vars
        if is_scalar_var(var_name, trace)
    ]


def check_for_zero_coeffs(
    trace: az.InferenceData,
    critical_params: list[str] | None = None,
) -> pd.DataFrame:
    """Check scalar parameters for coefficients indistinguishable from zero.

    Automatically detects scalar variables (excludes vector/time series variables).
    Shows quantiles and flags parameters that may be indistinguishable from zero.

    Args:
        trace: InferenceData from model fitting
        critical_params: List of parameter names that are critical (warn if any
            quantile crosses zero). If None, uses default threshold of 2+ crossings.

    Returns:
        DataFrame with quantiles and significance markers.
    """
    from IPython.display import display

    if critical_params is None:
        critical_params = []

    q = [0.01, 0.05, 0.10, 0.25, 0.50]
    q_tail = [1 - x for x in q[:-1]][::-1]
    q = q + q_tail

    scalar_vars = get_scalar_var_names(trace)

    quantiles = {
        var_name: get_scalar_var(var_name, trace).quantile(q)
        for var_name in scalar_vars
    }

    df = pd.DataFrame(quantiles).T.sort_index()
    problem_intensity = (
        pd.DataFrame(np.sign(df.T))
        .apply([lambda x: x.lt(0).sum(), lambda x: x.ge(0).sum()])
        .min()
        .astype(int)
    )
    marker = pd.Series(["*"] * len(problem_intensity), index=problem_intensity.index)
    markers = (
        marker.str.repeat(problem_intensity).reindex(problem_intensity.index).fillna("")
    )
    df["Check Significance"] = markers

    for param in df.index:
        if param in problem_intensity:
            stars = problem_intensity[param]
            if (stars > 0 if param in critical_params else stars > 2):
                print(
                    f"*** WARNING: Parameter '{param}' may be indistinguishable from zero "
                    f"({stars} stars). Check model specification! ***"
                )

    print("=" * 20)

    return df
