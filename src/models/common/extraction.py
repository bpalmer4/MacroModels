"""Extract variables from PyMC traces."""

import arviz as az
import pandas as pd

from src.models.common.results import vector_draws


def get_vector_var(var_name: str, trace: az.InferenceData) -> pd.DataFrame:
    """Extract chains/draws for a vector variable.

    Returns DataFrame with rows=time periods, columns=samples.

    Takes a trace and a name, where `vector_draws` takes the DataArray, so the
    two are not interchangeable at the call site. The flattening itself is
    shared: two routes to it existed and returned the same draws in the same
    order, which is one route too many to keep in step.
    """
    return vector_draws(trace.posterior[var_name])


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
