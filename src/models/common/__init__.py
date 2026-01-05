"""Common utilities for PyMC models.

Provides shared functionality for:
- Trace extraction (extraction.py)
- MCMC diagnostics (diagnostics.py)
- Posterior time series plotting (timeseries.py)
"""

from src.models.common.diagnostics import check_for_zero_coeffs, check_model_diagnostics
from src.models.common.extraction import (
    get_scalar_var,
    get_scalar_var_names,
    get_vector_var,
    is_scalar_var,
)
from src.models.common.timeseries import plot_posterior_timeseries

__all__ = [
    "check_for_zero_coeffs",
    "check_model_diagnostics",
    "get_scalar_var",
    "get_scalar_var_names",
    "get_vector_var",
    "is_scalar_var",
    "plot_posterior_timeseries",
]
