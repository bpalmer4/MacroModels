"""Analysis and diagnostics for expectations signal extraction model."""

from src.models.common.diagnostics import check_for_zero_coeffs, check_model_diagnostics
from src.models.common.timeseries import plot_posterior_timeseries
from src.models.expectations.analysis.validation import (
    compare_to_rba,
    plot_validation,
    print_validation_summary,
    validation_statistics,
)

__all__ = [
    # Diagnostics (from common)
    "check_model_diagnostics",
    "check_for_zero_coeffs",
    # Plotting (from common)
    "plot_posterior_timeseries",
    # Validation
    "compare_to_rba",
    "plot_validation",
    "print_validation_summary",
    "validation_statistics",
]
