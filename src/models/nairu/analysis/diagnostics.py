"""MCMC diagnostics for PyMC models.

Re-exports from src.models.common.diagnostics for backwards compatibility.
"""

from src.models.common.diagnostics import check_for_zero_coeffs, check_model_diagnostics

__all__ = [
    "check_for_zero_coeffs",
    "check_model_diagnostics",
]
