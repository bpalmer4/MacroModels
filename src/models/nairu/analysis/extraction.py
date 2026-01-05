"""Extract variables from PyMC traces.

Re-exports from src.models.common.extraction for backwards compatibility.
"""

from src.models.common.extraction import (
    get_scalar_var,
    get_scalar_var_names,
    get_vector_var,
    is_scalar_var,
)

__all__ = [
    "get_scalar_var",
    "get_scalar_var_names",
    "get_vector_var",
    "is_scalar_var",
]
