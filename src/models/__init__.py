"""PyMC model assembly and sampling utilities.

Models:
- cobb_douglas: Deterministic growth accounting (Solow residual)
- nairu_output_gap: Bayesian state-space estimation
"""

from src.models.base import SamplerConfig, sample_model, set_model_coefficients

__all__ = [
    "SamplerConfig",
    "sample_model",
    "set_model_coefficients",
]
