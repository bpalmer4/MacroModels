"""Base utilities for PyMC model building and sampling."""

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used at runtime in function signatures
from typing import Any

import arviz as az
import pymc as pm


@dataclass
class SamplerConfig:
    """Configuration for PyMC NUTS sampler.

    Attributes:
        draws: Number of samples to draw after tuning
        tune: Number of tuning samples
        chains: Number of independent chains
        cores: Number of CPU cores to use
        sampler: NUTS implementation ("pymc", "numpyro", or "blackjax")
        target_accept: Target acceptance probability (higher = more conservative)
        random_seed: Random seed for reproducibility

    """

    draws: int = 10_000
    tune: int = 3_500
    chains: int = 5
    cores: int = 5
    sampler: str = "numpyro"
    target_accept: float = 0.90
    random_seed: int = 42


def sample_model(
    model: pm.Model,
    config: SamplerConfig | None = None,
) -> az.InferenceData:
    """Sample from a PyMC model using NUTS.

    Args:
        model: PyMC model to sample from
        config: Sampler configuration (uses defaults if None)

    Returns:
        ArviZ InferenceData with posterior samples

    """
    if config is None:
        config = SamplerConfig()

    with model:
        return pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=config.cores,
            nuts_sampler=config.sampler,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
        )


def set_model_coefficients(
    model: pm.Model,
    settings: dict[str, dict[str, float]],
    constant: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Create model coefficients from settings, allowing fixed constants.

    For each coefficient in settings:
    - If the coefficient name is in `constant`, use that fixed value
    - Otherwise, create a prior based on the settings:
      - sigma only (no mu): HalfNormal
      - lower or upper specified: TruncatedNormal
      - otherwise: Normal

    Fixed constants are accumulated on model._fixed_constants for later retrieval.

    Args:
        model: PyMC model context
        settings: Dict of {coef_name: {"mu": float, "sigma": float, ...}}
        constant: Dict of {coef_name: fixed_value} for parameters to fix

    Returns:
        Dict of {coef_name: pm.Distribution or fixed value}

    """
    if constant is None:
        constant = {}

    if not hasattr(model, "_fixed_constants"):
        model._fixed_constants = {}  # noqa: SLF001 — our own metadata on PyMC model

    coefficients = {}

    with model:
        for name, params in settings.items():
            if name in constant:
                coefficients[name] = constant[name]
                model._fixed_constants[name] = constant[name]  # noqa: SLF001
            elif "sigma" in params and "mu" not in params:
                coefficients[name] = pm.HalfNormal(name, sigma=params["sigma"])
            elif "lower" in params or "upper" in params:
                coefficients[name] = pm.TruncatedNormal(
                    name,
                    mu=params.get("mu", 0),
                    sigma=params.get("sigma", 1),
                    lower=params.get("lower"),
                    upper=params.get("upper"),
                )
            else:
                coefficients[name] = pm.Normal(
                    name,
                    mu=params.get("mu", 0),
                    sigma=params.get("sigma", 1),
                )

    return coefficients


def get_fixed_constants(model: pm.Model) -> dict[str, Any]:
    """Get all fixed constants from a model."""
    return getattr(model, "_fixed_constants", {})


def save_trace(trace: az.InferenceData, path: str | Path) -> None:
    """Save trace to NetCDF file."""
    trace.to_netcdf(str(path))


def load_trace(path: str | Path) -> az.InferenceData:
    """Load trace from NetCDF file."""
    return az.from_netcdf(str(path))
