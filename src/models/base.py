"""Base utilities for PyMC model building and sampling."""

from dataclasses import dataclass, field
from pathlib import Path
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

    draws: int = 100_000
    tune: int = 5_000
    chains: int = 6
    cores: int = 6
    sampler: str = "numpyro"
    target_accept: float = 0.95
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
        trace = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=config.cores,
            nuts_sampler=config.sampler,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
        )

    return trace


def set_model_coefficients(
    model: pm.Model,
    settings: dict[str, dict[str, float]],
    constant: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Create model coefficients from settings, allowing fixed constants.

    For each coefficient in settings:
    - If the coefficient name is in `constant`, use that fixed value
    - Otherwise, create a Normal prior with the specified mu and sigma

    Args:
        model: PyMC model context
        settings: Dict of {coef_name: {"mu": float, "sigma": float}}
        constant: Dict of {coef_name: fixed_value} for parameters to fix

    Returns:
        Dict of {coef_name: pm.Distribution or fixed value}

    Example:
        settings = {
            "beta": {"mu": 0.5, "sigma": 0.1},
            "gamma": {"mu": -0.3, "sigma": 0.2},
        }
        constant = {"beta": 0.5}  # Fix beta at 0.5

        mc = set_model_coefficients(model, settings, constant)
        # mc["beta"] = 0.5 (fixed)
        # mc["gamma"] = pm.Normal("gamma", mu=-0.3, sigma=0.2)
    """
    if constant is None:
        constant = {}

    coefficients = {}

    with model:
        for name, params in settings.items():
            if name in constant:
                coefficients[name] = constant[name]
            elif "sigma" in params and "mu" not in params:
                # HalfNormal for scale parameters (sigma only, no mu)
                coefficients[name] = pm.HalfNormal(name, sigma=params["sigma"])
            else:
                coefficients[name] = pm.Normal(
                    name,
                    mu=params.get("mu", 0),
                    sigma=params.get("sigma", 1),
                )

    return coefficients


def save_trace(trace: az.InferenceData, path: str | Path) -> None:
    """Save trace to NetCDF file.

    Args:
        trace: ArviZ InferenceData to save
        path: Output file path (.nc extension recommended)
    """
    trace.to_netcdf(str(path))


def load_trace(path: str | Path) -> az.InferenceData:
    """Load trace from NetCDF file.

    Args:
        path: Path to NetCDF file

    Returns:
        ArviZ InferenceData
    """
    return az.from_netcdf(str(path))
