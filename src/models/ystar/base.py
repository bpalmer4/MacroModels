"""Base utilities for the ystar model.

Deliberately self-contained: this model does not import from `nairu`,
`rstar_hlw` or `expectations`. Its only dependency outside this package is
`src.data`.
"""

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used at runtime in function signatures
from typing import Any

import arviz as az
import pymc as pm


@dataclass
class SamplerConfig:
    """Configuration for the PyMC NUTS sampler."""

    draws: int = 2_000
    tune: int = 2_000
    chains: int = 4
    cores: int = 4
    sampler: str = "numpyro"
    target_accept: float = 0.95
    random_seed: int = 42
    # Store pointwise log likelihood in the trace, so variants can be ranked
    # with LOO/WAIC instead of by eyeballing coefficients. Only compare runs
    # that observe the SAME data: `--no-phillips` drops an observed variable,
    # so it is not comparable with the full model, while the sigma sweeps are.
    log_likelihood: bool = True


def sample_model(model: pm.Model, config: SamplerConfig | None = None) -> az.InferenceData:
    """Sample from a PyMC model using NUTS."""
    if config is None:
        config = SamplerConfig()

    # A model whose likelihood is written entirely with `pm.Potential` has no
    # observed RVs, so there are no pointwise contributions to store and LOO or
    # WAIC would be meaningless on it anyway. `rstar` is that model. Asking for
    # the log likelihood there does not degrade gracefully: PyMC's JAX path
    # returns None where it expects a list and raises a TypeError.
    log_likelihood = config.log_likelihood and bool(model.observed_RVs)

    with model:
        return pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=config.cores,
            nuts_sampler=config.sampler,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            idata_kwargs={"log_likelihood": log_likelihood},
        )


def set_model_coefficients(
    model: pm.Model,
    settings: dict[str, dict[str, float]],
    constant: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Create model coefficients from settings, allowing fixed constants.

    For each coefficient in `settings`:
    - if the name appears in `constant`, use that fixed value;
    - sigma only (no mu): HalfNormal;
    - lower or upper given: TruncatedNormal;
    - otherwise: Normal.

    Fixed constants accumulate on `model._fixed_constants` for later retrieval.
    """
    if constant is None:
        constant = {}

    if not hasattr(model, "_fixed_constants"):
        model._fixed_constants = {}  # noqa: SLF001 — our own metadata on the PyMC model

    coefficients: dict[str, Any] = {}

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
    """Return all fixed constants recorded on a model."""
    return getattr(model, "_fixed_constants", {})


def save_trace(trace: az.InferenceData, path: str | Path) -> None:
    """Save a trace to NetCDF."""
    trace.to_netcdf(str(path))


def load_trace(path: str | Path) -> az.InferenceData:
    """Load a trace from NetCDF."""
    return az.from_netcdf(str(path))
