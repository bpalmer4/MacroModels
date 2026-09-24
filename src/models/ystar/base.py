"""Base utilities for the ystar model.

Deliberately self-contained: this model does not import from `nairu`,
`rstar_hlw` or `expectations`. Its only dependency outside this package is
`src.data`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import pymc as pm

from src.models.common.model_constants import record_constant


@dataclass
class SamplerConfig:
    """Configuration for the PyMC NUTS sampler."""

    draws: int = 2_000
    tune: int = 2_000
    chains: int = 4
    cores: int = 4
    sampler: str = "numpyro"
    target_accept: float = 0.95
    # NumPyro's own default, stated here rather than inherited so that the
    # diagnostics can compare against the CONFIGURED cap: see `sample_model`,
    # which records it in the trace, and `_tree_depth_check`, which without it
    # can only compare against the deepest trajectory observed.
    #
    # RAISING IT TO 12 WAS TRIED AND DOES NOTHING. The argument was that 10 is
    # the wrong cap to pair with `target_accept` at 0.95, since high acceptance
    # is bought with a small step size and small steps need more leapfrog steps
    # to cover the same ground. rstar_bonds with the forward window showed 8.4%
    # of transitions at depth 10, which looked like truncation.
    #
    # It was not. At a cap of 12 the deepest trajectory is still 10 and mean
    # depth moves 9.08 to 9.10, so nothing was being truncated: these
    # trajectories U-turn at 10 of their own accord and 10 is simply what this
    # posterior costs. The 8.4% was the diagnostic reporting the share of draws
    # at the observed maximum, which only coincided with saturation because the
    # observed maximum happened to equal the cap.
    max_tree_depth: int = 10
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

    # `pm.sample` has no `max_treedepth` parameter of its own, and the two
    # samplers reach the setting by different names AND different routes.
    #
    # PyMC's own NUTS wants `max_treedepth`, travelling in `**kwargs` to the
    # step method. NumPyro wants `max_tree_depth`, and it is TWO levels down:
    # `nuts_sampler_kwargs` is forwarded to `sample_jax_nuts`, which does not
    # take the setting itself and passes only its `nuts_kwargs` on to the NUTS
    # kernel. Putting it one level up raises TypeError, which is how this was
    # found; putting the PyMC spelling anywhere on the NumPyro path would be
    # swallowed by `**kwargs` and silently ignored, leaving the cap at the
    # library default while the run looked healthy. Hence the explicit branch.
    if config.sampler == "pymc":
        depth_kwargs: dict[str, Any] = {"max_treedepth": config.max_tree_depth}
    else:
        depth_kwargs = {"nuts_sampler_kwargs": {"nuts_kwargs": {"max_tree_depth": config.max_tree_depth}}}

    with model:
        idata = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=config.cores,
            nuts_sampler=config.sampler,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            idata_kwargs={"log_likelihood": log_likelihood},
            **depth_kwargs,
        )

    # Record the cap beside the tree depths it applies to. Without it
    # `_tree_depth_check` has to treat the deepest observed trajectory as the
    # cap, which silently turns "these trajectories U-turn at 10" into "8.4% of
    # transitions were truncated at 10" — the same number whether or not
    # anything was truncated. NumPyro does not report `reached_max_treedepth`,
    # so this attribute is the only way to tell those apart. xarray attrs
    # survive the netCDF round trip, so it is readable from a saved trace.
    if hasattr(idata, "sample_stats"):
        idata.sample_stats.attrs["max_tree_depth"] = config.max_tree_depth
    return idata


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

    Fixed constants are recorded on the model (see `common.model_constants`).
    """
    if constant is None:
        constant = {}

    coefficients: dict[str, Any] = {}

    with model:
        for name, params in settings.items():
            if name in constant:
                coefficients[name] = constant[name]
                record_constant(model, name, constant[name])
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


def save_trace(trace: az.InferenceData, path: str | Path) -> None:
    """Save a trace to NetCDF."""
    trace.to_netcdf(str(path))


def load_trace(path: str | Path) -> az.InferenceData:
    """Load a trace from NetCDF."""
    return az.from_netcdf(str(path))
