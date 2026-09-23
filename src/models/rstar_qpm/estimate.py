"""Build, sample and persist the rstar_qpm model.

The states never enter the sampler. NUTS sees 21 scalars and a Kalman-filter
likelihood; the state paths are drawn afterwards, one simulation-smoother pass
per retained posterior draw, so their bands carry both parameter and state
uncertainty.
"""

import json
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from src.models.rstar_qpm.config import ModelConfig, Prior
from src.models.rstar_qpm.neutral import short_run_neutral
from src.models.rstar_qpm.observations import build_observations
from src.models.rstar_qpm.state_space import (
    QGAP,
    QSTAR,
    YGAP,
    YSTAR,
    G,
    Prepared,
    W,
    prepare,
    pytensor_loglik,
    simulation_smoother,
)
from src.models.ystar.base import SamplerConfig, sample_model

# The smallest horizon at which a rate change can reach the gap: the IS curve
# is lagged, so a rate set from t+1 first moves the gap at t+2.
MIN_HORIZON = 2


def _prior(name: str, prior: Prior) -> pm.Distribution:
    """Create one prior. Must be called inside the model context."""
    if prior.kind == "beta":
        return pm.Beta(name, alpha=prior.mu, beta=prior.sd)
    if prior.kind == "trunc":
        return pm.TruncatedNormal(name, mu=prior.mu, sigma=prior.sd, lower=0.0)
    if prior.kind == "half":
        return pm.HalfNormal(name, sigma=prior.sd)
    if prior.kind == "normal":
        return pm.Normal(name, mu=prior.mu, sigma=prior.sd)
    if prior.kind == "invgamma":
        alpha, beta = prior.invgamma_shape()
        return pm.InverseGamma(name, alpha=alpha, beta=beta)
    raise ValueError(f"unknown prior kind {prior.kind!r} for {name}")


def build_model(prep: Prepared, config: ModelConfig) -> pm.Model:
    """Build the priors, with the filter's log-likelihood as a Potential."""
    with pm.Model() as model:
        params: dict[str, Any] = {name: _prior(name, prior) for name, prior in config.estimated_priors.items()}
        params.update(config.fixed)
        pm.Potential("loglik", pytensor_loglik(prep, params))
    return model


def parameter_names(priors: dict[str, Prior]) -> list[str]:
    """Every scalar a draw reads from the trace: the sampled parameters."""
    return list(priors)


def posterior_params(trace: az.InferenceData, names: list[str], n_draws: int) -> list[dict[str, float]]:
    """Return `n_draws` parameter dicts spread evenly through the posterior."""
    posterior = getattr(trace, "posterior", None)
    if not isinstance(posterior, xr.Dataset):
        raise TypeError("trace has no posterior group")
    stacked = {name: np.asarray(posterior[name].values).ravel() for name in names}
    total = len(next(iter(stacked.values())))
    picks = np.linspace(0, total - 1, min(n_draws, total)).astype(int)
    return [{name: float(stacked[name][k]) for name in names} for k in picks]


def draw_states(
    trace: az.InferenceData,
    prep: Prepared,
    config: ModelConfig,
    seed: int,
) -> dict[str, Any]:
    """Push retained draws through the simulation smoother and short-run neutral.

    Returns time x draw frames for each path, plus the multiplier per draw.
    """
    if config.horizon < MIN_HORIZON:
        raise ValueError(f"horizon must be at least {MIN_HORIZON}: the IS curve is lagged")
    rng = np.random.default_rng(seed)
    draws = [{**p, **config.fixed} for p in
             posterior_params(trace, parameter_names(config.estimated_priors), config.state_draws)]
    paths: dict[str, list[np.ndarray]] = {
        k: [] for k in ("ystar", "g", "ygap", "wedge", "rstar", "qstar", "qgap", "srn")
    }
    multipliers = []
    rw = prep.data["rw"]
    for p in draws:
        states = simulation_smoother(prep, p, rng)
        if config.use_is:
            srn, multiplier = short_run_neutral(states, p, prep, config.horizon)
        else:
            # No IS curve, so no rate closes the gap: short-run neutral and
            # its multiplier do not exist.
            srn, multiplier = np.full(len(prep.index), np.nan), np.full(len(prep.index), np.nan)
        paths["ystar"].append(states[:, YSTAR])
        paths["g"].append(states[:, G])
        paths["ygap"].append(states[:, YGAP])
        paths["wedge"].append(states[:, W])
        paths["rstar"].append(rw + states[:, W])
        paths["qstar"].append(states[:, QSTAR])
        paths["qgap"].append(states[:, QGAP])
        paths["srn"].append(srn)
        # B varies by quarter with debt exposure; the latest quarter's is kept.
        multipliers.append(float(multiplier[-1]))
    frames = {k: pd.DataFrame(np.column_stack(v), index=prep.index) for k, v in paths.items()}
    return {"paths": frames, "multiplier": np.asarray(multipliers), "horizon": config.horizon}


def _paths(prefix: str, directory: Path) -> dict[str, Path]:
    return {
        "trace": directory / f"{prefix}_trace.nc",
        "states": directory / f"{prefix}_states.nc",
    }


def save_results(
    trace: az.InferenceData,
    *,
    frame: pd.DataFrame,
    constants: dict[str, Any],
    states: dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> None:
    """Persist the trace, and beside it the observations, state draws and constants.

    netCDF rather than pickle for the second file: it holds only arrays and
    text, so nothing executable is ever loaded back.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _paths(prefix, output_dir)
    trace.to_netcdf(str(files["trace"]))

    quarters = [str(q) for q in frame.index]
    data_vars: dict[str, Any] = {
        f"path_{name}": (("quarter", "draw"), paths.to_numpy()) for name, paths in states["paths"].items()
    }
    data_vars["multiplier"] = (("draw",), states["multiplier"])
    for col in frame.columns:
        data_vars[f"obs_{col}"] = (("quarter",), frame[col].to_numpy(dtype=float))
    dataset = xr.Dataset(
        data_vars,
        coords={"quarter": quarters, "draw": np.arange(len(states["multiplier"]))},
        attrs={"constants": json.dumps(constants), "horizon": int(states["horizon"])},
    )
    dataset.to_netcdf(str(files["states"]))
    print(f"\nSaved trace to: {files['trace']}")


def load_results(
    prefix: str = "rstar_qpm",
    output_dir: Path | None = None,
) -> tuple[az.InferenceData, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load a completed run: trace, observations, constants, state draws."""
    files = _paths(prefix, output_dir or ModelConfig().output_dir)
    trace = az.from_netcdf(str(files["trace"]))
    with xr.open_dataset(files["states"]) as dataset:
        dataset.load()
    index = pd.PeriodIndex([str(q) for q in dataset["quarter"].to_numpy()], freq="Q")
    frame = pd.DataFrame(
        {str(name)[4:]: dataset[name].to_numpy() for name in dataset.data_vars if str(name).startswith("obs_")},
        index=index,
    )
    paths = {
        str(name)[5:]: pd.DataFrame(dataset[name].to_numpy(), index=index)
        for name in dataset.data_vars if str(name).startswith("path_")
    }
    constants = json.loads(str(dataset.attrs["constants"]))
    states = {
        "paths": paths,
        "multiplier": dataset["multiplier"].to_numpy(),
        "horizon": int(dataset.attrs["horizon"]),
    }
    return trace, frame, constants, states


def estimate(
    frame: pd.DataFrame,
    config: ModelConfig,
    sampler_config: SamplerConfig,
) -> tuple[az.InferenceData, dict[str, Any]]:
    """Sample one frame and draw its states. No saving, so tests can reuse it."""
    prep = prepare(frame, config)
    model = build_model(prep, config)
    trace = sample_model(model, sampler_config)
    states = draw_states(trace, prep, config, seed=sampler_config.random_seed)
    return trace, states


def run_estimate(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    prefix: str = "rstar_qpm",
    *,
    verbose: bool = False,
) -> az.InferenceData:
    """Build the observations, sample, draw the states, and save."""
    config = config or ModelConfig()
    sampler_config = sampler_config or SamplerConfig()
    frame, sources = build_observations(config, verbose=True)
    if verbose:
        for name, prior in config.priors.items():
            print(f"  prior {name:14s} {prior.kind:6s} {prior.mu:g}, {prior.sd:g}")
    print("\nSampling...")
    trace, states = estimate(frame, config, sampler_config)
    save_results(
        trace, frame=frame,
        constants={**config.constants, "sources": sources.to_records()},
        states=states, output_dir=config.output_dir, prefix=prefix,
    )
    return trace
