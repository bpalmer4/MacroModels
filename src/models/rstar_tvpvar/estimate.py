"""Build, sample, and persist the TVP-VAR.

The state is large and almost all of it is nuisance: with 3 variables and 2
lags there are 21 coefficients per quarter, so on 135 quarters the model
carries 2,835 drifting coefficients plus 405 log variances. NUTS copes because
every random walk is NON-CENTRED — the sampler sees standard normals and the
scale is a separate parameter — which is the difference between this sampling
and not.

r* IS NOT COMPUTED HERE. It is a non-linear function of the coefficients (iterate
the VAR forward H quarters), and putting it in the graph would cost a 6x6 matrix
power at every quarter of every draw for no benefit. `results.py` computes it
from the saved posterior in numpy, which also means the horizon can be changed
without re-sampling.
"""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.rstar_tvpvar.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.rstar_tvpvar.observations import build_observations, design_matrix, ols_fit, ordering
from src.models.ystar.base import SamplerConfig, get_fixed_constants, sample_model


def _drift_scale(config: ModelConfig, n_coef: int) -> tuple[Any, str]:
    """Return the coefficient-drift sd, and a label for the run log.

    Shared across all coefficients so one number governs the total amount of
    time variation and can be swept. See `ModelConfig.sigma_q_prior` for why
    this is the parameter that decides the answer.
    """
    del n_coef
    if config.sigma_q is None:
        return pm.HalfNormal("sigma_q", sigma=config.sigma_q_prior), "estimated"
    return pt.as_tensor_variable(float(config.sigma_q)), f"imposed at {config.sigma_q:g}"


def build_model(
    data: np.ndarray,
    config: ModelConfig | None = None,
    verbose: bool = True,
) -> pm.Model:
    """Build the TVP-VAR with stochastic volatility."""
    if config is None:
        config = ModelConfig()

    design, target, mask = design_matrix(data, config.lags)
    n_periods, n_coef = design.shape
    n_vars = target.shape[1]
    descriptions: list[str] = []

    # The initial coefficients get a prior centred on a constant-coefficient
    # OLS fit, which is how a Primiceri TVP-VAR starts. On a TRAINING SAMPLE by
    # default: fitting it on the whole estimation sample centres the prior on
    # the very constant-coefficient answer the drift is being tested against.
    # See `ModelConfig.training_sample_quarters`.
    training = config.training_sample_quarters
    ols, log_var0 = ols_fit(data, config.lags, training)
    training_label = f"first {training} quarters" if training is not None else "full sample"

    model = pm.Model()
    with model:
        model._fixed_constants = dict(config.constants)  # noqa: SLF001 — our own metadata, as ystar.base does

        # --- Coefficients: a non-centred random walk per coefficient ---
        theta_0 = pm.Normal("theta_0", mu=ols, sigma=1.0, shape=(n_coef, n_vars))
        sigma_q, sigma_q_label = _drift_scale(config, n_coef)
        eta = pm.Normal("eta", 0.0, 1.0, shape=(n_periods, n_coef, n_vars))
        theta = pm.Deterministic("theta", theta_0 + sigma_q * pt.cumsum(eta, axis=0))
        descriptions.append(f"Coefficients: Theta_t = Theta_0 + sigma_q · cumsum(eta), sigma_q {sigma_q_label}")
        descriptions.append(f"Theta_0:      centred on an OLS fit over the {training_label}")

        # --- Stochastic volatility: a non-centred random walk per log variance ---
        h_0 = pm.Normal("h_0", mu=log_var0, sigma=config.h0_sigma, shape=n_vars)
        sigma_h = pm.HalfNormal("sigma_h", sigma=config.sigma_h_prior, shape=n_vars)
        xi = pm.Normal("xi", 0.0, 1.0, shape=(n_periods, n_vars))
        log_h = pm.Deterministic("log_h", h_0 + sigma_h * pt.cumsum(xi, axis=0))
        descriptions.append("Volatility:   log h_t = log h_{t-1} + sigma_h · xi_t   (per variable)")

        # --- Contemporaneous structure ---
        # Lower triangular, unit diagonal. |det A| = 1, so the change of
        # variables from e to A·e needs no Jacobian term in the likelihood.
        n_free = n_vars * (n_vars - 1) // 2
        a_free = pm.Normal("a_free", 0.0, config.a_sigma, shape=n_free)
        rows, cols = np.tril_indices(n_vars, k=-1)
        a_matrix = pt.set_subtensor(pt.eye(n_vars)[rows, cols], a_free)
        descriptions.append(f"Structure:    A lower triangular, unit diagonal, {n_free} free elements")

        # --- The VAR residuals, and the likelihood ---
        # einsum over the time axis: fitted[t, j] = sum_k design[t, k] theta[t, k, j]
        fitted = pt.sum(design[:, :, None] * theta, axis=1)
        resid_t = pt.as_tensor_variable(target) - fitted
        # A is applied on the right because `resid_t` is (time, variable), so
        # this is (A e_t)' for each t.
        orthogonal = resid_t @ a_matrix.T
        scale = pt.exp(0.5 * log_h)

        # The mask enters as a weight on the per-observation log density rather
        # than by dropping rows, so the time index stays aligned with the
        # calendar and the drifting states still connect across a gap.
        logp = pm.logp(pm.Normal.dist(0.0, scale), orthogonal)
        pm.Potential("var_likelihood", (logp * mask[:, None]).sum())
        descriptions.append(f"Likelihood:   A·e_t ~ N(0, diag(exp(h_t))) on {int(mask.sum())} of {n_periods} rows")

    if verbose:
        print("\nModel specification:")
        for line in descriptions:
            print(f"  {line}")
        print(f"  States:       {n_periods * n_coef * n_vars:,} drifting coefficients, "
              f"{n_periods * n_vars:,} log variances")
        print()

    return model


def save_results(
    trace: az.InferenceData,
    data: np.ndarray,
    obs_index: pd.PeriodIndex,
    constants: dict[str, Any],
    *,
    frame: pd.DataFrame | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "rstar_tvpvar",
) -> Path:
    """Persist trace and observations to `model_outputs`."""
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / f"{prefix}_trace.nc"
    trace.to_netcdf(str(trace_path))
    print(f"Saved trace to: {trace_path}")

    obs_path = output_dir / f"{prefix}_obs.pkl"
    with obs_path.open("wb") as f:
        pickle.dump(
            {"data": data, "obs_index": obs_index, "constants": constants, "frame": frame},
            f,
        )
    print(f"Saved observations to: {obs_path}")
    return output_dir


def run_estimate(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    prefix: str = "rstar_tvpvar",
    verbose: bool = False,
    seed: int | None = None,
) -> tuple[az.InferenceData, np.ndarray, pd.PeriodIndex]:
    """Build observations, sample the posterior, save the results."""
    if config is None:
        config = ModelConfig()
    if sampler_config is None:
        sampler_config = SamplerConfig()
    if seed is not None:
        sampler_config.random_seed = seed
    # There is no single observed RV to attach pointwise densities to, since the
    # likelihood is a Potential over a masked matrix, so LOO/WAIC is unavailable
    # here and asking for it only costs memory.
    sampler_config.log_likelihood = False

    drift = "estimated" if config.sigma_q is None else f"IMPOSED at {config.sigma_q:g}"
    print(f"Sample:       {config.start} -> {config.end or 'latest'}")
    # The run's OWN ordering, not the full five-variable tuple: with both
    # switches off this is the three-variable Lubik-Matthes set, and printing
    # `VARIABLES` logged "5 variables" for a run that estimated three.
    order = ordering(include_commodities=config.include_commodities, include_twi=config.include_twi)
    print(f"VAR:          {len(order)} variables, {config.lags} lags, {config.basis} basis")
    print(f"Variables:    {', '.join(order)}")
    print(f"Deflator:     {config.deflator}")
    print(f"r* horizon:   {config.horizon_quarters} quarters ahead")
    print(f"r* is:        {config.rstar_definition}, inflation conditioning "
          f"{'ON' if config.anchor_projection else 'off'}")
    training = config.training_sample_quarters
    print(f"Theta_0 prior: OLS over {f'the first {training} quarters' if training else 'the full sample'}")
    print(f"sigma_q:      {drift}, prior HalfNormal({config.sigma_q_prior:g})")
    blanked = config.blanked_quarters
    print(f"Blanked:      {', '.join(blanked) if blanked else 'nothing (all quarters in the likelihood)'}")
    print(f"Sampler seed: {sampler_config.random_seed}")

    print("\nBuilding observations...")
    data, obs_index, frame, sources = build_observations(
        start=config.start,
        end=config.end,
        basis=config.basis,
        deflator=config.deflator,
        include_commodities=config.include_commodities,
        include_twi=config.include_twi,
        blank_quarters=config.blanked_quarters,
        verbose=verbose,
    )

    print("Building model...")
    model = build_model(data, config=config)

    print("Sampling...")
    trace = sample_model(model, sampler_config)
    print()

    save_results(
        trace, data, obs_index,
        constants={**get_fixed_constants(model), "sources": sources.to_records()},
        frame=frame,
        output_dir=config.output_dir,
        prefix=prefix,
    )
    return trace, data, obs_index
