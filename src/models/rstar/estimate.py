"""Build, sample, and persist the rstar model."""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.rstar.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.rstar.observations import build_observations
from src.models.ystar.base import (
    SamplerConfig,
    get_fixed_constants,
    sample_model,
    set_model_coefficients,
)


def _break_design(obs_index: pd.PeriodIndex, breaks: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """Return the step design matrix and the break labels that fall in sample.

    Column k is 1 from break k onward and 0 before, so `wedge_0 + D @ jumps` is
    a step function. Breaks outside the sample are dropped with a note rather
    than silently ignored: a missing one changes the specification.
    """
    kept: list[str] = []
    columns: list[np.ndarray] = []
    for label in breaks:
        period = pd.Period(label, "Q")
        if period <= obs_index[0] or period > obs_index[-1]:
            print(f"  note: break {label} is outside the sample and has been dropped")
            continue
        kept.append(label)
        columns.append((obs_index >= period).astype(float))
    design = np.column_stack(columns) if columns else np.zeros((len(obs_index), 0))
    return design, kept


def _wedge_nu(model: pm.Model, config: ModelConfig) -> tuple[Any, str]:
    """Return the Student-t degrees of freedom for the wedge, and a label.

    Free `nu` is the model's one funnel: the innovations are drawn at a
    degrees-of-freedom that is itself sampled, so the two shape each other's
    geometry. Fixing it removes that. See `ModelConfig.nu_walk` for why the
    value is a judgement about the wedge rather than a sampler setting.

    Must be called inside the model context, since it may create a variable.
    """
    if config.nu_walk is None:
        return pm.Gamma("nu_walk", alpha=2.0, beta=2.0 / config.nu_prior_mean), "nu estimated"

    if config.nu_walk <= 0:
        raise ValueError(f"nu_walk must be positive, got {config.nu_walk}")

    fixed = getattr(model, "_fixed_constants", {})
    fixed["nu_walk"] = config.nu_walk
    model._fixed_constants = fixed  # noqa: SLF001 — our own metadata, as elsewhere in the package
    return config.nu_walk, f"nu={config.nu_walk:g} imposed"


def _wedge_innovations(
    nu: pt.TensorVariable | float,
    n: int,
    *,
    noncentred: bool,
) -> tuple[pt.TensorVariable, str]:
    """Return the wedge's Student-t innovations, centred or as a scale mixture.

    A Student-t(nu) is *exactly* `Normal(0, 1) · sqrt(lam)` with
    `lam ~ InverseGamma(nu/2, nu/2)`, so the two branches target the same
    posterior. Only the geometry differs, which is the whole point.

    The centred form is what the model had, and it is where its divergences
    come from. Three tests point at it rather than at the alternatives: raising
    `target_accept` to 0.97 changed nothing (11 divergences against 12, and
    slightly worse ESS); fixing `nu` changed nothing (7 at 2.36, 15 at 2.0);
    and the divergence count scales with how heavy the tails are (44 at
    nu = 1.5). That is the innovations themselves, whose per-quarter scale
    varies over orders of magnitude under a heavy tail, so no single step size
    serves the whole distribution. The mixture makes that scale an explicit
    `lam` the sampler can adapt to.

    Kept switchable because a pure reparameterisation is only useful as a check
    if the thing it is checked against can still be run.
    """
    if not noncentred:
        return pm.StudentT("eps_wedge", nu=nu, mu=0.0, sigma=1.0, shape=n), "centred"

    # Scale before the standardised draw, matching the order the package uses
    # elsewhere: the variance scale is declared first, then what rides on it.
    lam = pm.InverseGamma("lam_wedge", alpha=nu / 2.0, beta=nu / 2.0, shape=n)
    z = pm.Normal("z_wedge", mu=0.0, sigma=1.0, shape=n)
    eps = pm.Deterministic("eps_wedge", z * pt.sqrt(lam))
    return eps, "scale mixture"


def build_model(
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    config: ModelConfig | None = None,
    verbose: bool = True,
) -> pm.Model:
    """Build the rstar PyMC model: one state, one or two observation equations.

    The term premium is a stationary AR(1) with a free mean, and the natural
    rate is a random walk whose innovation sd is imposed. That pairing is the
    identification: a permanent component and a transitory one have different
    spectra, so the split is identified in principle by persistence. In
    practice it leans on `sigma_r`, which is why the world equation matters —
    it puts a second observable on the same state.
    """
    if config is None:
        config = ModelConfig()

    n = len(obs["y"])
    model = pm.Model()
    descriptions: list[str] = []

    with model:
        if not hasattr(model, "_fixed_constants"):
            model._fixed_constants = {}  # noqa: SLF001 — our own metadata, as ystar.base does
        model._fixed_constants.update(config.constants)  # noqa: SLF001

        mc = set_model_coefficients(model, {
            "mu_tp": {"mu": 0.75, "sigma": 1.0},
            "rho_tp": {"mu": 0.8, "sigma": 0.2, "lower": 0.0, "upper": 0.98},
            "sigma_tp": {"sigma": 1.0},
        })

        # --- The Australian wedge: the only state, and a step function ---
        # r* is world r* plus an Australia-specific wedge, taking "r* is
        # largely imported" as the maintained hypothesis rather than something
        # to be discovered. World r* is data, so r* inherits its movement and
        # is never smoother than it — which an earlier version, where r* was
        # its own smooth random walk, got wrong.
        base = pt.as_tensor_variable(obs["w"]) if config.use_world else pt.zeros(n)
        wedge_0 = pm.Normal("wedge_0", mu=0.0, sigma=2.0)

        if config.free_wedge:
            # A random walk with Student-t innovations: quiet most quarters,
            # with the occasional large move permitted, and nobody naming the
            # dates. Non-centred, so the imposed scale creates no funnel.
            nu, nu_desc = _wedge_nu(model, config)
            eps, eps_desc = _wedge_innovations(nu, n, noncentred=config.noncentred_wedge)
            nu_desc = f"{nu_desc}, {eps_desc}"
            wedge = wedge_0 + config.sigma_walk * pt.concatenate([pt.zeros(1), pt.cumsum(eps[1:])])
            descriptions.append(
                f"Wedge:      random walk, StudentT innovations, "
                f"sigma={config.sigma_walk:g} imposed, {nu_desc}",
            )
        else:
            design, kept = _break_design(obs_index, config.break_quarters)
            wedge = wedge_0
            if kept:
                jumps = pm.Normal("jumps", mu=0.0, sigma=config.jump_sigma, shape=len(kept))
                wedge = wedge + pt.dot(pt.as_tensor_variable(design), jumps)
            if config.wedge_drift > 0:
                eps = pm.Normal("eps_wedge", mu=0.0, sigma=1.0, shape=n)
                wedge = wedge + config.wedge_drift * pt.concatenate([pt.zeros(1), pt.cumsum(eps[1:])])
            model._fixed_constants["break_labels"] = kept  # noqa: SLF001 — names the jumps in results
            breaks_desc = ", ".join(kept) if kept else "none in sample"
            descriptions.append(f"Wedge:      steps at {breaks_desc}"
                                f"{'' if config.wedge_drift else '   (flat in between)'}")

        wedge = pm.Deterministic("wedge", wedge)
        r_star = pm.Deterministic("r_star", base + wedge)
        descriptions.append("State:      r*_t = world r*_t + wedge_t")

        # --- The term premium: defined, not fitted ---
        # tp is what the yield leaves over once r* is taken out, so the yield
        # equation is an identity and carries no residual. An earlier version
        # gave it one: with r* and tp both free to explain a single series the
        # noise had nothing to do, sigma_y collapsed toward zero (posterior mean
        # 0.035, ess 11, r_hat 1.30) and mu_tp rode a ridge against the level of
        # r*. Removing the redundant parameter removes the ridge.
        tp = pm.Deterministic("tp", pt.as_tensor_variable(obs["y"]) - r_star)
        descriptions.append("Yield:      tp_t = y_t - r*_t   (identity, no residual)")

        # The stationarity of that premium is what identifies the split, so it
        # enters as a prior on tp rather than as a likelihood: a stationary
        # initial draw, then an AR(1) transition. rho is bounded below 1 —
        # a unit root would make the premium a second random walk and the
        # decomposition meaningless.
        stationary_sd = mc["sigma_tp"] / pt.sqrt(1.0 - mc["rho_tp"] ** 2)
        pm.Potential(
            "tp_prior",
            pm.logp(pm.Normal.dist(mu=mc["mu_tp"], sigma=stationary_sd), tp[0])
            + pm.logp(
                pm.Normal.dist(
                    mu=mc["mu_tp"] + mc["rho_tp"] * (tp[:-1] - mc["mu_tp"]),
                    sigma=mc["sigma_tp"],
                ),
                tp[1:],
            ).sum(),
        )
        descriptions.append("Premium:    tp ~ AR(1) about mu   (stationary: the identifying prior)")

        if not config.use_world:
            descriptions.append("World:      dropped (--no-world): the wedge absorbs the world level too")

    if verbose:
        print("\nModel specification:")
        for line in descriptions:
            print(f"  {line}")
        print()

    return model


def save_results(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    constants: dict[str, Any],
    *,
    chart_obs: pd.DataFrame | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "rstar",
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
            {"obs": obs, "obs_index": obs_index, "constants": constants, "chart_obs": chart_obs},
            f,
        )
    print(f"Saved observations to: {obs_path}")

    return output_dir


def run_estimate(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    prefix: str = "rstar",
    verbose: bool = False,
    seed: int | None = None,
) -> tuple[az.InferenceData, dict[str, np.ndarray], pd.PeriodIndex]:
    """Build observations, sample the posterior, save the results."""
    if config is None:
        config = ModelConfig()
    if sampler_config is None:
        sampler_config = SamplerConfig()
    if seed is not None:
        sampler_config.random_seed = seed

    print(f"Sample:       {config.start} -> {config.end or 'latest'}")
    print(f"World anchor: {config.world_source if config.use_world else 'none (--no-world)'}")
    if config.free_wedge:
        print(f"Wedge:        free random walk, StudentT innovations, sigma_walk={config.sigma_walk:g}")
    else:
        print(f"Breaks:       {', '.join(config.break_quarters)}")
        print(f"Wedge drift:  {config.wedge_drift:g} (0 = pure step function)")
    source = (
        f"joint ({config.joint_prefix})" if config.input_source == "joint"
        else f"separate ({config.ystar_prefix} + {config.ustar_prefix})"
    )
    print(f"Rule inputs:  {source}")
    print(f"Policy rule:  d_i = {config.rule_pi:g} x (pi - {config.anchor:g}) + {config.rule_gap:g} x "
          f"{'(-u gap)' if config.taylor_use_ugap else 'output gap'}   (first difference)")
    print(f"Sampler seed: {sampler_config.random_seed}")

    print("\nBuilding observations...")
    obs, obs_index, chart_obs = build_observations(
        start=config.start,
        end=config.end,
        world_source=config.world_source,
        input_source=config.input_source,
        joint_prefix=config.joint_prefix,
        ystar_prefix=config.ystar_prefix,
        ustar_prefix=config.ustar_prefix,
        verbose=verbose,
    )

    print("Building model...")
    model = build_model(obs, obs_index, config=config)

    print("Sampling...")
    trace = sample_model(model, sampler_config)
    print()

    save_results(
        trace, obs, obs_index,
        constants=get_fixed_constants(model),
        chart_obs=chart_obs,
        output_dir=config.output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index
