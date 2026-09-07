"""Build, sample, and persist the ustar model."""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

from src.models.ustar.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.ustar.observations import build_observations
from src.models.ystar.base import (
    SamplerConfig,
    get_fixed_constants,
    sample_model,
    set_model_coefficients,
)
from src.utilities.rate_conversion import quarterly


def _output_gap(obs: dict[str, np.ndarray], model: pm.Model, config: ModelConfig) -> Any:  # noqa: ANN401
    """Return the output gap the Okun equation sees. Not data, not estimated here.

    The two gap sources need different treatment, because their uncertainty has
    a different shape.

    **defined** — the gap is `c·d_t` with `d_t` observed and `c` a single
    scalar, so its uncertainty is one number, perfectly correlated across
    quarters: `sd/|mean|` is 0.241146 in every quarter of the sample, to
    machine precision. A per-quarter prior would misdescribe that as 134
    independent errors, and would be improper anyway in the three quarters
    where inflation sits exactly on the anchor and the sd is exactly zero.
    Nothing is needed instead: `u = u* - beta·c·d_t` means `beta` and `c` enter
    only as a product, so `beta` absorbs `c`'s uncertainty exactly. The cost is
    that `beta` is then a scaling onto this gap, not an Okun coefficient.

    **actual** — `log_gdp - y*` carries potential's own uncertainty, which is
    genuinely per-quarter (sd 0.23 to 0.45, and `sd/|mean|` ranging from 0.03
    to 33). There the measurement-error prior is the right object.
    """
    if not config.use_output_gap:
        # The diagnostic in `ModelConfig.use_output_gap`: the Okun equation
        # keeps its shape and becomes u = u* + e_o, so what drops out is the
        # gap's information rather than the equation.
        return np.zeros_like(obs["gap_mean"])

    if not config.gap_measurement_error or config.gap_source == "defined":
        return obs["gap_mean"]

    with model:
        return pm.Normal(
            "ygap",
            mu=obs["gap_mean"],
            sigma=obs["gap_sd"],
            shape=len(obs["gap_mean"]),
        )


def _ustar_state(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    config: ModelConfig,
    obs_index: pd.PeriodIndex | None = None,
) -> Any:  # noqa: ANN401
    """u* as a driftless Gaussian random walk with an imposed innovation sd.

    The initial level is given a wide prior centred on the sample's own mean
    unemployment rate: the data should place u*, and starting it anywhere else
    would just be a slow transient at the front of the sample.
    """
    n = len(obs["u"])
    with model:
        # Recorded directly: set_model_coefficients only writes constants for
        # names present in its settings dict, and sigma_ustar has no prior to
        # be a setting for. Without this it never reaches the run log or the
        # diagnostics, where it is the number the answer hinges on.
        if not hasattr(model, "_fixed_constants"):
            model._fixed_constants = {}  # noqa: SLF001 — our own metadata, as base.py does
        model._fixed_constants.update(config.constants)  # noqa: SLF001

        if config.free_sigma_ustar:
            mu, sd, lower, upper = config.sigma_ustar_prior
            sigma = pm.TruncatedNormal("sigma_ustar", mu=mu, sigma=sd, lower=lower, upper=upper)
        else:
            sigma = config.sigma_ustar

        drift: Any = 0.0
        if config.ustar_drift:
            if "pi_exp" not in obs:
                raise ValueError(
                    "ustar_drift needs the expectations series, which is only loaded "
                    "with the Phillips curve: drop --no-phillips",
                )
            # Lagged, so this quarter's drift is set by expectations formed
            # before this quarter's unemployment was observed.
            excess = np.maximum(0.0, np.asarray(obs["pi_exp"], dtype=float) - config.anchor)
            if obs_index is None:
                raise ValueError("ustar_drift needs obs_index to apply its end date")
            excess = np.where(obs_index < pd.Period(config.ustar_drift_end, freq="Q"), excess, 0.0)
            mc = set_model_coefficients(
                model, {"lambda_ustar": {"mu": 0.0, "sigma": config.lambda_prior_sd}},
            )
            drift = -mc["lambda_ustar"] * excess[:-1]

        if config.ustar_converge:
            # u*_t = u*_{t-1} + phi·(u*_eq - u*_{t-1}) + e, written as a scan so
            # the mean reversion is on the state's own past rather than on an
            # exogenous series. Non-centred innovations, as everywhere else here.
            mc = set_model_coefficients(
                model,
                {
                    "phi_ustar": {"mu": 0.05, "sigma": 0.05, "lower": 0.0, "upper": 1.0},
                    "ustar_eq": {"mu": 5.0, "sigma": 2.0},
                },
            )
            z = pm.Normal("z_ustar", mu=0.0, sigma=1.0, shape=n)
            init = pm.Normal("ustar_init", mu=float(obs["u"][0]), sigma=3.0)

            def step(eps: Any, prev: Any, phi: Any, eq: Any) -> Any:  # noqa: ANN401
                return prev + phi * (eq - prev) + eps

            path, _ = pytensor.scan(
                fn=step,
                sequences=[z[1:] * sigma],
                outputs_info=[init],
                non_sequences=[mc["phi_ustar"], mc["ustar_eq"]],
            )
            return pm.Deterministic("ustar", pt.concatenate([[init], path]))

        return pm.GaussianRandomWalk(
            "ustar",
            mu=drift,
            sigma=sigma,
            init_dist=pm.Normal.dist(mu=float(np.mean(obs["u"])), sigma=3.0),
            shape=n,
        )


def _okun_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    ustar: Any,  # noqa: ANN401
    ygap: Any,  # noqa: ANN401
    config: ModelConfig,
) -> str:
    """Fit u = u* - beta x ygap + e, the equation that sets u*'s level.

    `beta` is two-sided by default: a positive-truncated prior would assert
    Okun's law rather than let the posterior report P(beta > 0), which is the
    quantity worth reading.
    """
    with model:
        prior = {"mu": 0.5, "sigma": 0.5}
        if not config.two_sided_beta:
            prior["lower"] = 0.0
        mc = set_model_coefficients(
            model,
            {"beta_okun": prior, "sigma_okun": {"sigma": 1.0}},
        )
        pm.Normal(
            "observed_u",
            mu=ustar - mc["beta_okun"] * ygap,
            sigma=mc["sigma_okun"],
            observed=obs["u"],
        )
    return "u = u* - beta x ygap + e"


def _phillips_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    ustar: Any,  # noqa: ANN401
    anchor: float,
) -> str:
    """Fit the price Phillips curve, anchored on the target.

        pi = quarterly(anchor) + beta x [quarterly(pi_exp) - quarterly(anchor)]
             + gamma x u_gap + rho x d4pm + xi x GSCPI^2 x sign(GSCPI) + e

    Two distinct objects on the nominal side, which is what makes the pair
    coherent: a constant target, and one expectations series entering only as
    its deviation from that target. `beta` is then the pass-through of
    de-anchoring — 0 means the target holds, 1 means expectations are what
    bind. Pairing the same term with an *expectations* baseline instead would
    make it one estimate of expectations minus another.

    The gap is `(u - u*)/u`, the scale-invariant percentage form the NAIRU
    model uses, so `gamma_pi` here is on the same scale as `gamma_pi` there and
    the two are directly comparable. Inflation, expectations and the target are
    all on the quarterly basis for the same reason.

    The GSCPI term is squared with the sign restored, so it is convex in the
    size of the disruption without turning an easing of supply chains into an
    inflationary impulse.
    """
    with model:
        mc = set_model_coefficients(
            model,
            {
                "gamma_pi": {"mu": -1.5, "sigma": 1.0},
                "beta_pi": {"mu": 0.5, "sigma": 0.3},
                "rho_pi": {"mu": 0.0, "sigma": 0.1},
                "xi_gscpi": {"mu": 0.0, "sigma": 0.1},
                "epsilon_pi": {"sigma": 0.25},
            },
        )
        u = pt.as_tensor_variable(obs["u"])
        ugap = pm.Deterministic("ugap", (u - ustar) / u)

        anchor_quarterly = quarterly(anchor)
        # A quarterly-space difference, so beta_pi = 1 exactly reproduces
        # quarterly(expectations) and beta_pi = 0 leaves the bare target.
        excess_quarterly = quarterly(obs["pi_exp"]) - anchor_quarterly

        mu = (
            anchor_quarterly
            + mc["gamma_pi"] * ugap
            + mc["beta_pi"] * excess_quarterly
            + mc["rho_pi"] * obs["d4pm"]
            + mc["xi_gscpi"] * obs["gscpi"] ** 2 * np.sign(obs["gscpi"])
        )
        pm.Normal(
            "observed_pi",
            mu=mu,
            sigma=mc["epsilon_pi"],
            observed=obs["pi"],
        )
    return (
        f"pi = q({anchor:g}) + beta x [q(pi_exp) - q({anchor:g})]"
        " + gamma x u_gap + rho x d4pm + xi x GSCPI^2 + e"
    )


def build_model(
    obs: dict[str, np.ndarray],
    config: ModelConfig | None = None,
    verbose: bool = True,
    obs_index: pd.PeriodIndex | None = None,
) -> pm.Model:
    """Build the ustar PyMC model: one state, one or two observation equations."""
    if config is None:
        config = ModelConfig()

    model = pm.Model()
    descriptions: list[str] = []

    ygap = _output_gap(obs, model, config)
    ustar = _ustar_state(obs, model, config, obs_index)
    state = "u*_t = u*_{t-1} + e   (sigma imposed)"
    if config.ustar_converge:
        state = "u*_t = u*_{t-1} + phi x (u*_eq - u*_{t-1}) + e   (sigma imposed)"
    elif config.ustar_drift:
        state = (
            f"u*_t = u*_{{t-1}} - lambda x max(0, pi_exp - {config.anchor:g}) + e   "
            f"(drift off from {config.ustar_drift_end}, sigma imposed)"
        )
    descriptions.append(f"State:        {state}")

    descriptions.append(f"Okun:         {_okun_equation(obs, model, ustar, ygap, config)}")

    if config.include_phillips:
        descriptions.append(f"Phillips:     {_phillips_equation(obs, model, ustar, config.anchor)}")
    else:
        # The unemployment gap is the model's headline output either way, so it
        # is recorded here when the Phillips curve is not there to record it.
        with model:
            pm.Deterministic("ugap", pt.as_tensor_variable(obs["u"]) - ustar)

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
    prefix: str = "ustar",
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
            {
                "obs": obs,
                "obs_index": obs_index,
                "constants": constants,
                "chart_obs": chart_obs,
            },
            f,
        )
    print(f"Saved observations to: {obs_path}")

    return output_dir


def run_estimate(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    prefix: str = "ustar",
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
    print(f"Output gap:   {config.gap_source} (from {config.gap_prefix}), "
          f"measurement error {'on' if config.gap_measurement_error else 'off'}")
    print(f"Equations:    Okun{' + Phillips' if config.include_phillips else ' only'}")
    if config.free_sigma_ustar:
        mu, sd, lower, upper = config.sigma_ustar_prior
        bound = f"lower={lower:g}, upper={upper:g}" if upper is not None else f"lower={lower:g}"
        print(f"Drift:        sigma_ustar estimated, TruncatedNormal({mu:g}, {sd:g}, {bound})")
    else:
        print(f"Imposed:      sigma_ustar={config.sigma_ustar:g}")
    print(f"Sampler seed: {sampler_config.random_seed}")

    print("\nBuilding observations...")
    obs, obs_index, chart_obs = build_observations(
        start=config.start,
        end=config.end,
        gap_source=config.gap_source,
        gap_prefix=config.gap_prefix,
        include_phillips=config.include_phillips,
        verbose=verbose,
    )

    print("Building model...")
    model = build_model(obs, config=config, obs_index=obs_index)

    print("Sampling...")
    trace = sample_model(model, sampler_config)
    print()

    constants = get_fixed_constants(model)
    save_results(
        trace, obs, obs_index,
        constants=constants,
        chart_obs=chart_obs,
        output_dir=config.output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index
