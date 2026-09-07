"""Build, sample, and persist the joint y*/u* model.

Equation order matters and follows `ystar`'s convention: the variance scale
first, then the state equations, then the observation equations.
"""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

from src.models.ystar.base import (
    SamplerConfig,
    get_fixed_constants,
    sample_model,
    set_model_coefficients,
)
from src.models.ystar.equations.potential import potential_output_equation
from src.models.ystar.equations.scale import scale_equation
from src.models.ystar_ustar.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.ystar_ustar.observations import build_observations
from src.utilities.rate_conversion import quarterly


def _keep_mask(config: ModelConfig, obs_index: pd.PeriodIndex) -> np.ndarray | None:
    """Resolve `config.exclude_window` to a boolean keep-mask over the sample.

    Returns None when no window is excluded. Raises rather than quietly
    excluding nothing if the window misses the sample.
    """
    if config.exclude_window is None:
        return None

    lo, hi = config.exclude_window
    dropped = (obs_index >= pd.Period(lo, freq="Q")) & (obs_index <= pd.Period(hi, freq="Q"))
    if not dropped.any():
        raise ValueError(
            f"exclude_window {lo} to {hi} covers no quarter of the estimation sample "
            f"{obs_index.min()} to {obs_index.max()}",
        )
    if dropped.all():
        raise ValueError(f"exclude_window {lo} to {hi} would drop the whole sample")

    return ~np.asarray(dropped)


def _observe(
    name: str,
    mu: Any,  # noqa: ANN401 — a pytensor expression or an ndarray
    sigma: Any,  # noqa: ANN401
    observed: np.ndarray,
    keep: np.ndarray | None,
) -> None:
    """Add a Normal likelihood, optionally over a subset of quarters.

    Dropping rows from both sides is what "these quarters carry no likelihood"
    means operationally: the states still run through the window under their
    priors, and nothing is spliced, but the window makes no claim about the fit.
    """
    if keep is None:
        pm.Normal(name, mu=mu, sigma=sigma, observed=observed)
        return

    if keep.shape != observed.shape:
        raise ValueError(f"keep has shape {keep.shape}, expected {observed.shape}")

    rows = np.flatnonzero(keep)
    pm.Normal(name, mu=mu[rows], sigma=sigma, observed=observed[rows])


def _ustar_state(obs: dict[str, np.ndarray], model: pm.Model, config: ModelConfig) -> Any:  # noqa: ANN401
    """u* as a driftless Gaussian random walk with an imposed innovation sd.

    Taken unchanged from `ustar`, including the wide initial prior centred on
    the sample's own mean unemployment rate.
    """
    with model:
        if not hasattr(model, "_fixed_constants"):
            model._fixed_constants = {}  # noqa: SLF001 — our own metadata, as base.py does
        model._fixed_constants.update(config.constants)  # noqa: SLF001
        # The prior settings that vary by run, recorded so `analyse.py` can draw
        # each posterior against the prior it was actually sampled under rather
        # than against a hard-coded guess.
        model._fixed_constants.update({  # noqa: SLF001
            "sigma_v_prior": config.sigma_v_prior,
            "beta_okun_prior_sd": config.beta_okun_prior_sd,
            "two_sided_c": config.two_sided_c,
            "two_sided_beta": config.two_sided_beta,
        })

        drift: Any = 0.0
        if config.ustar_drift:
            if "pi_exp" not in obs:
                raise ValueError(
                    "ustar_drift needs the expectations series, which is only loaded "
                    "with the Phillips curve; drop --no-phillips or --no-ustar-drift",
                )
            # Lagged, so this quarter's drift is set by what expectations were
            # before this quarter's unemployment was observed.
            excess = np.maximum(0.0, np.asarray(obs["pi_exp"], dtype=float) - config.anchor)
            mc = set_model_coefficients(
                model, {"lambda_ustar": {"mu": 0.0, "sigma": 0.1}},
            )
            drift = -mc["lambda_ustar"] * excess[:-1]

        n = len(obs["u"])
        if config.ustar_converge:
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
                sequences=[z[1:] * config.sigma_ustar],
                outputs_info=[init],
                non_sequences=[mc["phi_ustar"], mc["ustar_eq"]],
            )
            return pm.Deterministic("ustar", pt.concatenate([[init], path]))

        return pm.GaussianRandomWalk(
            "ustar",
            mu=drift,
            sigma=config.sigma_ustar,
            init_dist=pm.Normal.dist(mu=float(np.mean(obs["u"])), sigma=3.0),
            shape=n,
        )


def _cycle_gap(obs: dict[str, np.ndarray], model: pm.Model, config: ModelConfig) -> tuple[Any, str]:
    """Return the gap as a free AR(1) latent, for the `cycle` spec.

    Written in terms of the gap's **stationary** sd, not its innovation sd:

        gap_t = rho·gap_{t-1} + sqrt(1 - rho^2)·sigma_c·z_t,   z ~ N(0, 1)

    so the unconditional sd of the gap is `sigma_c` whatever `rho` does. The
    first attempt used `sigma_c` as the innovation sd, which made the implied
    amplitude sigma_c/sqrt(1 - rho^2). With rho free to 0.99 that is an
    amplitude prior of up to 4.3, the gap became a second trend at sd 2.15,
    and the run collapsed: ESS of 6 to 12, r_hat to 1.71, chains in different
    modes, and both `beta_okun` and `kappa_gap` straddling zero. This form
    makes persistence and amplitude orthogonal, so imposing 0.60 means "a cycle
    of about 0.6 per cent of GDP" rather than something that depends on a
    parameter being estimated alongside it.

    `rho` is truncated below 1: a non-stationary gap would be a second trend.
    """
    n = len(obs["log_gdp"])
    with model:
        rho = pm.TruncatedNormal("rho_gap", mu=0.8, sigma=0.2, lower=0.0, upper=0.99)
        # Non-centred, for the same reason v is: the data are only moderately
        # informative about any single quarter's innovation.
        z = pm.Normal("z_gap", mu=0.0, sigma=1.0, shape=n)
        innovations = z * config.sigma_c * pt.sqrt(1.0 - rho**2)

        def step(eps: Any, prev: Any, rho_: Any) -> Any:  # noqa: ANN401
            return rho_ * prev + eps

        # The first quarter is drawn from the stationary distribution directly,
        # which is sigma_c by construction here.
        initial = z[0] * config.sigma_c
        path, _ = pytensor.scan(
            fn=step,
            sequences=[innovations[1:]],
            outputs_info=[initial],
            non_sequences=[rho],
        )
        gap = pm.Deterministic("output_gap", pt.concatenate([[initial], path]))
    return gap, (
        f"gap_t = rho x gap_{{t-1}} + sqrt(1-rho^2) x e   "
        f"(stationary sd {config.sigma_c:g} imposed, rho free)"
    )


def _gap_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    config: ModelConfig,
) -> tuple[Any, str]:
    """Return the output gap, `c·(pi_ann - anchor) + v`, and a description.

    `v` is non-centred, `v = z·sigma_v` with `z ~ N(0, 1)`. This is the case
    non-centring was designed for and the opposite of `rstar`'s: the data are
    weakly informative about `v` (that is the whole problem it poses), so
    sampling `z` on a fixed scale avoids the funnel between `sigma_v` and 134
    latents that the centred form would build.
    """
    if config.gap_spec == "cycle":
        return _cycle_gap(obs, model, config)

    n = len(obs["log_gdp"])
    deviation = np.asarray(obs["pi_gap"], dtype=float) - config.anchor

    with model:
        settings: dict[str, dict[str, float]] = {
            "c": {"mu": 0.0, "sigma": 2.0} if config.two_sided_c else {"sigma": 2.0},
        }
        mc = set_model_coefficients(model, settings)
        defined = mc["c"] * deviation

        if not config.free_gap_component:
            gap = pm.Deterministic("output_gap", defined)
            return gap, "gap_t = c x (pi_ann - anchor)   [v off: this is ystar's identity]"

        if config.sigma_v is not None:
            sigma_v: Any = config.sigma_v
            scale_desc = f"sigma_v={config.sigma_v:g} imposed"
        else:
            sigma_v = pm.HalfNormal("sigma_v", sigma=config.sigma_v_prior)
            scale_desc = f"sigma_v ~ HalfNormal({config.sigma_v_prior:g})"

        z = pm.Normal("z_v", mu=0.0, sigma=1.0, shape=n)
        free = pm.Deterministic("v", z * sigma_v)
        gap = pm.Deterministic("output_gap", defined + free)
        pm.Deterministic("defined_gap", defined)

    return gap, f"gap_t = c x (pi_ann - anchor) + v_t   ({scale_desc})"


def _gdp_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    gap: Any,  # noqa: ANN401
    keep: np.ndarray | None,
) -> str:
    """Fit log_gdp = y* + gap + e_c."""
    with model:
        mc = set_model_coefficients(model, {"sigma_e": {"sigma": 1.0}})
        _observe(
            "observed_gdp",
            latents["potential_output"] + gap,
            mc["sigma_e"],
            np.asarray(obs["log_gdp"], dtype=float),
            keep,
        )
    return "log_gdp = y* + gap + e_c"


def _okun_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    ustar: Any,  # noqa: ANN401
    gap: Any,  # noqa: ANN401
    *,
    config: ModelConfig,
    keep: np.ndarray | None,
) -> str:
    """Fit u = u* - beta x gap + e_o, the equation that identifies sigma_v.

    This is the only place the gap appears besides the GDP equation, and the
    covariance it creates between the two residuals is what separates `v` from
    `e_c`. Without it `sigma_v` returns its prior.
    """
    with model:
        prior = {"mu": 0.5, "sigma": config.beta_okun_prior_sd}
        if not config.two_sided_beta:
            prior["lower"] = 0.0
        constant = {} if config.sigma_okun is None else {"sigma_okun": config.sigma_okun}
        mc = set_model_coefficients(
            model,
            {"beta_okun": prior, "sigma_okun": {"sigma": 1.0}},
            constant=constant,
        )
        _observe(
            "observed_u",
            ustar - mc["beta_okun"] * gap,
            mc["sigma_okun"],
            np.asarray(obs["u"], dtype=float),
            keep,
        )
    return "u = u* - beta x gap + e_o"


def _phillips_on_gap(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    gap: Any,  # noqa: ANN401
    anchor: float,
    keep: np.ndarray | None,
) -> str:
    """Fit the Phillips curve on the output gap, for the `cycle` spec.

    Identical to `_phillips_equation` except that the demand term is `kappa·gap`
    rather than `gamma·(u - u*)/u`. That single change is what removes the
    circularity: the regressor is now a latent state that inflation observes,
    rather than a transform of inflation itself.

    `u*` remains a NAIRU, transitively rather than directly. It is the
    unemployment rate at which the gap is zero (from the Okun equation), and the
    gap is zero where inflation sits at target once expectations and supply are
    accounted for (from this equation). That is the same chain `ystar` and
    `ustar` relied on, with the arrow drawn the honest way round.
    """
    with model:
        mc = set_model_coefficients(
            model,
            {
                "kappa_gap": {"mu": 0.5, "sigma": 0.5},
                "beta_pi": {"mu": 0.5, "sigma": 0.3},
                "rho_pi": {"mu": 0.0, "sigma": 0.1},
                "xi_gscpi": {"mu": 0.0, "sigma": 0.1},
                "epsilon_pi": {"sigma": 0.25},
            },
        )
        anchor_quarterly = quarterly(anchor)
        excess_quarterly = quarterly(obs["pi_exp"]) - anchor_quarterly
        mu = (
            anchor_quarterly
            + mc["kappa_gap"] * gap
            + mc["beta_pi"] * excess_quarterly
            + mc["rho_pi"] * obs["d4pm"]
            + mc["xi_gscpi"] * obs["gscpi"] ** 2 * np.sign(obs["gscpi"])
        )
        _observe("observed_pi", mu, mc["epsilon_pi"], np.asarray(obs["pi_qtr"], dtype=float), keep)
    return (
        f"pi_q = q({anchor:g}) + beta x [q(pi_exp) - q({anchor:g})]"
        " + kappa x gap + rho x d4pm + xi x GSCPI^2 + e_p"
    )


def _phillips_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    ustar: Any,  # noqa: ANN401
    anchor: float,
    keep: np.ndarray | None,
) -> str:
    """Fit the price Phillips curve on the quarterly rate, anchored on the target.

    Taken unchanged from `ustar`, including the `(u - u*)/u` gap form so that
    `gamma_pi` stays directly comparable with `ustar`'s and `nairu`'s.

    The circularity that `ustar` has is attenuated here rather than removed.
    The regressor is still `u - u* = -beta·gap + e_o`, and `gap` still contains
    `c·(pi_ann - anchor)`, so part of it remains a rescaled copy of inflation.
    What changes is that `gap` also contains `v`, which is not inflation, so the
    manufactured share falls in proportion to how much of the gap is `v`. That
    share is `sigma_v`, which the model estimates, so for the first time the
    contamination is a number the posterior reports rather than an unknown.
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
        excess_quarterly = quarterly(obs["pi_exp"]) - anchor_quarterly

        mu = (
            anchor_quarterly
            + mc["gamma_pi"] * ugap
            + mc["beta_pi"] * excess_quarterly
            + mc["rho_pi"] * obs["d4pm"]
            + mc["xi_gscpi"] * obs["gscpi"] ** 2 * np.sign(obs["gscpi"])
        )
        _observe(
            "observed_pi",
            mu,
            mc["epsilon_pi"],
            np.asarray(obs["pi_qtr"], dtype=float),
            keep,
        )
    return (
        f"pi_q = q({anchor:g}) + beta x [q(pi_exp) - q({anchor:g})]"
        " + gamma x u_gap + rho x d4pm + xi x GSCPI^2 + e_p"
    )


def build_model(
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    config: ModelConfig | None = None,
    verbose: bool = True,
) -> pm.Model:
    """Build the joint y*/u* PyMC model."""
    if config is None:
        config = ModelConfig()

    keep = _keep_mask(config, obs_index)
    # The window is applied to every equation by default. Under "gdp" scope the
    # Okun and Phillips equations keep all their quarters, which reproduces what
    # `ystar` and `ustar` do when run separately.
    keep_gdp = keep
    keep_other = keep if config.exclude_scope == "all" else None

    model = pm.Model()
    latents: dict[str, Any] = {}
    descriptions: list[str] = []

    desc = scale_equation(obs, model, latents, constant=config.scale_constants)
    descriptions.append(f"Scale:        {desc}")
    if config.exclude_window is not None:
        get_fixed_constants(model)["exclude_window"] = config.exclude_window
        get_fixed_constants(model)["exclude_scope"] = config.exclude_scope

    desc = potential_output_equation(obs, model, latents)
    descriptions.append(f"Potential:    {desc}")

    ustar = _ustar_state(obs, model, config)
    nairu_state = (
        "u*_t = u*_{t-1} + phi x (u*_eq - u*_{t-1}) + e_u   (sigma imposed)"
        if config.ustar_converge
        else "u*_t = u*_{t-1} + e_u   (sigma imposed)"
    )
    descriptions.append(f"NAIRU:        {nairu_state}")

    gap, gap_desc = _gap_equation(obs, model, config)
    descriptions.append(f"Gap:          {gap_desc}")

    descriptions.append(f"Output:       {_gdp_equation(obs, model, latents, gap, keep_gdp)}")

    if config.include_okun:
        descriptions.append(
            f"Okun:         {_okun_equation(obs, model, ustar, gap, config=config, keep=keep_other)}",
        )
    if config.include_phillips:
        phillips = (
            _phillips_on_gap(obs, model, gap, config.anchor, keep_other)
            if config.gap_spec == "cycle"
            else _phillips_equation(obs, model, ustar, config.anchor, keep_other)
        )
        descriptions.append(f"Phillips:     {phillips}")
        if config.gap_spec == "cycle":
            # The unemployment gap is not a regressor here, but it is still the
            # model's headline output, so it is recorded for the charts.
            with model:
                pm.Deterministic("ugap", (pt.as_tensor_variable(obs["u"]) - ustar) / obs["u"])
    if not config.include_phillips:
        # u* has no nominal content without it, so the headline is recorded here
        # under a name that does not claim to be a NAIRU.
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
    prefix: str = "ystar_ustar",
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
    prefix: str = "ystar_ustar",
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

    scale = config.scale_constants
    print(f"Sample:       {config.start} -> {config.end or 'latest'}")
    print(
        f"Imposed:      sigma_ystar={scale['ratio_ystar'] * scale['sigma_c']:.4g}, "
        f"sigma_g={scale['ratio_g'] * scale['sigma_c']:.4g}, "
        f"sigma_ustar={config.sigma_ustar:g}",
    )
    equations = ["GDP"]
    if config.include_okun:
        equations.append("Okun")
    if config.include_phillips:
        equations.append("Phillips")
    print(f"Equations:    {' + '.join(equations)}")
    if not config.free_gap_component:
        print("Gap:          c x (pi - anchor), no free component (ystar's identity)")
    elif config.sigma_v is not None:
        print(f"Gap:          c x (pi - anchor) + v,  sigma_v={config.sigma_v:g} imposed")
    else:
        print(f"Gap:          c x (pi - anchor) + v,  sigma_v ~ HalfNormal({config.sigma_v_prior:g})")
    print(f"Sampler seed: {sampler_config.random_seed}")

    print("\nBuilding observations...")
    obs, obs_index, chart_obs = build_observations(
        start=config.start,
        end=config.end,
        gap_pi_basis=config.gap_pi_basis,
        include_phillips=config.include_phillips,
        verbose=verbose,
    )

    if config.exclude_window is not None:
        lo, hi = config.exclude_window
        dropped = (obs_index >= pd.Period(lo, freq="Q")) & (obs_index <= pd.Period(hi, freq="Q"))
        scope = "all equations" if config.exclude_scope == "all" else "the GDP equation only"
        print(f"Excluded:     {lo} to {hi}  ({int(dropped.sum())} quarters) from {scope}")

    print("Building model...")
    model = build_model(obs, obs_index, config=config, verbose=True)

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
