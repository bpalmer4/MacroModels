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

from src.models.common.model_constants import attach, get_dictionary, record_constant
from src.models.common.spline import basis
from src.models.ystar.base import (
    SamplerConfig,
    sample_model,
    set_model_coefficients,
)
from src.models.ystar.equations.potential import potential_output_equation
from src.models.ystar.equations.scale import scale_equation
from src.models.ystar_ustar.config import (
    ANCHOR_GLIDE_START,
    ANCHOR_PHASE_END,
    ANCHOR_STEP_START,
    DEFAULT_OUTPUT_DIR,
    ModelConfig,
)
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
    mu: pt.TensorVariable,
    sigma: float | pt.TensorVariable,
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


def _ustar_spline(model: pm.Model, config: ModelConfig, obs_index: pd.PeriodIndex) -> pt.TensorVariable:
    """u* as a natural cubic spline with knots at `config.spline_knots`.

        u*_t = sum_j c_j B_j(t)

    Deterministic given the coefficients, so `sigma_ustar` disappears rather
    than being chosen, and the path can change slope. The decay structure it
    replaces draws a monotone approach to one equilibrium, so it reports a
    decline that never quite stops and then flattens onto the equilibrium it
    was heading for; both are properties of the functional form rather than
    readings of the data.

    The basis is a partition of unity, so the coefficients are in
    unemployment-rate units and a coefficient is roughly the level u* passes
    through near its knot.
    """
    design = basis(obs_index, tuple(config.spline_knots), natural=True)
    mu0, sd0, lo, hi = config.spline_coef_prior
    with model:
        pm.Data("sigma_ustar", np.nan)  # no innovation variance under this law
        coef = pm.TruncatedNormal("coef", mu=mu0, sigma=sd0, lower=lo, upper=hi, shape=design.shape[1])
        return pm.Deterministic("ustar", pt.dot(pt.as_tensor_variable(design), coef))


def _ustar_state(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    config: ModelConfig,
    obs_index: pd.PeriodIndex,
) -> pt.TensorVariable:
    """u* under the law `config.ustar_structure` names: a spline, or a walk.

    The walk's initial level carries a wide prior centred on the sample's own
    mean unemployment rate.
    """
    with model:
        attach(model, config.constants)
        # The prior settings that vary by run, recorded so `analyse.py` can draw
        # each posterior against the prior it was actually sampled under rather
        # than against a hard-coded guess.
        attach(model, {
            "sigma_v_prior": config.sigma_v_prior,
            "beta_okun_prior_sd": config.beta_okun_prior_sd,
            "two_sided_c": config.two_sided_c,
            "two_sided_beta": config.two_sided_beta,
        })

    # After the constants, so every law records them: the spline returns early
    # and `results` reads `anchor` from here to draw the inflation decomposition.
    if config.ustar_structure == "spline":
        return _ustar_spline(model, config, obs_index)

    with model:
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
        if config.ustar_structure == "decay":
            mc = set_model_coefficients(
                model,
                {
                    "phi_ustar": {"mu": 0.05, "sigma": 0.05, "lower": 0.0, "upper": 1.0},
                    "ustar_eq": {"mu": 5.0, "sigma": 2.0},
                },
            )
            z = pm.Normal("z_ustar", mu=0.0, sigma=1.0, shape=n)
            init = pm.Normal("ustar_init", mu=float(obs["u"][0]), sigma=3.0)

            def step(
                eps: pt.TensorVariable, prev: pt.TensorVariable, phi: pt.TensorVariable, eq: pt.TensorVariable,
            ) -> pt.TensorVariable:
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

        def step(eps: pt.TensorVariable, prev: pt.TensorVariable, rho_: pt.TensorVariable) -> pt.TensorVariable:
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


def _anchor_series(
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    config: ModelConfig,
) -> np.ndarray:
    """Return the anchor for each quarter: a constant, or expectations phasing to it.

    The weight runs 0 at the phase's first quarter to 1 at `ANCHOR_PHASE_END`,
    with expectations before the phase and `config.anchor` after it. "step"
    starts the phase in 1998Q1, so the anchor is expectations through the years
    the expectations series says were not yet anchored; "glide" starts it at the
    sample's own first quarter, which is `nairu`'s `_phase_between` shifted a
    quarter: that one runs weight 0 from its PHASE_START of 1992Q4, before this
    sample begins, so 1993Q1 already carries weight 0.042 there and 0 here.
    """
    constant = np.full(len(obs_index), float(config.anchor))
    if config.anchor_phase == "none":
        return constant

    expectations = np.asarray(obs["pi_exp"], dtype=float)
    start = ANCHOR_GLIDE_START if config.anchor_phase == "glide" else ANCHOR_STEP_START
    phase = pd.period_range(start, ANCHOR_PHASE_END, freq="Q")

    weight = np.where(obs_index < phase[0], 0.0, 1.0)
    for i, period in enumerate(phase):
        weight[obs_index == period] = i / (len(phase) - 1)
    return (1.0 - weight) * expectations + weight * constant


def _anchor_label(anchor: np.ndarray) -> str:
    """Render the anchor for an equation description: its value, or `a_t` if it varies."""
    first = float(anchor[0])
    return f"{first:g}" if bool(np.all(anchor == first)) else "a_t"


def _identity_gap(obs: dict[str, np.ndarray], model: pm.Model, latents: dict[str, Any]) -> tuple[Any, str]:
    """Return the gap as actual less potential, `y - y*`, and a description.

    Not a state. The gap is observed GDP minus one latent, so the GDP block
    carries a single trend and the free-cycle spec's failure, where a second
    state competes with `y*` for the level of GDP, has nothing to arise from.

    There is no residual: with the gap defined this way the GDP equation is an
    identity, so it contributes no likelihood and is not added.
    """
    with model:
        gap = pm.Deterministic(
            "output_gap",
            pt.as_tensor_variable(np.asarray(obs["log_gdp"], dtype=float)) - latents["potential_output"],
        )
    return gap, "gap_t = y_t - y*_t   [an identity: no GDP residual]"


def _gap_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    config: ModelConfig,
    latents: dict[str, Any],
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
    if config.gap_spec == "identity":
        return _identity_gap(obs, model, latents)

    n = len(obs["log_gdp"])
    deviation = np.asarray(obs["pi_gap"], dtype=float) - np.asarray(obs["anchor"], dtype=float)

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
    gap: pt.TensorVariable,
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


def _okun_error_correction(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    ustar: pt.TensorVariable,
    gap: pt.TensorVariable,
    *,
    config: ModelConfig,
    keep: np.ndarray | None,
) -> str:
    """Fit the error-correction form of Okun. See `_okun_equation`.

        du_t = -kappa x (u_{t-1} - u*_{t-1} + beta x gap_{t-1})
               + gamma x d(gap)_t + e_o

    THE CORRECTION IS TOWARD THE LEVEL RELATION `u = u* - beta x gap`, not
    toward `u = u*`. Correcting toward u* alone leaves the gap in the equation
    only as a difference, and under the identity gap that is the only place
    `y*` appears at all, so its LEVEL stops being identified and drifts: a
    first attempt did exactly that and returned a current gap of +26 per cent.
    Keeping the gap in the attractor is what makes this an error-correction
    model of Okun's law rather than of unemployment alone.

    `beta` is the long-run slope and `gamma` the short-run response to growth
    relative to potential. Letting them differ is the point of the form: the
    pure level equation forces them equal.

    NO INTERCEPT. The attractor already says where unemployment returns to, so
    a free constant would shift that level and compete with u* for it.

    The likelihood drops the first quarter, since every term is a difference
    or a lag, and `keep` is sliced to match rather than reused: the pandemic
    mask is indexed on the sample, not on the differenced series.
    """
    u = np.asarray(obs["u"], dtype=float)
    du = u[1:] - u[:-1]

    with model:
        prior = {"mu": 0.5, "sigma": config.beta_okun_prior_sd}
        if not config.two_sided_beta:
            prior["lower"] = 0.0
        constant = {} if config.sigma_okun is None else {"sigma_okun": config.sigma_okun}
        mc = set_model_coefficients(
            model,
            {
                "beta_okun": prior,
                # Bounded to (0, 1): a quarterly correction cannot be negative
                # without the gap diverging, and cannot exceed one without
                # overshooting every quarter. Centred low because unemployment
                # gaps in this sample close over years, not quarters.
                "kappa_okun": {"mu": 0.10, "sigma": 0.10, "lower": 0.0, "upper": 1.0},
                "gamma_okun": dict(prior),
                "sigma_okun": {"sigma": 1.0},
            },
            constant=constant,
        )
        # THE PRODUCT IS ESTIMATED, NOT THE FACTORS. Written as
        # `-kappa x (u - u* + beta x gap)` the long-run slope appears only
        # multiplied by kappa, so the data identify `kappa x beta` while the
        # split between them runs along a curved ridge: kappa up and beta down
        # leave the likelihood unchanged. That form gave R-hat 1.35, ESS 9 and
        # 3296 divergences in 10,000. Estimating `theta = kappa x beta`
        # directly removes the ridge, since no two parameters multiply, and
        # the long-run slope comes back as a derived quantity with its own
        # posterior.
        theta = mc["beta_okun"]
        predicted = (
            -mc["kappa_okun"] * (u[:-1] - ustar[:-1])
            - theta * gap[:-1]
            # Minus, matching the long-run sign: a rising gap lowers
            # unemployment, so `gamma` is directly comparable with `beta`.
            - mc["gamma_okun"] * (gap[1:] - gap[:-1])
        )
        pm.Deterministic("beta_okun_longrun", theta / mc["kappa_okun"])
        _observe(
            "observed_u",
            predicted,
            mc["sigma_okun"],
            du,
            None if keep is None else keep[1:],
        )
    return (
        "du = -kappa x (u - u*)_{t-1} - theta x gap_{t-1} - gamma x d(gap)_t + e_o"
        "   (long run beta = theta / kappa)"
    )


def _okun_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    ustar: pt.TensorVariable,
    gap: pt.TensorVariable,
    *,
    config: ModelConfig,
    keep: np.ndarray | None,
) -> str:
    """Fit the Okun relation, in whichever form `config.okun_form` names.

    **"gap"**, u = u* - beta x gap + e_o. A level relation, and the only place
    a level of unemployment is tied to a level of output, so it is what
    identifies u* against the gap. Under the inflation-defined gap it is also
    the covariance that separates `v` from `e_c`; without it `sigma_v` returns
    its prior.

    **"ec"**, the error-correction form:

        du_t = -kappa x (u_{t-1} - u*_{t-1}) - beta x d(gap)_t + e_o

    Okun's original relates the CHANGE in unemployment to output growth, and
    `d(gap) = dy - dy*` is growth relative to potential, which is the form
    that statement takes once potential is a state. The pure difference form
    stops there, and stopping there is what makes it unusable here: changes
    against changes carry no information about where u* sits, so u* would be
    left to the Phillips curve alone. The error-correction term restores it.
    In steady state `du = 0` and `d(gap) = 0` give `u = u*`, so u* remains the
    level unemployment returns to, and `kappa` is how fast.
    """
    if config.okun_form == "ec":
        return _okun_error_correction(obs, model, ustar, gap, config=config, keep=keep)

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
    gap: pt.TensorVariable,
    anchor: np.ndarray,
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
    label = _anchor_label(anchor)
    return (
        f"pi_q = q({label}) + beta x [q(pi_exp) - q({label})]"
        " + kappa x gap + rho x d4pm + xi x GSCPI^2 + e_p"
    )


def _phillips_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    ustar: pt.TensorVariable,
    anchor: np.ndarray,
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
    label = _anchor_label(anchor)
    return (
        f"pi_q = q({label}) + beta x [q(pi_exp) - q({label})]"
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

    # Built once and carried in `obs`, so the gap identity and the Phillips
    # baseline cannot drift apart: they are the same anchor by construction.
    anchor = _anchor_series(obs, obs_index, config)
    obs = {**obs, "anchor": anchor}

    desc = scale_equation(obs, model, latents, constant=config.scale_constants)
    descriptions.append(f"Scale:        {desc}")
    if config.exclude_window is not None:
        record_constant(model, "exclude_window", config.exclude_window)
        record_constant(model, "exclude_scope", config.exclude_scope)

    desc = potential_output_equation(obs, model, latents)
    descriptions.append(f"Potential:    {desc}")

    ustar = _ustar_state(obs, model, config, obs_index)
    # Saved as a list so the pickle stays plain, and read back by `results.py`
    # for the inflation decomposition, which otherwise assumes a scalar anchor.
    record_constant(model, "anchor_series", anchor.tolist())
    nairu_state = {
        "spline": (
            f"u*_t = sum_j c_j B_j(t)   (natural cubic, knots "
            f"{', '.join(config.spline_knots)})"
        ),
        "decay": "u*_t = u*_{t-1} + phi x (u*_eq - u*_{t-1}) + e_u   (sigma imposed)",
        "walk": "u*_t = u*_{t-1} + e_u   (sigma imposed)",
    }[config.ustar_structure]
    descriptions.append(f"NAIRU:        {nairu_state}")

    gap, gap_desc = _gap_equation(obs, model, config, latents)
    descriptions.append(f"Gap:          {gap_desc}")

    # Under the identity gap this equation is `y = y* + (y - y*)`, true by
    # construction, so adding it would only drive sigma_e to zero.
    if config.gap_spec == "identity":
        descriptions.append("Output:       y = y* + gap   (definition, carries no likelihood)")
    else:
        descriptions.append(f"Output:       {_gdp_equation(obs, model, latents, gap, keep_gdp)}")

    if config.include_okun:
        descriptions.append(
            f"Okun:         {_okun_equation(obs, model, ustar, gap, config=config, keep=keep_other)}",
        )
    if config.include_phillips:
        phillips = (
            _phillips_on_gap(obs, model, gap, anchor, keep_other)
            if config.gap_spec == "cycle"
            else _phillips_equation(obs, model, ustar, anchor, keep_other)
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
        # The spline has no innovation variance, so quoting the inherited
        # value would describe a setting the run does not use.
        + (
            "u* deterministic given its coefficients"
            if config.ustar_structure == "spline"
            else f"sigma_ustar={config.sigma_ustar:g}"
        ),
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
    obs, obs_index, chart_obs, sources = build_observations(
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

    # The providers behind the observations travel with the run, so the charts
    # name what was actually loaded rather than a separately maintained string.
    constants = {**get_dictionary(model), "sources": sources.to_records()}
    save_results(
        trace, obs, obs_index,
        constants=constants,
        chart_obs=chart_obs,
        output_dir=config.output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index
