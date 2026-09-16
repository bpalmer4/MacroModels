"""Build, sample, and persist the rstar model."""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.rstar_bonds.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.rstar_bonds.observations import build_observations
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


def _stationary_ar1_prior(
    name: str,
    series: pt.TensorVariable,
    mu: pt.TensorVariable,
    rho: pt.TensorVariable,
    sigma: pt.TensorVariable,
) -> None:
    """Impose a stationary AR(1) on a series defined as an identity.

    Enters as a prior rather than a likelihood, because the series is not a
    free latent: it is data minus the state. Giving it a residual instead would
    put two free components on one observable, which is the ridge the yield
    equation already ran into once. The stationary initial draw matters as much
    as the transition, since it is what stops the first observation being free.
    """
    stationary_sd = sigma / pt.sqrt(1.0 - rho**2)
    pm.Potential(
        name,
        pm.logp(pm.Normal.dist(mu=mu, sigma=stationary_sd), series[0])
        + pm.logp(
            pm.Normal.dist(mu=mu + rho * (series[:-1] - mu), sigma=sigma),
            series[1:],
        ).sum(),
    )


def _carry_weight(rho_g: pt.TensorVariable, horizon: int, name: str) -> pt.TensorVariable:
    """Return the share of the policy gap a yield of `horizon` quarters carries.

        k = (1/H)·(1 - rho_g^H)/(1 - rho_g)

    the average of `rho_g^h` over the term. A function of the gap's own
    persistence, never asserted, so adding a maturity adds an equation without
    adding a knob. Shorter terms carry more: at rho_g = 0.85, twelve quarters
    give 0.44 against 0.16 at forty.
    """
    h = float(horizon)
    return pm.Deterministic(name, (1.0 - rho_g**h) / ((1.0 - rho_g) * h))


def _world_base(config: ModelConfig, obs: dict[str, np.ndarray], n: int) -> tuple[pt.TensorVariable, list[str]]:
    """Return the world component of r*, and a description.

    Must be called inside the model context, since it may create a variable.
    """
    if not config.use_world:
        return pt.zeros(n), ["World:      dropped (--no-world): the wedge absorbs the world level too"]

    world = pt.as_tensor_variable(obs["w"])
    if not config.free_world_loading:
        return world, []

    # Centred on the maintained hypothesis so the test is whether the data
    # move it, and wide enough that they can move it to zero if they want.
    b_world = pm.Normal("b_world", mu=1.0, sigma=1.0)
    return b_world * world, [
        "World:      r*_t = b_world · world_t + wedge_t   (loading estimated, prior N(1, 1))",
    ]


def _wedge(
    model: pm.Model,
    config: ModelConfig,
    obs_index: pd.PeriodIndex,
    n: int,
) -> tuple[pt.TensorVariable, list[str]]:
    """Return the Australia-specific wedge over world r*, and a description.

    r* is world r* plus this wedge, taking "r* is largely imported" as the
    maintained hypothesis rather than something to be discovered. World r* is
    data, so r* inherits its movement and is never smoother than it — which an
    earlier version, where r* was its own smooth random walk, got wrong.

    Must be called inside the model context, since it creates variables.
    """
    wedge_0 = pm.Normal("wedge_0", mu=0.0, sigma=2.0)

    if config.free_wedge:
        # A random walk with Student-t innovations: quiet most quarters, with
        # the occasional large move permitted, and nobody naming the dates.
        nu, nu_desc = _wedge_nu(model, config)
        eps, eps_desc = _wedge_innovations(nu, n, noncentred=config.noncentred_wedge)
        nu_desc = f"{nu_desc}, {eps_desc}"
        wedge = wedge_0 + config.sigma_walk * pt.concatenate([pt.zeros(1), pt.cumsum(eps[1:])])
        return wedge, [
            (
                f"Wedge:      random walk, StudentT innovations, "
                f"sigma={config.sigma_walk:g} imposed, {nu_desc}"
            ),
        ]

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
    flat = "" if config.wedge_drift else "   (flat in between)"
    return wedge, [f"Wedge:      steps at {breaks_desc}{flat}"]


def _premium_prior(
    config: ModelConfig,
    obs: dict[str, np.ndarray],
    tp: pt.TensorVariable,
    mc: dict[str, Any],
) -> list[str]:
    """Impose the stationary prior that identifies the permanent/transitory split.

    Two forms. The plain one makes `tp` stationary about a free `mu_tp`, which
    leaves the level of r* resting on that prior: `wedge_0` and `mu_tp` trade
    off at -0.7, so the data pin their sum and the prior picks the split. The
    answer it produced was not credible. At 2026Q3 it put the Australian real
    term premium at 1.67 against the US Kim-Wright premium of 0.82, and
    Australian neutral at 0.91 against a world 1.31, explaining a +0.56 real
    yield gap by moving the premium +0.84 and neutral -0.40.

    With `us_premium_anchor`, `tp` must instead track the published US premium
    and only the spread over it is estimated. The asserted quantity becomes the
    average Australian premium *over* the US one, a liquidity claim about thin
    indexed AGS against TIPS: small, arguable, and checkable against something
    outside the model.

    Two mismatches remain. Kim-Wright is a nominal term premium while `tp` is
    real, so the inflation risk premium stays on the Australian side. And a US
    premium stands in for a global one.

    With `au_premium_anchor`, `tp` tracks the AOFM's published AUSTRALIAN
    premium instead, which removes the second of those and leaves the first as
    the entire residual. That is the better-argued version of the same device:
    the asserted quantity becomes "the average gap between a real and a nominal
    Australian term premium", which is an inflation risk premium and nothing
    else, rather than a liquidity spread, a currency risk premium and an
    inflation risk premium bundled together.

    It is also a test rather than a convenience. The plain form makes `tp` move
    0.30 points across the sample while the AOFM premium moves 1.99 and the
    wedge moves -2.29, so if the long yield's decline was premium, the plain
    form has been booking it as r*.

    Must be called inside the model context, since it creates a Potential.
    """
    if config.au_premium_anchor:
        spread = pm.Deterministic("tp_spread", tp - pt.as_tensor_variable(obs["au_tp"]))
        _stationary_ar1_prior("tp_prior", spread, mc["mu_spread"], mc["rho_tp"], mc["sigma_tp"])
        return ["Premium:    tp_t = au_tp_t + spread_t,  spread ~ AR(1) about mu_spread"]

    if config.us_premium_anchor:
        spread = pm.Deterministic("tp_spread", tp - pt.as_tensor_variable(obs["us_tp"]))
        _stationary_ar1_prior("tp_prior", spread, mc["mu_spread"], mc["rho_tp"], mc["sigma_tp"])
        return ["Premium:    tp_t = us_tp_t + spread_t,  spread ~ AR(1) about mu_spread"]

    _stationary_ar1_prior("tp_prior", tp, mc["mu_tp"], mc["rho_tp"], mc["sigma_tp"])
    return ["Premium:    tp ~ AR(1) about mu   (stationary: the identifying prior)"]


def _window_one(
    config: ModelConfig,
    obs: dict[str, np.ndarray],
    r_star: pt.TensorVariable,
    carried: pt.TensorVariable,
    mc: dict[str, Any],
) -> list[str]:
    """Attach the long-end observable, in whichever of its two forms is in use.

    DEFAULT: the indexed real yield, split by an identity into r* and a latent
    premium, with the premium's stationarity carrying the identification. The
    identity has no residual deliberately — an earlier version gave it one and
    `sigma_y` collapsed toward zero with `mu_tp` riding a ridge against the
    level of r*, because two free things were explaining one series.

    `nominal_window`: the AOFM's risk-neutral yield, deflated. The premium has
    already been removed from the observable, so there is nothing for a latent
    premium to do and the whole `tp` block goes. What remains is an ordinary
    observation equation: a risk-neutral real yield IS the expected real policy
    path, which is r* plus the carried share of today's gap.

    That form needs the residual the default cannot have. Here nothing competes
    with it for the same variance, and it has real work: `sigma_rn` absorbs
    AOFM's estimation error and the deflator's horizon mismatch, and it is the
    only thing that can.

    What the second form buys: `mu_tp` is gone, so the -0.87 correlation with
    `wedge_0` that IS this model's level problem has nothing to attach to. The
    level then rests on the stationarity of `g` alone.

    Must be called inside the model context, since it creates variables.
    """
    if config.nominal_window:
        sigma_rn = pm.HalfNormal("sigma_rn", sigma=config.sigma_rn_sigma)
        pm.Normal(
            "y_rn_obs",
            mu=r_star + carried,
            sigma=sigma_rn,
            observed=pt.as_tensor_variable(obs["y"]),
        )
        carried_term = " + k·g_t" if config.use_short else ""
        return [
            f"Yield:      y_rn_t = r*_t{carried_term} + e_t   (premium removed by AOFM)",
            "Premium:    none in the model   (no mu_tp, so no level trade-off)",
        ]

    tp = pm.Deterministic("tp", pt.as_tensor_variable(obs["y"]) - r_star - carried)
    carried_term = " - k·g_t" if config.use_short else ""
    descriptions = [f"Yield:      tp_t = y_t - r*_t{carried_term}   (identity, no residual)"]

    # The stationarity of that premium is what identifies the split, so it
    # enters as a prior on tp rather than as a likelihood: a stationary initial
    # draw, then an AR(1) transition. rho is bounded below 1 — a unit root would
    # make the premium a second random walk and the decomposition meaningless.
    descriptions.extend(_premium_prior(config, obs, tp, mc))
    return descriptions


def _forward_window(config: ModelConfig, obs: dict[str, np.ndarray], r_star: pt.TensorVariable) -> list[str]:
    """Attach the market's deflated 5y5y forward as a window on r* itself.

    `f_t = r*_t + bias + e_t`. Unlike the other two windows this one loads on r*
    with no premium and no policy gap in the way, so it speaks to the LEVEL.

    The bias is what the forward carries that r* does not: the market's view of
    the cycle over years five to ten, and whatever premium AOFM did not strip.
    It is free but tightly priored, and it now shares the level-carrying job with
    `mu_spread` rather than replacing it. Widen the prior and the level goes back
    to being unidentified.

    Must be called inside the model context, since it creates variables.
    """
    if not config.use_forward or "f" not in obs:
        return []
    # IMPOSED when the prior sd is zero, which is not the default: see
    # `ModelConfig.forward_bias_sigma` for the test. Imposing it costs 35
    # divergences against the free bias's 22, because the bias is absorbing a
    # genuine disagreement between the forward and the other two windows about
    # the level, and removing it pushes that disagreement onto the wedge walk.
    if config.forward_bias_sigma > 0:
        bias = pm.Normal("forward_bias", mu=config.forward_bias_mu, sigma=config.forward_bias_sigma)
    else:
        bias = pt.as_tensor_variable(float(config.forward_bias_mu))
    # HalfNormal, and a ZERO-AVOIDING PRIOR WAS TRIED HERE AND IS WORSE.
    #
    # The case for trying one was that the residual divergences have a single
    # signature: divergent draws sit at `sigma_f` 0.059 against a posterior mean
    # of 0.102, a shift of -2.06 sd and much the largest of any parameter, with
    # `wedge_0` +0.98, `forward_bias` -0.87 and `mu_spread` -0.79 moving with
    # it. As `sigma_f` approaches zero the forward stops being an observation
    # and becomes the constraint r* = f - bias, which the three level parameters
    # can only satisfy by being squeezed together. HalfNormal has its mode at
    # exactly that point, and `sigma_f` cannot really be zero: it carries AOFM's
    # own estimation error and the mismatch between a 5y5y horizon and the
    # deflator's.
    #
    # InverseGamma(3, 0.2), mean 0.10, gave 15 divergences against 5, BFMI 0.21
    # against 0.32, `ess_bulk` 501 against 712 and `r_hat` 1.0097 against
    # 1.0025. Zero density at the boundary is bought with a sharp log-density
    # barrier beside it, and that curvature is harder to traverse than the
    # smooth approach it replaced: a wall in place of a funnel.
    #
    # The experiment settled something else worth keeping. The posterior did NOT
    # move: `sigma_f` 0.102 -> 0.093, `mu_spread` 0.232 -> 0.237, `wedge_0`
    # 2.440 -> 2.443, r* 1.005 -> 1.001 under two very different priors. So the
    # likelihood pins this residual, the low-`sigma_f` region is a real feature
    # of the geometry rather than something the prior invited, and the residual
    # divergences are the forward window genuinely disagreeing with the other
    # two about the level. That is a finding about the model, not the sampler.
    sigma_f = pm.HalfNormal("sigma_f", sigma=config.sigma_f_sigma)
    pm.Normal("obs_forward", mu=r_star + bias, sigma=sigma_f, observed=obs["f"])
    return ["Forward:    f_t = r*_t + bias + e_t   (AOFM 5y5y, deflated)"]


def _policy_gap(
    model: pm.Model,
    config: ModelConfig,
    obs: dict[str, np.ndarray],
    r_star: pt.TensorVariable,
) -> tuple[pt.TensorVariable, pt.TensorVariable, list[str]]:
    """Return the policy gap, its persistence, and a description.

    A long real yield is roughly the average expected real short rate over its
    term, plus a premium. If the short rate reverts to r* with persistence
    `rho_g`, that average puts weight

        k = (1/H)·(1 - rho_g^H)/(1 - rho_g)

    on today's gap and the rest on r*. So `k` is a function of the gap's own
    persistence rather than a free parameter: the same `rho_g` that governs the
    stationary prior fixes how much of the gap the long yield carries. That is
    the overidentifying restriction the second window buys, and it costs no new
    knob.

    Note what this does *not* do. Two observables and three free levels
    (`wedge_0`, `mu_tp`, `mu_g`) still leave a flat direction: add d to r*, take
    d off both means, and every fitted value is unchanged. One mean must still
    be asserted. What changes is that the assertion is now checkable against the
    other window instead of only having to look plausible.

    Must be called inside the model context, since it creates variables.
    """
    mg = set_model_coefficients(model, {
        "mu_g": (
            {"mu": config.mu_g_mu, "sigma": config.mu_g_sigma}
            if config.assert_stance
            else {"mu": 0.0, "sigma": 3.0}
        ),
        "rho_g": {"mu": 0.85, "sigma": 0.1, "lower": 0.0, "upper": 0.98},
        "sigma_g": {"sigma": 1.0},
    })

    g = pm.Deterministic("g", pt.as_tensor_variable(obs["r"]) - r_star)
    _stationary_ar1_prior("g_prior", g, mg["mu_g"], mg["rho_g"], mg["sigma_g"])

    stance = "asserted" if config.assert_stance else "free"
    descriptions = [
        "Cash:       g_t = r_t - r*_t   (identity, no residual)",
        f"Stance:     g ~ AR(1) about mu_g   ({stance})",
    ]
    return g, mg["rho_g"], descriptions


def build_model(
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    config: ModelConfig | None = None,
    verbose: bool = True,
) -> pm.Model:
    """Build the rstar PyMC model: one state, two windows onto it.

    The term premium is a stationary AR(1) and the natural rate is a random
    walk whose innovation sd is imposed. That pairing identifies the *movements*:
    a permanent component and a transitory one have different spectra, so the
    split is identified in principle by persistence, and in practice it leans on
    `sigma_r`, which is why the world equation matters.

    It does not identify the *level*. From one series the starting point of a
    random walk and the mean of a stationary process are not separable, which
    is the -0.76 correlation between `wedge_0` and `mu_tp`. The real cash rate
    is a second window on the same state: the long yield is roughly the average
    expected short rate over its term plus a premium, so the short rate speaks
    to r* directly. One of the two means must still be asserted. The gain is
    that the other is then reported, and can be checked.
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

        # `mu_tp` is the level. With one observable it was the *only* thing
        # speaking to it, since `tp = y - r*` makes a prior on the premium a
        # prior on r*: the 2026Q2 headline of 1.24 is a yield of 2.50 less a
        # premium of 1.26 that the prior essentially chose. Its posterior was
        # 0.976 [-0.67, 2.75] against a N(0.75, 1.0) prior whose own interval
        # is [-0.89, 2.39] — the mean moved 0.23 and the interval got *wider*,
        # because the trade-off with `wedge_0` spreads it faster than the
        # likelihood tightens it.
        #
        # The second window does not make it estimable, it makes the choice of
        # which mean to assert available. Asserted here by default, so this run
        # stays comparable to the one-window vintage; freed when the stance is
        # asserted instead, which is when `mu_tp` becomes an output to check
        # against an external term premium estimate.
        mu_tp_prior = {"mu": 0.75, "sigma": 3.0 if config.assert_stance else 1.0}
        # Under the nominal window there is no term premium in the model at all,
        # so none of these are created. Leaving them in would put three
        # parameters in the trace that nothing in the likelihood touches, which
        # would sample their priors and make the diagnostics harder to read.
        settings: dict[str, dict[str, float]] = {} if config.nominal_window else {
            "mu_tp": mu_tp_prior,
            "rho_tp": {"mu": 0.8, "sigma": 0.2, "lower": 0.0, "upper": 0.98},
            "sigma_tp": {"sigma": 1.0},
        }
        if (config.us_premium_anchor or config.au_premium_anchor) and not config.nominal_window:
            # The asserted level moves from "the average Australian term
            # premium", which nothing outside the model speaks to, to a spread
            # over a published series, which a reader can dispute on its own
            # terms. `mu_tp` stays in the model as a reported quantity.
            #
            # Under the AU pin the spread is real-minus-nominal on the same
            # curve, so it is an inflation risk premium and the prior centre is
            # still a small positive number: the same N(0.25, 0.5) serves, and
            # it is left shared deliberately, so the two pins differ in which
            # series they track rather than in how hard they are pushed.
            settings["mu_spread"] = {"mu": config.mu_spread_mu, "sigma": config.mu_spread_sigma}
        mc = set_model_coefficients(model, settings)

        # --- The Australian wedge: the only state ---
        base, base_desc = _world_base(config, obs, n)
        descriptions.extend(base_desc)
        wedge, wedge_desc = _wedge(model, config, obs_index, n)
        descriptions.extend(wedge_desc)

        wedge = pm.Deterministic("wedge", wedge)
        r_star = pm.Deterministic("r_star", base + wedge)
        descriptions.append("State:      r*_t = world r*_t + wedge_t")

        # --- The policy gap: the second window on the same state ---
        # Declared before the branch: both are absent in the one-window model,
        # and the third window tests for that rather than assuming they exist.
        gap: pt.TensorVariable | None
        rho_g: pt.TensorVariable | None
        if config.use_short:
            gap, rho_g, gap_desc = _policy_gap(model, config, obs, r_star)
            descriptions.extend(gap_desc)
            carried = _carry_weight(rho_g, config.horizon_quarters, "k") * gap
            descriptions.append(
                f"Weight:     k = (1-rho_g^H)/((1-rho_g)H), H={config.horizon_quarters}",
            )
        else:
            gap = rho_g = None
            carried = pt.zeros(n)
            descriptions.append("Cash:       dropped (--no-short): the one-window model")

        # --- The term premium: defined, not fitted ---
        # tp is what the yield leaves over once r* is taken out, so the yield
        # equation is an identity and carries no residual. An earlier version
        # gave it one: with r* and tp both free to explain a single series the
        # noise had nothing to do, sigma_y collapsed toward zero (posterior mean
        # 0.035, ess 11, r_hat 1.30) and mu_tp rode a ridge against the level of
        # r*. Removing the redundant parameter removes the ridge.
        # Subtracting `k·g` is not only about the level. The one-window version,
        # `tp = y - r*`, sets k = 0: it assumes the long yield carries nothing
        # from where the short rate currently sits. It cannot, so that premium
        # was really `true tp + k·g`, and `g` was extraordinarily negative
        # through the QE window. On the previous vintage, 2021Q4 had a real cash
        # rate of -2.25 against r* of -0.14, so g = -2.11, while the fitted tp
        # was -0.40. Adding k·g back at rho_g = 0.9 gives k = 0.246 and a
        # corrected premium of +0.13; across rho_g of 0.8 to 0.95 the negative
        # premium in that window is erased or reversed.
        #
        # That matters because the negative premium under QE was the model's
        # best external check: an event it was never told about, landing where
        # a bond-market model ought to put it. If it survives this correction
        # the check stands. If it does not, the check was the model reading a
        # floored cash rate through a missing coefficient. Either way the
        # question is now answerable, which it was not before.
        descriptions.extend(_window_one(config, obs, r_star, carried, mc))
        descriptions.extend(_forward_window(config, obs, r_star))

        # --- The third window: a medium maturity on the same state ---
        # The same r*, a different share of the same policy gap, and its own
        # premium. What this adds is not another level but a *slope*: the two
        # premia share a state, so `mu_tp - mu_tp_m` is identified even though
        # neither level is.
        if config.use_curve and rho_g is not None and gap is not None:
            mm = set_model_coefficients(model, {
                "mu_tp_m": {"mu": 0.35, "sigma": 1.0},
                "rho_tp_m": {"mu": 0.8, "sigma": 0.2, "lower": 0.0, "upper": 0.98},
                "sigma_tp_m": {"sigma": 1.0},
            })
            k_m = _carry_weight(rho_g, config.curve_horizon_quarters, "k_m")
            tp_m = pm.Deterministic(
                "tp_m", pt.as_tensor_variable(obs["m"]) - r_star - k_m * gap,
            )
            _stationary_ar1_prior(
                "tp_m_prior", tp_m, mm["mu_tp_m"], mm["rho_tp_m"], mm["sigma_tp_m"],
            )
            pm.Deterministic("tp_slope", mc["mu_tp"] - mm["mu_tp_m"])
            descriptions.append(
                f"Curve:      tp_m = m_t - r*_t - k_m·g_t, H={config.curve_horizon_quarters} "
                f"({config.curve_maturity}y)",
            )
            descriptions.append("Slope:      mu_tp - mu_tp_m identified even though neither level is")


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
    prefix: str = "rstar_bonds",
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


def _print_window_spec(config: ModelConfig) -> None:
    """Print which long-end observable is in use, and which level is asserted.

    One of the two means must be asserted in every form of this model, so the
    run log names it rather than leaving the reader to infer it from the flags.
    """
    if config.nominal_window:
        print(
            f"Window one:   AOFM risk-neutral {config.au_premium_maturity}y "
            f"({config.au_premium_source}), deflated — no term premium in the model",
        )
    elif config.au_premium_anchor:
        print(
            f"AU premium:   pinned to AOFM {config.au_premium_maturity}y "
            f"({config.au_premium_source}); only the real-nominal spread is estimated",
        )

    if not config.use_short:
        print("Short rate:   none (--no-short): one-window model, mu_tp carries the level")
        return

    if config.nominal_window:
        asserted = "stance carries the level alone (no mu_tp)"
    elif config.au_premium_anchor or config.us_premium_anchor:
        asserted = f"mu_spread ~ N({config.mu_spread_mu:g}, {config.mu_spread_sigma:g}), stance free"
    elif config.assert_stance:
        asserted = f"stance ~ N({config.mu_g_mu:g}, {config.mu_g_sigma:g}), mu_tp free"
    else:
        asserted = "mu_tp ~ N(0.75, 1), stance free"

    short_label = "real 90d bank bill" if config.short_rate == "bill" else "real cash"
    print(f"Short rate:   {short_label}, deflator={config.deflator}, H={config.horizon_quarters}")
    print(f"Level:        {asserted}")


def run_estimate(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    prefix: str = "rstar_bonds",
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
    if config.us_premium_anchor or config.world_source == "market":
        # Only say it where a US premium is actually subtracted; on the default
        # cleveland anchor with no pin, nothing reads it.
        print(f"US premium:   {config.us_premium_source}")
    _print_window_spec(config)
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
    obs, obs_index, chart_obs, sources = build_observations(
        start=config.start,
        end=config.end,
        world_source=config.world_source,
        deflator=config.deflator,
        short_rate=config.short_rate,
        us_premium_anchor=config.us_premium_anchor,
        us_premium_source=config.us_premium_source,
        use_curve=config.use_curve,
        curve_maturity=config.curve_maturity,
        au_premium_anchor=config.au_premium_anchor,
        au_premium_source=config.au_premium_source,
        au_premium_maturity=config.au_premium_maturity,
        nominal_window=config.nominal_window,
        use_forward=config.use_forward,
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

    # The providers behind the observations travel with the run, so the charts
    # name what was actually loaded rather than a separately maintained string.
    save_results(
        trace, obs, obs_index,
        constants={**get_fixed_constants(model), "sources": sources.to_records()},
        chart_obs=chart_obs,
        output_dir=config.output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index
