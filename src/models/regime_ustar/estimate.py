"""Build, sample and persist the regime u* model."""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from src.models.common.model_constants import attach
from src.models.common.spline import basis
from src.models.regime_ustar.config import ModelConfig
from src.models.ystar.base import SamplerConfig, sample_model


def _ustar_spline(frame: pd.DataFrame, config: ModelConfig) -> Any:
    """u* as a natural cubic spline with knots at the regime dates.

        ustar_t = sum_j c_j B_j(t)

    Deterministic given the coefficients, so there is no innovation variance
    to impose. That removes `sigma_ustar`, which nothing measured and which
    is what the deleted long-run model died on.

    The basis is a partition of unity, its rows summing to 1 at machine
    precision, so `c_j` are in unemployment-rate units and take the same prior
    the attractors did. A coefficient IS roughly the level u* passes through
    near its knot, rather than an abstract weight.

    Natural, so the curve is linear beyond the outer knots. The last segment
    runs 2020Q1 to the end with no knot after it and the largest inflation
    surprises in the sample under it, and a free cubic extrapolating at a
    boundary is the worst case for this basis.
    """
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError("the observation frame must carry a quarterly PeriodIndex")
    design = basis(
        index, config.breaks,
        natural=config.natural_spline,
        multiplicity=config.knot_multiplicity,
    )
    mu0, sd0, lo, hi = config.eq_prior
    coef = pm.TruncatedNormal("coef", mu=mu0, sigma=sd0, lower=lo, upper=hi, shape=design.shape[1])
    return pm.Deterministic("ustar", pt.dot(pt.as_tensor_variable(design), coef))


def _ustar_state(frame: pd.DataFrame, regimes: np.ndarray, config: ModelConfig) -> Any:
    """u* as ONE continuous state whose law of motion changes at each regime.

        u*_t = u*_{t-1} + phi_k x (eq_k - u*_{t-1}) + e_t,   k = regime(t)

    Level continuity across a regime boundary is structural, not imposed:
    there is one state and it is carried forward, so nothing can jump at a
    date. What changes at the boundary is where u* is being pulled and how
    hard, which is a change in direction rather than in position.

    The innovation sd is imposed. This is the dial that sank the deleted
    long-run model, and leaving it free to a weak Phillips likelihood is how
    u* becomes the unemployment rate again. The attractors are what is meant
    to move u*; `sigma_ustar` is only meant to let it wander a little around
    the path they set.

    With `config.adaptive_sigma` the scale is no longer flat across the
    sample: it is keyed to how far the one-year average of unemployment has
    moved over the preceding two years, so u* may innovate more in 1974-75,
    1990-91 and 2020 than in the quiet stretches. `kappa` = 0 recovers the
    constant-sigma model exactly, so the posterior on `kappa` tests the idea.
    See `ModelConfig.adaptive_sigma` for the objection to it.

    Non-centred innovations, as everywhere else in this package.
    """
    n = len(frame)
    n_regimes = int(regimes.max()) + 1
    mu0, sd0, lo, hi = config.eq_prior
    pmu, psd, plo, phi_hi = config.phi_prior

    eq = pm.TruncatedNormal("eq", mu=mu0, sigma=sd0, lower=lo, upper=hi, shape=n_regimes)
    phi_shape = n_regimes if config.free_phi_per_regime else 1
    phi = pm.TruncatedNormal("phi", mu=pmu, sigma=psd, lower=plo, upper=phi_hi, shape=phi_shape)

    # The innovation sd, constant or keyed to how much unemployment has moved.
    if config.adaptive_sigma:
        move = np.asarray(frame["u_move"], dtype=float)
        rel = np.maximum(config.move_floor, move / np.nanmean(move))
        kmu, ksd, klo, khi = config.kappa_prior
        kappa = pm.TruncatedNormal("kappa", mu=kmu, sigma=ksd, lower=klo, upper=khi)
        sigma_t = config.sigma_ustar * pt.power(pt.as_tensor_variable(rel), kappa)
        pm.Deterministic("sigma_ustar_t", sigma_t)
    else:
        sigma_t = pt.as_tensor_variable(np.full(n, config.sigma_ustar))

    z = pm.Normal("z_ustar", mu=0.0, sigma=1.0, shape=n)
    init = pm.Normal("ustar_init", mu=float(frame["u"].iloc[0]), sigma=3.0)

    # Per-quarter attractor and speed, so the scan carries no regime logic.
    eq_t = eq[regimes]
    phi_t = phi[regimes] if config.free_phi_per_regime else pt.repeat(phi, n)

    def step(target: Any, speed: Any, eps: Any, prev: Any) -> Any:
        return prev + speed * (target - prev) + eps

    path, _ = pytensor.scan(
        fn=step,
        sequences=[eq_t[1:], phi_t[1:], z[1:] * sigma_t[1:]],
        outputs_info=[init],
    )
    return pm.Deterministic("ustar", pt.concatenate([[init], path]))


def _wage_equation(frame: pd.DataFrame, ustar: Any, u_obs: np.ndarray, config: ModelConfig) -> None:
    """Add the second observation equation, on unit labour costs.

        ulc_yoy - pi_e = alpha + gamma x (u - ustar)/u + lambda x dU/U + v

    This is the only part of the model that attacks the circularity. The price
    equation alone makes `u - ustar` the inflation gap scaled by `u/beta`, so
    no labour-market observable enters except `u` itself. Unit labour costs are
    a labour-market price, so the same `ustar` now has a second thing to answer
    to, and the two equations can disagree.

    Form follows `nairu/equations/phillips_wage.py`: the same proportional gap,
    expected inflation entering with a coefficient of one (so the dependent
    variable is real ULC growth), a free intercept absorbing trend productivity
    rather than netting it off, and the speed-limit term on the CHANGE in
    unemployment. Written year-ended rather than `nairu`'s quarterly, so
    `gamma` and the price equation's `beta` are in the same units and their
    agreement is a diagnostic rather than a units comparison.

    `gamma` is constrained non-positive, as `nairu` constrains its own: the
    content of a wage Phillips curve is that slack slows wage growth.
    """
    gmu, gsd, gupper = config.gamma_wage_prior
    lmu, lsd = config.lambda_wage_prior

    alpha = pm.Normal("alpha_wage", mu=0.0, sigma=config.alpha_wage_prior_sd)
    gamma = pm.TruncatedNormal("gamma_wage", mu=gmu, sigma=gsd, upper=gupper)
    lam = pm.Normal("lambda_wage", mu=lmu, sigma=lsd)
    sigma_w = pm.HalfNormal("sigma_wage", sigma=config.sigma_wage_prior_sd)

    mu_w = (
        alpha
        + gamma * (u_obs - ustar) / u_obs
        + lam * np.asarray(frame["speed"], dtype=float)
    )
    pm.Normal(
        "ulc_obs",
        mu=mu_w,
        sigma=sigma_w,
        observed=np.asarray(frame["ulc"] - frame["pi_e"], dtype=float),
    )


def _controls(frame: pd.DataFrame, config: ModelConfig) -> Any:
    """Return the supply and terms-of-trade terms added to the Phillips curve.

    The GSCPI term is squared and sign-preserving, so pressure matters more
    than proportionally and an easing of supply chains is not treated as its
    mirror. Zero when neither control is on.
    """
    total: Any = 0.0
    if config.supply_control:
        rho = pm.Normal("rho_pi", mu=0.0, sigma=config.rho_prior_sd)
        xi = pm.Normal("xi_gscpi", mu=0.0, sigma=config.xi_prior_sd)
        gscpi = np.asarray(frame["gscpi"], dtype=float)
        total = total + rho * np.asarray(frame["d4pm"], dtype=float) + xi * gscpi**2 * np.sign(gscpi)
    if config.tot_control:
        gamma = pm.Normal("gamma_tot", mu=0.0, sigma=config.tot_prior_sd)
        total = total + gamma * np.asarray(frame["tot"], dtype=float)
    return total


def _okun_equation(frame: pd.DataFrame, ustar: Any, config: ModelConfig) -> None:
    """Add the output observation on u*, as error correction.

        du_t = a + b x dy_t + lambda x (u_{t-1} - u*_{t-1}) + e

    Unemployment falls when output grows, and separately drifts back toward
    u* when it is away from it. `lambda` is that pull, so it should be
    NEGATIVE; the prior is two-sided so the posterior reports whether the data
    show an error correction rather than being told there is one.

    **This is the only equation here that attacks the circularity.** The
    Phillips curve alone makes `u - u*` the inflation gap scaled by `u/beta`,
    so nothing but `u` enters, on both sides. Real GDP growth is a second
    observable and can disagree.

    `u*` is lagged to match `u_{t-1}`, which costs the equation nothing: the
    first quarter has no lag and is already absent from the frame.
    """
    b_mu, b_sd = config.b_okun_prior
    l_mu, l_sd = config.lambda_okun_prior

    a = pm.Normal("a_okun", mu=0.0, sigma=config.a_okun_prior_sd)
    b = pm.Normal("b_okun", mu=b_mu, sigma=b_sd)
    lam = pm.Normal("lambda_okun", mu=l_mu, sigma=l_sd)
    sigma_o = pm.HalfNormal("sigma_okun", sigma=config.sigma_okun_prior_sd)

    u_lag = np.asarray(frame["u_lag"], dtype=float)
    mu_o = a + b * np.asarray(frame["dy"], dtype=float) + lam * (u_lag - pt.concatenate([ustar[:1], ustar[:-1]]))
    pm.Normal("du_obs", mu=mu_o, sigma=sigma_o, observed=np.asarray(frame["du"], dtype=float))


def build_model(frame: pd.DataFrame, regimes: np.ndarray, config: ModelConfig) -> pm.Model:
    """Return the state law above, observed through one accelerationist Phillips curve.

        pi_t - pi^e_t = -beta x (u_t - u*_t) / u_t + supply_t + e_t

    and, when `config.wage_equation`, a SECOND observation on the same u*
    built from unit labour costs. See `_wage_equation`: without it the
    unemployment gap is algebraically the inflation gap rescaled, and no
    labour-market observable enters except u itself.

    written in LEVELS, with `pi^e` formed by attention that switches on above
    6 per cent inflation until 1983, then the expectations model's own series.
    See `observations.salience_expectation`.

    That is what lets the equation say something about a re-anchoring. The
    accelerationist form cannot: when expectations are collapsing onto a new
    target, inflation stops changing for a nominal reason rather than a real
    one, and `Delta pi` near zero reads as equilibrium whatever unemployment is
    doing. `long_run_ustar/MODEL_NOTES.md` diagnoses exactly that for 1993, and
    it is why every method in the package returns about 10.7 there.

    Residual scale is separate either side of the handoff when
    `config.regime_sigma`, because inflation less an adaptive expectation and
    inflation less a measured one are different objects with different noise.
    """
    with pm.Model() as model:
        ustar = (
            _ustar_spline(frame, config)
            if config.state == "spline"
            else _ustar_state(frame, regimes, config)
        )

        # Positive by construction: the Phillips curve's content is that slack
        # slows inflation. Left free to change sign, a weakly identified
        # equation would report a sign rather than a level, and the level is
        # what is being asked for. The restriction is an assumption, not a
        # finding, and `beta` near zero is how the model says the data did not
        # support it.
        # One slope, or one per beta group, broadcast to a slope per quarter.
        # Each group carries the same prior: the groups say the slope may
        # differ by era, not which era is steep.
        if config.beta_groups:
            groups = np.asarray(config.beta_groups, dtype=int)[regimes]
            beta_by_group = pm.HalfNormal("beta", sigma=config.beta_prior_sd, shape=int(groups.max()) + 1)
            beta = beta_by_group[groups]
        else:
            beta = pm.HalfNormal("beta", sigma=config.beta_prior_sd)
        measured = np.asarray(frame["measured"], dtype=int)
        if config.regime_sigma and 0 < measured.sum() < len(measured):
            sigma_both = pm.HalfNormal("sigma", sigma=config.sigma_prior_sd, shape=2)
            sigma = sigma_both[measured]
        else:
            sigma = pm.HalfNormal("sigma", sigma=config.sigma_prior_sd)

        # The PROPORTIONAL gap, (u - u*) / u, as `ustar/estimate.py:212` and
        # `ystar_ustar/estimate.py:411` both use. Convex in u, which matters
        # here more than it does there.
        #
        # Inverting gives u* = u x (1 + (pi - pi^e)/beta), so the factor that
        # turns an inflation surprise into an unemployment statement is u/beta
        # rather than 1/beta. It therefore SCALES WITH THE UNEMPLOYMENT RATE:
        # small when the labour market is tight, large when it is slack. Under
        # the level gap a +0.28 average surprise in the 1960s became +0.60 on
        # u* at an unemployment rate of 1.9, and the 2022-23 surprises of +3.4
        # to +5.2 became gaps of 7 to 11 points. Both are amplifications the
        # level form applies regardless of where unemployment actually was.
        #
        # There is evidence for convexity beyond consistency: `ystar_ustar`'s
        # notes split its fitted sample and find a slope of -1.82 on the tight
        # side against -0.45 on the slack side.
        u_obs = np.asarray(frame["u"], dtype=float)
        mu = -beta * (u_obs - ustar) / u_obs

        # A free constant for the named regimes, zero everywhere else. `which`
        # sends every other quarter to a zero appended past the end of alpha,
        # so no regime logic reaches the likelihood.
        if config.intercept_regimes:
            order = {k: j for j, k in enumerate(sorted(config.intercept_regimes))}
            which = np.array([order.get(int(r), len(order)) for r in regimes], dtype=int)
            alpha = pm.Normal("alpha", mu=0.0, sigma=config.intercept_prior_sd, shape=len(order))
            mu = mu + pt.concatenate([alpha, pt.zeros(1)])[which]
        mu = mu + _controls(frame, config)

        if config.wage_equation:
            _wage_equation(frame, ustar, u_obs, config)

        if config.okun_equation:
            _okun_equation(frame, ustar, config)

        observed = np.asarray(frame["surprise"], dtype=float)

        # An AR(1) error, written as a conditional mean rather than a state.
        # With e_t = phi e_{t-1} + eta_t and e_t = surprise_t - mu_t, the
        # previous error is known once the parameters are, so
        #
        #     E[surprise_t | t-1] = mu_t + phi x (surprise_{t-1} - mu_{t-1})
        #
        # and `sigma` becomes the scale of the INNOVATION rather than of the
        # residual. Costs the first quarter, which has no lagged error.
        #
        # The residual of the static form is autocorrelated at +0.63 overall
        # and +0.60 to +0.88 inside every regime, so the equation was treating
        # a persistent component as independent noise. Year-ended inflation is
        # persistent and the expectation it is measured against is a slow
        # trend, so the surprise inherits that persistence and nothing in the
        # Phillips curve accounted for it.
        if config.ar1_error:
            pmu, psd = config.phi_e_prior
            phi_e = pm.TruncatedNormal("phi_e", mu=pmu, sigma=psd, lower=-0.99, upper=0.99)
            mu = mu[1:] + phi_e * (observed[:-1] - mu[:-1])
            observed = observed[1:]
            if getattr(sigma, "ndim", 0) == 1:
                sigma = sigma[1:]  # a per-quarter scale must lose the same quarter

        if config.student_t:
            nu = pm.Exponential("nu_minus_two", lam=1.0 / config.nu_prior) + 2.0
            pm.Deterministic("nu", nu)
            pm.StudentT("dpi_obs", nu=nu, mu=mu, sigma=sigma, observed=observed)
        else:
            pm.Normal("dpi_obs", mu=mu, sigma=sigma, observed=observed)

    attach(model, config.constants)
    return model


def estimate(
    frame: pd.DataFrame,
    regimes: np.ndarray,
    labels: list[str],
    config: ModelConfig,
    sampler: SamplerConfig | None = None,
) -> az.InferenceData:
    """Sample the model and attach what is needed to read the trace back."""
    trace = sample_model(build_model(frame, regimes, config), sampler)
    posterior = getattr(trace, "posterior", None)
    if isinstance(posterior, xr.Dataset):
        posterior.attrs["regime_labels"] = labels
    return trace


def save_trace(trace: az.InferenceData, frame: pd.DataFrame, labels: list[str], config: ModelConfig) -> Path:
    """Write the trace and the frame it was fitted to, side by side."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"{config.prefix}_trace.nc"
    trace.to_netcdf(str(path))
    with (config.output_dir / f"{config.prefix}_data.pkl").open("wb") as handle:
        pickle.dump({"frame": frame, "labels": labels, "constants": config.constants}, handle)
    return path


def load_trace(
    config: ModelConfig,
) -> tuple[az.InferenceData, pd.DataFrame, list[str], dict[str, object]]:
    """Read back a saved run, for charting without re-sampling.

    The constants come back too: they are the specification the trace was
    estimated under, and charting it under anything else is reading the wrong
    model. See `ModelConfig.with_saved_settings`.
    """
    trace = az.from_netcdf(str(config.output_dir / f"{config.prefix}_trace.nc"))
    with (config.output_dir / f"{config.prefix}_data.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    return trace, saved["frame"], saved["labels"], saved.get("constants", {})
